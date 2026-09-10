#pragma once

// ============================================================
//  Embedding.hpp, Token embeddings + positional encodings
//
//  This is the first component with learned parameters.
//  It converts a sequence of TokenIDs into a float matrix
//  that the transformer layers can operate on.
//
//  Two parts
//  ─────────
//  1. Token embedding table  E   [vocab_size × d_model]
//     A learned weight matrix.  Row i is the embedding vector
//     for token ID i.  Initialised with Xavier uniform.
//     <PAD> (ID 0) is always the zero vector and stays that
//     way, padding should contribute nothing.
//
//  2. Positional encoding     PE  [max_len × d_model]
//     Fixed (not learned) sinusoidal encodings.  Row p gives
//     the positional signal for sequence position p.
//     Formula (Vaswani et al. 2017):
//       PE[p, 2i]   = sin(p / 10000^(2i/d_model))
//       PE[p, 2i+1] = cos(p / 10000^(2i/d_model))
//
//  Forward pass
//  ────────────
//  Given ids = [t0, t1, ..., t_{L-1}]   // [seq_len]
//
//    X[p] = E[t_p] + PE[p]              // [d_model]
//
//  Output X has shape [seq_len × d_model].
//  Padding positions (t_p == kPAD_ID) produce the zero vector
//  because E[0] == 0 and we zero out PE at those positions too.
//
//  Backward pass
//  ─────────────
//  Only E is learned; PE is fixed.
//  The gradient w.r.t. E is a sparse update: for each position
//  p in the sequence, dL/dE[t_p] += dL/dX[p].
//  accumulate_grad() does this accumulation.
//
//  Dimensions (compile-time defaults, runtime-configurable)
//  ─────────────────────────────────────────────────────────
//  vocab_size, from Vocab::size()  (e.g. ~2000-8000)
//  d_model, 128   (set at construction)
//  max_len, 64    (must match Encoder::max_len)
//
//  Storage layout: row-major std::vector<float>
//  Matrix M[r][c] = M_data[r * cols + c]
// ============================================================

#include "../vocab/Vocab.hpp"

#include <vector>
#include <cmath>
#include <cstddef>
#include <cassert>
#include <random>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>

namespace miniGPT {

// ── Matrix2D: thin wrapper around a flat float vector ─────
// Provides .at(row, col) and .data() access.
// Shape is fixed at construction.
struct Matrix2D {
    std::size_t         rows, cols;
    std::vector<float>  data;   // [rows × cols], row-major

    Matrix2D() : rows(0), cols(0) {}

    Matrix2D(std::size_t r, std::size_t c, float fill = 0.f)
        : rows(r), cols(c), data(r * c, fill) {}

    // Element access
    float& at(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }
    float at(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }

    // Zero the entire matrix
    void zero() { std::fill(data.begin(), data.end(), 0.f); }

    // Return a single row as a const span (pointer + length)
    const float* row_ptr(std::size_t r) const {
        return data.data() + r * cols;
    }
    float* row_ptr(std::size_t r) {
        return data.data() + r * cols;
    }
};

// ── EmbeddingTable class ──────────────────────────────────
class EmbeddingTable {
public:
    // ── Construction ──────────────────────────────────────

    // @param vocab_size  Number of token types (from Vocab::size())
    // @param d_model     Embedding dimension (128 for our model)
    // @param max_len     Maximum sequence length (must match Encoder)
    // @param seed        RNG seed for reproducible initialisation
    EmbeddingTable(std::size_t vocab_size,
                   std::size_t d_model,
                   std::size_t max_len,
                   uint32_t    seed = 42)
        : vocab_size_(vocab_size)
        , d_model_(d_model)
        , max_len_(max_len)
        , E_(vocab_size, d_model, 0.f)      // [vocab_size × d_model]
        , PE_(max_len,   d_model, 0.f)      // [max_len    × d_model]
        , dE_(vocab_size, d_model, 0.f)     // gradient accumulator
    {
        if (d_model == 0 || vocab_size == 0 || max_len == 0)
            throw std::invalid_argument("EmbeddingTable: zero dimension");

        init_embeddings(seed);
        init_positional_encodings();
    }

    // ── Forward pass ──────────────────────────────────────

    // Convert a token ID sequence to an embedding matrix.
    //
    // @param ids   Token IDs, length seq_len  // [seq_len]
    // @param mask  Attention mask, same length (true = real token)
    // @returns     X  [seq_len × d_model]
    //
    // X[p] = E[ids[p]] + PE[p]   for real tokens
    // X[p] = 0                   for padding (ids[p] == kPAD_ID)
    [[nodiscard]] Matrix2D forward(
            const std::vector<TokenID>& ids,
            const std::vector<bool>&    mask) const
    {
        const std::size_t seq_len = ids.size();
        if (seq_len > max_len_)
            throw std::invalid_argument(
                "EmbeddingTable::forward: seq_len exceeds max_len");
        if (mask.size() != seq_len)
            throw std::invalid_argument(
                "EmbeddingTable::forward: mask/ids length mismatch");

        Matrix2D X(seq_len, d_model_, 0.f);  // [seq_len × d_model]

        for (std::size_t p = 0; p < seq_len; ++p) {
            if (!mask[p]) continue;          // leave padding as zero

            TokenID tid = ids[p];
            if (tid >= vocab_size_)
                throw std::out_of_range(
                    "EmbeddingTable::forward: token ID out of range");

            // X[p] = E[tid] + PE[p]
            const float* e_row  = E_.row_ptr(tid);
            const float* pe_row = PE_.row_ptr(p);
            float*       x_row  = X.row_ptr(p);

            for (std::size_t d = 0; d < d_model_; ++d)
                x_row[d] = e_row[d] + pe_row[d];
        }

        return X;   // [seq_len × d_model]
    }

    // ── Backward pass ─────────────────────────────────────

    // Accumulate embedding gradients from upstream gradient dX.
    //
    // @param dX   Gradient of loss w.r.t. X  [seq_len × d_model]
    // @param ids  Same token IDs used in forward()  // [seq_len]
    // @param mask Same mask used in forward()
    //
    // dE[ids[p]] += dX[p]   for each real position p
    // PE has no gradient (fixed).
    // PAD row (ID 0) gets no gradient (it must stay zero).
    void backward(const Matrix2D&              dX,
                  const std::vector<TokenID>&  ids,
                  const std::vector<bool>&     mask)
    {
        const std::size_t seq_len = ids.size();
        assert(dX.rows == seq_len && dX.cols == d_model_);

        for (std::size_t p = 0; p < seq_len; ++p) {
            if (!mask[p]) continue;

            TokenID tid = ids[p];
            if (tid == kPAD_ID) continue;    // PAD stays zero

            const float* dx_row = dX.row_ptr(p);
            float*       de_row = dE_.row_ptr(tid);

            for (std::size_t d = 0; d < d_model_; ++d)
                de_row[d] += dx_row[d];
        }
    }

    // ── Parameter update (AdamW will replace this later) ──

    // Simple SGD step, for smoke-testing the backward pass.
    // AdamW will be wired in during the training loop (component ⑩).
    void sgd_step(float lr) {
        for (std::size_t i = 0; i < vocab_size_; ++i) {
            if (i == kPAD_ID) continue;      // never update PAD
            float* e_row  = E_.row_ptr(i);
            float* de_row = dE_.row_ptr(i);
            for (std::size_t d = 0; d < d_model_; ++d)
                e_row[d] -= lr * de_row[d];
        }
    }

    // Zero the gradient accumulator between batches
    void zero_grad() { dE_.zero(); }

    // ── Accessors ─────────────────────────────────────────
    std::size_t vocab_size() const noexcept { return vocab_size_; }
    std::size_t d_model()    const noexcept { return d_model_; }
    std::size_t max_len()    const noexcept { return max_len_; }

    // Direct access to weight matrices (needed by training loop)
          Matrix2D& E()        { return E_; }
    const Matrix2D& E()  const { return E_; }
    const Matrix2D& PE() const { return PE_; }
          Matrix2D& dE()       { return dE_; }

    // ── Persistence ───────────────────────────────────────

    // Save learned embedding weights to a binary file.
    // Format: [rows:u64][cols:u64][data:float×rows×cols]
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot write embeddings: " + path);
        uint64_t r = E_.rows, c = E_.cols;
        f.write(reinterpret_cast<const char*>(&r), sizeof(r));
        f.write(reinterpret_cast<const char*>(&c), sizeof(c));
        f.write(reinterpret_cast<const char*>(E_.data.data()),
                E_.data.size() * sizeof(float));
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot read embeddings: " + path);
        uint64_t r, c;
        f.read(reinterpret_cast<char*>(&r), sizeof(r));
        f.read(reinterpret_cast<char*>(&c), sizeof(c));
        if (!f) throw std::runtime_error("Embedding file header truncated: " + path);
        if (r != E_.rows || c != E_.cols)
            throw std::runtime_error("Embedding shape mismatch on load");
        f.read(reinterpret_cast<char*>(E_.data.data()),
               static_cast<std::streamsize>(E_.data.size() * sizeof(float)));
        // Verify the full payload was actually read, reject truncated files.
        if (f.gcount() != static_cast<std::streamsize>(E_.data.size() * sizeof(float)))
            throw std::runtime_error("Embedding file payload truncated: " + path);
    }

    // ── Debug ─────────────────────────────────────────────
    void print_stats(std::ostream& os = std::cout) const {
        // Compute mean and max absolute value of E
        double sum = 0.0, maxabs = 0.0;
        std::size_t n = E_.data.size();
        for (float v : E_.data) {
            sum    += v;
            maxabs  = std::max(maxabs, (double)std::abs(v));
        }
        os << "EmbeddingTable: vocab=" << vocab_size_
           << " d_model=" << d_model_
           << " max_len=" << max_len_ << "\n"
           << "  E mean=" << sum / n
           << " maxabs=" << maxabs << "\n";
    }

private:
    // ── Dimensions ────────────────────────────────────────
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t max_len_;

    // ── Parameters and gradients ──────────────────────────
    Matrix2D E_;    // token embeddings        [vocab_size × d_model]
    Matrix2D PE_;   // positional encodings    [max_len    × d_model]
    Matrix2D dE_;   // gradient accumulator    [vocab_size × d_model]

    // ── Initialisation ────────────────────────────────────

    // Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), +sqrt(...))
    // For an embedding table fan_in = 1 (one-hot input), fan_out = d_model
    void init_embeddings(uint32_t seed) {
        std::mt19937 rng(seed);
        float limit = std::sqrt(6.f / (1.f + static_cast<float>(d_model_)));
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (std::size_t i = 0; i < vocab_size_; ++i) {
            if (i == kPAD_ID) continue;  // PAD stays zero
            float* row = E_.row_ptr(i);
            for (std::size_t d = 0; d < d_model_; ++d)
                row[d] = dist(rng);
        }
    }

    // Sinusoidal positional encodings (fixed, Vaswani et al. 2017)
    //   PE[p, 2i]   = sin(p / 10000^(2i / d_model))
    //   PE[p, 2i+1] = cos(p / 10000^(2i / d_model))
    void init_positional_encodings() {
        for (std::size_t p = 0; p < max_len_; ++p) {
            float* row = PE_.row_ptr(p);
            for (std::size_t i = 0; i < d_model_ / 2; ++i) {
                float denom = std::pow(10000.f,
                    2.f * static_cast<float>(i) / static_cast<float>(d_model_));
                float angle = static_cast<float>(p) / denom;
                row[2 * i]     = std::sin(angle);   // even dims
                row[2 * i + 1] = std::cos(angle);   // odd dims
            }
            // If d_model is odd, fill the last dim with sin
            if (d_model_ % 2 != 0) {
                std::size_t last = d_model_ - 1;
                float denom = std::pow(10000.f,
                    2.f * static_cast<float>(last / 2) /
                    static_cast<float>(d_model_));
                row[last] = std::sin(static_cast<float>(p) / denom);
            }
        }
    }
};

} // namespace miniGPT
