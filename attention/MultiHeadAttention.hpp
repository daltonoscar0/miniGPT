#pragma once

// ============================================================
//  MultiHeadAttention.hpp, Multi-head self-attention
//
//  Splits d_model into h independent "heads", runs scaled
//  dot-product attention in each head's subspace, then
//  concatenates and projects back to d_model.
//
//  Architecture
//  ────────────
//  Input X         [seq_len × d_model]
//
//  For each head h ∈ {0 .. n_heads-1}:
//    Q_h = X · W_Q_h    [seq_len × d_k]   d_k = d_model / n_heads
//    K_h = X · W_K_h    [seq_len × d_k]
//    V_h = X · W_V_h    [seq_len × d_v]   d_v = d_k
//    A_h = Attention(Q_h, K_h, V_h)       [seq_len × d_v]
//
//  Concat  [A_0 | A_1 | ... | A_{h-1}]    [seq_len × d_model]
//  Output  = Concat · W_O                 [seq_len × d_model]
//
//  Parameters
//  ──────────
//  W_Q_h   [d_model × d_k]   per head  (stored as one [d_model × d_model])
//  W_K_h   [d_model × d_k]   per head
//  W_V_h   [d_model × d_k]   per head
//  W_O     [d_model × d_model]
//  b_O     [d_model]          output bias
//
//  Storage: W_Q, W_K, W_V are stored as full [d_model × d_model]
//  matrices. Head h's slice is columns [h*d_k .. (h+1)*d_k).
//  This makes the projection a single matmul rather than h
//  separate ones, which is cleaner for backprop.
//
//  Backward pass
//  ─────────────
//  Given dOut [seq_len × d_model]:
//
//  1. dConcat = dOut · W_Oᵀ              [seq_len × d_model]
//     dW_O   += Concatᵀ · dOut           [d_model × d_model]
//     db_O   += sum over seq of dOut     [d_model]
//
//  2. For each head h:
//     dA_h   = dConcat[:, h*d_k:(h+1)*d_k]   [seq_len × d_k]
//     {dQ_h, dK_h, dV_h} = attn_backward(dA_h, cache_h)
//     dW_Q_h += Xᵀ · dQ_h               [d_model × d_k]
//     dW_K_h += Xᵀ · dK_h
//     dW_V_h += Xᵀ · dV_h
//     dX     += dQ_h · W_Q_hᵀ  +  dK_h · W_K_hᵀ  +  dV_h · W_V_hᵀ
// ============================================================

#include "Attention.hpp"
#include "../embed/Embedding.hpp"

#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <stdexcept>
#include <string>

namespace miniGPT {

// ── MHACache ──────────────────────────────────────────────
// Everything the backward pass needs from the forward pass.
struct MHACache {
    Matrix2D              X;          // input            [seq_len × d_model]
    Matrix2D              concat;     // pre-output-proj  [seq_len × d_model]
    std::vector<AttentionCache> heads; // one per head
};

// ── MultiHeadAttention ────────────────────────────────────
class MultiHeadAttention {
public:
    // ── Construction ──────────────────────────────────────

    // @param d_model   total embedding dimension (e.g. 128)
    // @param n_heads   number of attention heads (e.g. 4)
    // @param seed      RNG seed for weight initialisation
    //
    // Requires d_model % n_heads == 0.
    MultiHeadAttention(std::size_t d_model,
                       std::size_t n_heads,
                       uint32_t    seed = 42)
        : d_model_(d_model)
        , n_heads_(n_heads)
        , d_k_(d_model / n_heads)
        // Weight matrices, full projections
        , W_Q_(d_model, d_model, 0.f)   // [d_model × d_model]
        , W_K_(d_model, d_model, 0.f)
        , W_V_(d_model, d_model, 0.f)
        , W_O_(d_model, d_model, 0.f)
        , b_O_(1, d_model, 0.f)         // [1 × d_model] broadcast
        // Gradient accumulators
        , dW_Q_(d_model, d_model, 0.f)
        , dW_K_(d_model, d_model, 0.f)
        , dW_V_(d_model, d_model, 0.f)
        , dW_O_(d_model, d_model, 0.f)
        , db_O_(1, d_model, 0.f)
    {
        if (d_model % n_heads != 0)
            throw std::invalid_argument(
                "MultiHeadAttention: d_model must be divisible by n_heads");
        if (n_heads == 0 || d_model == 0)
            throw std::invalid_argument(
                "MultiHeadAttention: zero dimension");

        init_weights(seed);
    }

    // ── Forward ───────────────────────────────────────────

    // @param X      [seq_len × d_model]
    // @param mask   [seq_len] attention mask
    // @param causal  true for autoregressive LM
    // @param cache  filled for backward pass
    // @returns      [seq_len × d_model]
    [[nodiscard]] Matrix2D forward(
            const Matrix2D&          X,
            const std::vector<bool>& mask,
            bool                     causal,
            MHACache&                cache) const
    {
        const std::size_t seq_len = X.rows;
        if (X.cols != d_model_)
            throw std::invalid_argument("MHA forward: X.cols != d_model");

        // Project X -> Q, K, V for all heads at once
        // Q_full [seq_len × d_model], head h occupies cols [h*d_k, (h+1)*d_k)
        Matrix2D Q_full = matmul(X, W_Q_);  // [seq_len × d_model]
        Matrix2D K_full = matmul(X, W_K_);  // [seq_len × d_model]
        Matrix2D V_full = matmul(X, W_V_);  // [seq_len × d_model]

        // Concat buffer, filled head by head
        Matrix2D concat(seq_len, d_model_, 0.f);  // [seq_len × d_model]

        cache.heads.resize(n_heads_);

        for (std::size_t h = 0; h < n_heads_; ++h) {
            // Slice out this head's Q, K, V columns
            // [seq_len × d_k]
            Matrix2D Q_h = slice_cols(Q_full, h * d_k_, d_k_);
            Matrix2D K_h = slice_cols(K_full, h * d_k_, d_k_);
            Matrix2D V_h = slice_cols(V_full, h * d_k_, d_k_);

            // Run scaled dot-product attention
            Matrix2D A_h = ScaledDotProductAttention::forward(
                Q_h, K_h, V_h, mask, causal, cache.heads[h]);

            // Write A_h into the correct slice of concat
            write_cols(concat, A_h, h * d_k_);
        }

        // Output projection: out = concat · W_O + b_O
        Matrix2D out = matmul(concat, W_O_);  // [seq_len × d_model]

        // Add bias (broadcast b_O across all seq positions)
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_model_; ++d)
                out.at(i, d) += b_O_.at(0, d);

        // Save for backward
        cache.X      = X;
        cache.concat = concat;

        return out;  // [seq_len × d_model]
    }

    // ── Backward ──────────────────────────────────────────

    // @param dOut   [seq_len × d_model]  upstream gradient
    // @param cache  from forward()
    // @returns      dX  [seq_len × d_model]
    [[nodiscard]] Matrix2D backward(
            const Matrix2D& dOut,
            const MHACache& cache)
    {
        const std::size_t seq_len = dOut.rows;

        // ── 1. Output projection backward ─────────────────
        // dW_O += concatᵀ · dOut      [d_model × d_model]
        {
            auto g = matmul_at(cache.concat, dOut);
            mat_add_inplace(dW_O_, g);
        }

        // db_O += sum over seq positions
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_model_; ++d)
                db_O_.at(0, d) += dOut.at(i, d);

        // dConcat = dOut · W_Oᵀ       [seq_len × d_model]
        Matrix2D dConcat = matmul_bt(dOut, W_O_);

        // ── 2. Per-head backward ───────────────────────────
        Matrix2D dX(seq_len, d_model_, 0.f);  // accumulate here

        for (std::size_t h = 0; h < n_heads_; ++h) {
            // Slice this head's gradient from dConcat
            Matrix2D dA_h = slice_cols(dConcat, h * d_k_, d_k_);  // [seq_len × d_k]

            // Attention backward -> dQ_h, dK_h, dV_h
            auto grads = ScaledDotProductAttention::backward(
                dA_h, cache.heads[h]);

            // dW_Q_h += Xᵀ · dQ_h    [d_model × d_k]
            {
                auto g = matmul_at(cache.X, grads.dQ);
                write_cols_add(dW_Q_, g, h * d_k_);
            }
            // dW_K_h += Xᵀ · dK_h
            {
                auto g = matmul_at(cache.X, grads.dK);
                write_cols_add(dW_K_, g, h * d_k_);
            }
            // dW_V_h += Xᵀ · dV_h
            {
                auto g = matmul_at(cache.X, grads.dV);
                write_cols_add(dW_V_, g, h * d_k_);
            }

            // dX += dQ_h · W_Q_hᵀ  [seq_len × d_model] <- accumulated
            {
                Matrix2D W_Q_h = slice_cols(W_Q_, h * d_k_, d_k_);  // [d_model × d_k]
                auto g = matmul_bt(grads.dQ, W_Q_h);  // [seq_len × d_model]... wait
                // matmul_bt(A[seq×dk], B[d_model×dk]) -> [seq × d_model] 
                mat_add_inplace(dX, g);
            }
            {
                Matrix2D W_K_h = slice_cols(W_K_, h * d_k_, d_k_);
                auto g = matmul_bt(grads.dK, W_K_h);
                mat_add_inplace(dX, g);
            }
            {
                Matrix2D W_V_h = slice_cols(W_V_, h * d_k_, d_k_);
                auto g = matmul_bt(grads.dV, W_V_h);
                mat_add_inplace(dX, g);
            }
        }

        return dX;  // [seq_len × d_model]
    }

    // ── Parameter update ──────────────────────────────────

    // Simple SGD, AdamW replaces this in the training loop.
    void sgd_step(float lr) {
        mat_axpy(-lr, dW_Q_, W_Q_);
        mat_axpy(-lr, dW_K_, W_K_);
        mat_axpy(-lr, dW_V_, W_V_);
        mat_axpy(-lr, dW_O_, W_O_);
        mat_axpy(-lr, db_O_, b_O_);
    }

    void zero_grad() {
        dW_Q_.zero(); dW_K_.zero(); dW_V_.zero();
        dW_O_.zero(); db_O_.zero();
    }

    // ── Accessors ─────────────────────────────────────────
    std::size_t d_model()  const noexcept { return d_model_; }
    std::size_t n_heads()  const noexcept { return n_heads_; }
    std::size_t d_k()      const noexcept { return d_k_; }

    Matrix2D& W_Q() { return W_Q_; }
    Matrix2D& W_K() { return W_K_; }
    Matrix2D& W_V() { return W_V_; }
    Matrix2D& W_O() { return W_O_; }
    Matrix2D& b_O() { return b_O_; }

    const Matrix2D& W_Q() const { return W_Q_; }
    const Matrix2D& W_K() const { return W_K_; }
    const Matrix2D& W_V() const { return W_V_; }
    const Matrix2D& W_O() const { return W_O_; }

    Matrix2D& dW_Q() { return dW_Q_; }
    Matrix2D& dW_K() { return dW_K_; }
    Matrix2D& dW_V() { return dW_V_; }
    Matrix2D& dW_O() { return dW_O_; }

private:
    std::size_t d_model_, n_heads_, d_k_;

    // Weights  [d_model × d_model]
    Matrix2D W_Q_, W_K_, W_V_, W_O_;
    Matrix2D b_O_;   // [1 × d_model]

    // Gradient accumulators
    Matrix2D dW_Q_, dW_K_, dW_V_, dW_O_, db_O_;

    // ── Initialisation ────────────────────────────────────
    // Xavier uniform for projection matrices
    void init_weights(uint32_t seed) {
        std::mt19937 rng(seed);
        float limit = std::sqrt(6.f / (static_cast<float>(d_model_) +
                                       static_cast<float>(d_model_)));
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (float& v : W_Q_.data) v = dist(rng);
        for (float& v : W_K_.data) v = dist(rng);
        for (float& v : W_V_.data) v = dist(rng);
        for (float& v : W_O_.data) v = dist(rng);
        // b_O stays zero
    }

    // ── Column slice helpers ──────────────────────────────

    // Extract columns [col_start, col_start+n_cols) from M
    // M [rows × cols] -> out [rows × n_cols]
    static Matrix2D slice_cols(const Matrix2D& M,
                                std::size_t col_start,
                                std::size_t n_cols)
    {
        Matrix2D out(M.rows, n_cols);
        for (std::size_t r = 0; r < M.rows; ++r)
            for (std::size_t c = 0; c < n_cols; ++c)
                out.at(r, c) = M.at(r, col_start + c);
        return out;
    }

    // Write src [rows × n_cols] into dest starting at col_start
    static void write_cols(Matrix2D& dest,
                            const Matrix2D& src,
                            std::size_t col_start)
    {
        for (std::size_t r = 0; r < src.rows; ++r)
            for (std::size_t c = 0; c < src.cols; ++c)
                dest.at(r, col_start + c) = src.at(r, c);
    }

    // Add src [rows × n_cols] into dest starting at col_start
    static void write_cols_add(Matrix2D& dest,
                                const Matrix2D& src,
                                std::size_t col_start)
    {
        for (std::size_t r = 0; r < src.rows; ++r)
            for (std::size_t c = 0; c < src.cols; ++c)
                dest.at(r, col_start + c) += src.at(r, c);
    }

    // dest += src  (element-wise)
    static void mat_add_inplace(Matrix2D& dest, const Matrix2D& src) {
        assert(dest.rows == src.rows && dest.cols == src.cols);
        for (std::size_t i = 0; i < dest.data.size(); ++i)
            dest.data[i] += src.data[i];
    }

    // dest += alpha * src
    static void mat_axpy(float alpha,
                          const Matrix2D& src,
                          Matrix2D& dest)
    {
        assert(dest.data.size() == src.data.size());
        for (std::size_t i = 0; i < dest.data.size(); ++i)
            dest.data[i] += alpha * src.data[i];
    }
};

} // namespace miniGPT
