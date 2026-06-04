#pragma once

// ============================================================
//  FeedForward.hpp  —  Position-wise feedforward sublayer
//
//  Applied independently to each token position after attention.
//  Two linear transformations with a ReLU in between:
//
//    FFN(x) = max(0, x · W1 + b1) · W2 + b2
//
//  Shapes
//  ──────
//  Input  X        [seq_len × d_model]
//  W1              [d_model × d_ff]      d_ff = 4 * d_model = 512
//  b1              [1 × d_ff]
//  Hidden H = ReLU(X · W1 + b1)  [seq_len × d_ff]
//  W2              [d_ff × d_model]
//  b2              [1 × d_model]
//  Output          [seq_len × d_model]
//
//  Why d_ff = 4 * d_model?
//  ───────────────────────
//  The original transformer paper uses this ratio.  The FFN
//  expands into a higher-dimensional space where the model can
//  represent more complex functions, then projects back down.
//  At d_model=128, d_ff=512 gives ~131K parameters per layer
//  in the FFN alone — the majority of the model's capacity.
//
//  Why ReLU and not GELU?
//  ──────────────────────
//  GELU is slightly better empirically but its backward pass
//  requires computing the Gaussian CDF — more complexity for
//  marginal gain at this scale.  ReLU's backward is just a
//  binary mask (1 where H>0, 0 elsewhere), which is trivial
//  to implement and verify correctly.
//
//  Backward pass
//  ─────────────
//  Given dOut [seq_len × d_model]:
//
//  1. dW2 += Hᵀ · dOut              [d_ff × d_model]
//     db2 += sum over seq of dOut   [d_model]
//     dH   = dOut · W2ᵀ             [seq_len × d_ff]
//
//  2. dH_pre = dH * ReLU_mask        [seq_len × d_ff]
//     (zero out positions where H <= 0)
//
//  3. dW1 += Xᵀ · dH_pre            [d_model × d_ff]
//     db1 += sum over seq of dH_pre  [d_ff]
//     dX   = dH_pre · W1ᵀ           [seq_len × d_model]
// ============================================================

#include "../embed/Embedding.hpp"
#include "../attention/Attention.hpp"   // for matmul helpers

#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <stdexcept>

namespace miniGPT {

// ── FFNCache ──────────────────────────────────────────────
struct FFNCache {
    Matrix2D X;        // input              [seq_len × d_model]
    Matrix2D H;        // post-ReLU hidden   [seq_len × d_ff]
    Matrix2D relu_mask;// 1.0 where H>0      [seq_len × d_ff]
};

// ── FeedForward ───────────────────────────────────────────
class FeedForward {
public:
    // ── Construction ──────────────────────────────────────

    // @param d_model  input/output dimension (e.g. 128)
    // @param d_ff     hidden dimension (default 4 * d_model = 512)
    // @param seed     RNG seed
    FeedForward(std::size_t d_model,
                std::size_t d_ff = 0,
                uint32_t    seed = 42)
        : d_model_(d_model)
        , d_ff_(d_ff == 0 ? 4 * d_model : d_ff)
        , W1_(d_model,  d_ff_,   0.f)   // [d_model × d_ff]
        , b1_(1,        d_ff_,   0.f)   // [1 × d_ff]
        , W2_(d_ff_,    d_model, 0.f)   // [d_ff × d_model]
        , b2_(1,        d_model, 0.f)   // [1 × d_model]
        , dW1_(d_model, d_ff_,   0.f)
        , db1_(1,       d_ff_,   0.f)
        , dW2_(d_ff_,   d_model, 0.f)
        , db2_(1,       d_model, 0.f)
    {
        if (d_model == 0)
            throw std::invalid_argument("FeedForward: d_model cannot be 0");
        init_weights(seed);
    }

    // ── Forward ───────────────────────────────────────────

    // @param X      [seq_len × d_model]
    // @param cache  filled for backward pass
    // @returns      [seq_len × d_model]
    [[nodiscard]] Matrix2D forward(const Matrix2D& X,
                                   FFNCache&        cache) const
    {
        if (X.cols != d_model_)
            throw std::invalid_argument("FFN forward: X.cols != d_model");

        const std::size_t seq_len = X.rows;

        // ── Step 1: H_pre = X · W1 + b1   [seq_len × d_ff] ───
        Matrix2D H_pre = matmul(X, W1_);   // [seq_len × d_ff]

        // Add bias b1 (broadcast across seq positions)
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_ff_; ++d)
                H_pre.at(i, d) += b1_.at(0, d);

        // ── Step 2: H = ReLU(H_pre)  [seq_len × d_ff] ────────
        Matrix2D H(seq_len, d_ff_, 0.f);
        Matrix2D relu_mask(seq_len, d_ff_, 0.f);

        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_ff_; ++d) {
                float v = H_pre.at(i, d);
                if (v > 0.f) {
                    H.at(i, d)         = v;
                    relu_mask.at(i, d) = 1.f;
                }
                // else: H stays 0, mask stays 0
            }

        // ── Step 3: out = H · W2 + b2   [seq_len × d_model] ──
        Matrix2D out = matmul(H, W2_);     // [seq_len × d_model]

        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_model_; ++d)
                out.at(i, d) += b2_.at(0, d);

        // ── Save cache ────────────────────────────────────────
        cache.X         = X;
        cache.H         = H;
        cache.relu_mask = relu_mask;

        return out;   // [seq_len × d_model]
    }

    // ── Backward ──────────────────────────────────────────

    // @param dOut   [seq_len × d_model]
    // @param cache  from forward()
    // @returns      dX  [seq_len × d_model]
    [[nodiscard]] Matrix2D backward(const Matrix2D& dOut,
                                    const FFNCache& cache)
    {
        const std::size_t seq_len = dOut.rows;
        assert(dOut.cols == d_model_);

        // ── 1. Gradient through W2 ────────────────────────────
        // dW2 += Hᵀ · dOut                [d_ff × d_model]
        {
            auto g = matmul_at(cache.H, dOut);
            mat_add_inplace(dW2_, g);
        }

        // db2 += sum over seq positions    [d_model]
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_model_; ++d)
                db2_.at(0, d) += dOut.at(i, d);

        // dH = dOut · W2ᵀ                 [seq_len × d_ff]
        Matrix2D dH = matmul_bt(dOut, W2_);

        // ── 2. Gradient through ReLU ──────────────────────────
        // dH_pre = dH * relu_mask          [seq_len × d_ff]
        Matrix2D dH_pre(seq_len, d_ff_, 0.f);
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_ff_; ++d)
                dH_pre.at(i, d) = dH.at(i, d) * cache.relu_mask.at(i, d);

        // ── 3. Gradient through W1 ────────────────────────────
        // dW1 += Xᵀ · dH_pre              [d_model × d_ff]
        {
            auto g = matmul_at(cache.X, dH_pre);
            mat_add_inplace(dW1_, g);
        }

        // db1 += sum over seq positions    [d_ff]
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_ff_; ++d)
                db1_.at(0, d) += dH_pre.at(i, d);

        // dX = dH_pre · W1ᵀ               [seq_len × d_model]
        Matrix2D dX = matmul_bt(dH_pre, W1_);

        return dX;   // [seq_len × d_model]
    }

    // ── Parameter update ──────────────────────────────────

    void sgd_step(float lr) {
        mat_axpy(-lr, dW1_, W1_);
        mat_axpy(-lr, db1_, b1_);
        mat_axpy(-lr, dW2_, W2_);
        mat_axpy(-lr, db2_, b2_);
    }

    void zero_grad() {
        dW1_.zero(); db1_.zero();
        dW2_.zero(); db2_.zero();
    }

    // ── Accessors ─────────────────────────────────────────
    std::size_t d_model() const noexcept { return d_model_; }
    std::size_t d_ff()    const noexcept { return d_ff_; }

    const Matrix2D& W1() const { return W1_; }
    const Matrix2D& W2() const { return W2_; }
          Matrix2D& W1()       { return W1_; }
          Matrix2D& W2()       { return W2_; }
          Matrix2D& dW1()      { return dW1_; }
          Matrix2D& dW2()      { return dW2_; }
          Matrix2D& db1()      { return db1_; }
          Matrix2D& db2()      { return db2_; }

private:
    std::size_t d_model_, d_ff_;

    Matrix2D W1_, b1_;    // [d_model × d_ff],  [1 × d_ff]
    Matrix2D W2_, b2_;    // [d_ff × d_model],  [1 × d_model]

    Matrix2D dW1_, db1_;  // gradients
    Matrix2D dW2_, db2_;

    // ── Xavier uniform init ───────────────────────────────
    void init_weights(uint32_t seed) {
        std::mt19937 rng(seed);

        // W1: fan_in = d_model, fan_out = d_ff
        {
            float limit = std::sqrt(6.f / (static_cast<float>(d_model_) +
                                           static_cast<float>(d_ff_)));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (float& v : W1_.data) v = dist(rng);
        }
        // W2: fan_in = d_ff, fan_out = d_model
        {
            float limit = std::sqrt(6.f / (static_cast<float>(d_ff_) +
                                           static_cast<float>(d_model_)));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (float& v : W2_.data) v = dist(rng);
        }
        // biases stay zero
    }

    // ── Helpers (mirrors MHA) ─────────────────────────────
    static void mat_add_inplace(Matrix2D& dest, const Matrix2D& src) {
        assert(dest.data.size() == src.data.size());
        for (std::size_t i = 0; i < dest.data.size(); ++i)
            dest.data[i] += src.data[i];
    }

    static void mat_axpy(float alpha, const Matrix2D& src, Matrix2D& dest) {
        assert(dest.data.size() == src.data.size());
        for (std::size_t i = 0; i < dest.data.size(); ++i)
            dest.data[i] += alpha * src.data[i];
    }
};

} // namespace miniGPT
