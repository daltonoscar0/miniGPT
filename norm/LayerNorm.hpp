#pragma once

// ============================================================
//  LayerNorm.hpp  —  Layer normalization
//
//  Applied after attention and after the FFN sublayer within
//  each transformer block (pre-norm arrangement used here).
//
//  For each token position i independently:
//
//    μ_i    = mean(x_i)               scalar
//    σ²_i   = var(x_i) + ε            scalar  (ε for stability)
//    x̂_i   = (x_i − μ_i) / √σ²_i    [d_model]  normalised
//    y_i    = γ ⊙ x̂_i + β            [d_model]  scaled + shifted
//
//  γ (gain) and β (bias) are learned per-dimension parameters,
//  both initialised to γ=1, β=0 (identity at init).
//
//  Why layer norm?
//  ───────────────
//  Each token's activations can have wildly different scales
//  after attention and the FFN.  Layer norm brings every
//  position to zero mean and unit variance before the next
//  sublayer sees it, which keeps gradients well-behaved and
//  dramatically stabilises training.
//
//  Pre-norm vs post-norm
//  ──────────────────────
//  Original transformer: norm AFTER residual (post-norm).
//  GPT-2 and most modern models: norm BEFORE sublayer (pre-norm).
//  We use pre-norm: LayerNorm → sublayer → residual add.
//  Pre-norm is easier to train from scratch without careful
//  learning rate warmup.
//
//  Backward pass
//  ─────────────
//  Given dOut [seq_len × d_model], for each position i:
//
//  Let N = d_model, x̂ = normalised input, σ = √(var + ε)
//
//  dγ += sum_i (dOut_i ⊙ x̂_i)        [d_model]
//  dβ += sum_i dOut_i                  [d_model]
//
//  dx̂_i = dOut_i ⊙ γ                  [d_model]
//
//  dx_i = (1/σ) * (dx̂_i
//          - mean(dx̂_i)
//          - x̂_i * mean(dx̂_i ⊙ x̂_i)) [d_model]
//
//  This is the standard LayerNorm backward derived from the
//  chain rule through mean and variance.
// ============================================================

#include "../embed/Embedding.hpp"

#include <cmath>
#include <cassert>
#include <stdexcept>
#include <numeric>

namespace miniGPT {

// ── LayerNormCache ────────────────────────────────────────
struct LayerNormCache {
    Matrix2D x_hat;   // normalised pre-scale    [seq_len × d_model]
    Matrix2D sigma;   // per-position std dev     [seq_len × 1]
};

// ── LayerNorm ─────────────────────────────────────────────
class LayerNorm {
public:
    // ── Construction ──────────────────────────────────────

    // @param d_model  feature dimension to normalise over
    // @param eps      small constant for numerical stability
    explicit LayerNorm(std::size_t d_model, float eps = 1e-5f)
        : d_model_(d_model)
        , eps_(eps)
        , gamma_(1, d_model, 1.f)   // [1 × d_model]  init = 1
        , beta_( 1, d_model, 0.f)   // [1 × d_model]  init = 0
        , dgamma_(1, d_model, 0.f)
        , dbeta_( 1, d_model, 0.f)
    {
        if (d_model == 0)
            throw std::invalid_argument("LayerNorm: d_model cannot be 0");
    }

    // ── Forward ───────────────────────────────────────────

    // @param X      [seq_len × d_model]
    // @param cache  filled for backward pass
    // @returns      [seq_len × d_model]
    [[nodiscard]] Matrix2D forward(const Matrix2D&   X,
                                   LayerNormCache&   cache) const
    {
        if (X.cols != d_model_)
            throw std::invalid_argument("LayerNorm: X.cols != d_model");

        const std::size_t seq_len = X.rows;
        const float       N       = static_cast<float>(d_model_);

        Matrix2D out(seq_len, d_model_);
        Matrix2D x_hat(seq_len, d_model_);
        Matrix2D sigma(seq_len, 1);

        for (std::size_t i = 0; i < seq_len; ++i) {
            const float* x_row = X.row_ptr(i);

            // ── Mean ──────────────────────────────────────
            float mu = 0.f;
            for (std::size_t d = 0; d < d_model_; ++d)
                mu += x_row[d];
            mu /= N;

            // ── Variance ──────────────────────────────────
            float var = 0.f;
            for (std::size_t d = 0; d < d_model_; ++d) {
                float diff = x_row[d] - mu;
                var += diff * diff;
            }
            var /= N;

            float sig = std::sqrt(var + eps_);
            sigma.at(i, 0) = sig;

            // ── Normalise + scale + shift ──────────────────
            float* xh_row  = x_hat.row_ptr(i);
            float* out_row = out.row_ptr(i);

            for (std::size_t d = 0; d < d_model_; ++d) {
                float xh       = (x_row[d] - mu) / sig;
                xh_row[d]      = xh;
                out_row[d]     = gamma_.at(0, d) * xh + beta_.at(0, d);
            }
        }

        cache.x_hat  = x_hat;
        cache.sigma  = sigma;

        return out;   // [seq_len × d_model]
    }

    // ── Backward ──────────────────────────────────────────

    // @param dOut   [seq_len × d_model]
    // @param cache  from forward()
    // @returns      dX  [seq_len × d_model]
    [[nodiscard]] Matrix2D backward(const Matrix2D&      dOut,
                                    const LayerNormCache& cache)
    {
        const std::size_t seq_len = dOut.rows;
        const float       N       = static_cast<float>(d_model_);
        assert(dOut.cols == d_model_);

        Matrix2D dX(seq_len, d_model_);

        for (std::size_t i = 0; i < seq_len; ++i) {
            const float* dout_row = dOut.row_ptr(i);
            const float* xh_row   = cache.x_hat.row_ptr(i);
            float        sig      = cache.sigma.at(i, 0);
            float*       dx_row   = dX.row_ptr(i);

            // dγ += dOut_i ⊙ x̂_i
            // dβ += dOut_i
            for (std::size_t d = 0; d < d_model_; ++d) {
                dgamma_.at(0, d) += dout_row[d] * xh_row[d];
                dbeta_.at(0, d)  += dout_row[d];
            }

            // dx̂_i = dOut_i ⊙ γ
            // (store temporarily in dx_row)
            for (std::size_t d = 0; d < d_model_; ++d)
                dx_row[d] = dout_row[d] * gamma_.at(0, d);

            // mean(dx̂_i)
            float mean_dxh = 0.f;
            for (std::size_t d = 0; d < d_model_; ++d)
                mean_dxh += dx_row[d];
            mean_dxh /= N;

            // mean(dx̂_i ⊙ x̂_i)
            float mean_dxh_xh = 0.f;
            for (std::size_t d = 0; d < d_model_; ++d)
                mean_dxh_xh += dx_row[d] * xh_row[d];
            mean_dxh_xh /= N;

            // dx_i = (1/σ)(dx̂_i − mean(dx̂_i) − x̂_i·mean(dx̂_i⊙x̂_i))
            float inv_sig = 1.f / sig;
            for (std::size_t d = 0; d < d_model_; ++d)
                dx_row[d] = inv_sig * (dx_row[d]
                                       - mean_dxh
                                       - xh_row[d] * mean_dxh_xh);
        }

        return dX;   // [seq_len × d_model]
    }

    // ── Parameter update ──────────────────────────────────

    void sgd_step(float lr) {
        for (std::size_t d = 0; d < d_model_; ++d) {
            gamma_.at(0, d) -= lr * dgamma_.at(0, d);
            beta_.at(0, d)  -= lr * dbeta_.at(0, d);
        }
    }

    void zero_grad() {
        dgamma_.zero();
        dbeta_.zero();
    }

    // ── Accessors ─────────────────────────────────────────
    std::size_t d_model() const noexcept { return d_model_; }
    float       eps()     const noexcept { return eps_; }

    const Matrix2D& gamma()  const { return gamma_; }
    const Matrix2D& beta()   const { return beta_; }
          Matrix2D& gamma()        { return gamma_; }
          Matrix2D& beta()         { return beta_; }
          Matrix2D& dgamma()       { return dgamma_; }
          Matrix2D& dbeta()        { return dbeta_; }

private:
    std::size_t d_model_;
    float       eps_;

    Matrix2D gamma_;    // [1 × d_model]  scale
    Matrix2D beta_;     // [1 × d_model]  shift
    Matrix2D dgamma_;   // gradient accumulators
    Matrix2D dbeta_;
};

} // namespace miniGPT
