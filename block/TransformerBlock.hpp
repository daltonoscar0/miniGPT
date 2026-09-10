#pragma once

// ============================================================
//  TransformerBlock.hpp, Single transformer block
//
//  Pre-norm GPT-style layout:
//
//    y1 = X  + MHA(LayerNorm1(X))
//    y2 = y1 + FFN(LayerNorm2(y1))
//
//  The residual connections are critical:
//  - They let gradients flow directly from loss to early layers
//  - They let the block learn a *residual* correction rather
//    than a full transformation (easier to optimise)
//
//  Backward pass
//  ─────────────
//  Forward stores: X, ln1_out, mha_out, y1, ln2_out, ffn_out
//
//  dY2 (upstream gradient):
//    dFFN_out  = dY2                          (through residual)
//    dY1      += dY2                          (residual branch)
//    dLN2_out  = FFN.backward(dFFN_out)
//    dY1      += LN2.backward(dLN2_out)
//
//  dY1:
//    dMHA_out  = dY1                          (through residual)
//    dX       += dY1                          (residual branch)
//    dLN1_out  = MHA.backward(dMHA_out)
//    dX       += LN1.backward(dLN1_out)
//
//  Returns dX  [seq_len × d_model]
// ============================================================

#include "../attention/MultiHeadAttention.hpp"
#include "../ffn/FeedForward.hpp"
#include "../norm/LayerNorm.hpp"
#include "../embed/Embedding.hpp"

#include <cassert>
#include <stdexcept>

namespace miniGPT {

// ── BlockCache ────────────────────────────────────────────
struct BlockCache {
    Matrix2D       X;           // block input          [seq_len × d_model]
    LayerNormCache ln1_cache;   // before MHA
    MHACache       mha_cache;
    Matrix2D       y1;          // after MHA + residual [seq_len × d_model]
    LayerNormCache ln2_cache;   // before FFN
    FFNCache       ffn_cache;
};

// ── TransformerBlock ──────────────────────────────────────
class TransformerBlock {
public:
    // ── Construction ──────────────────────────────────────

    // @param d_model   embedding dimension (e.g. 128)
    // @param n_heads   attention heads     (e.g. 4)
    // @param d_ff      FFN hidden dim      (default 4*d_model)
    // @param seed      RNG seed
    TransformerBlock(std::size_t d_model,
                     std::size_t n_heads,
                     std::size_t d_ff  = 0,
                     uint32_t    seed  = 42)
        : d_model_(d_model)
        , ln1_(d_model)
        , mha_(d_model, n_heads, seed)
        , ln2_(d_model)
        , ffn_(d_model, d_ff == 0 ? 4 * d_model : d_ff, seed + 1)
    {}

    // ── Forward ───────────────────────────────────────────

    // @param X      [seq_len × d_model]
    // @param mask   [seq_len]
    // @param causal  true for LM
    // @param cache  filled for backward
    // @returns      [seq_len × d_model]
    [[nodiscard]] Matrix2D forward(const Matrix2D&          X,
                                   const std::vector<bool>& mask,
                                   bool                     causal,
                                   BlockCache&              cache) const
    {
        const std::size_t seq_len = X.rows;
        if (X.cols != d_model_)
            throw std::invalid_argument("TransformerBlock: X.cols != d_model");

        // ── Sub-layer 1: MHA with pre-norm + residual ─────
        // ln1_out = LayerNorm1(X)          [seq_len × d_model]
        Matrix2D ln1_out = ln1_.forward(X, cache.ln1_cache);

        // mha_out = MHA(ln1_out)           [seq_len × d_model]
        Matrix2D mha_out = mha_.forward(ln1_out, mask, causal,
                                        cache.mha_cache);

        // y1 = X + mha_out                 [seq_len × d_model]
        Matrix2D y1(seq_len, d_model_);
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_model_; ++d)
                y1.at(i, d) = X.at(i, d) + mha_out.at(i, d);

        // ── Sub-layer 2: FFN with pre-norm + residual ─────
        // ln2_out = LayerNorm2(y1)         [seq_len × d_model]
        Matrix2D ln2_out = ln2_.forward(y1, cache.ln2_cache);

        // ffn_out = FFN(ln2_out)           [seq_len × d_model]
        Matrix2D ffn_out = ffn_.forward(ln2_out, cache.ffn_cache);

        // y2 = y1 + ffn_out                [seq_len × d_model]
        Matrix2D y2(seq_len, d_model_);
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t d = 0; d < d_model_; ++d)
                y2.at(i, d) = y1.at(i, d) + ffn_out.at(i, d);

        // Save for backward
        cache.X  = X;
        cache.y1 = y1;

        return y2;   // [seq_len × d_model]
    }

    // ── Backward ──────────────────────────────────────────

    // @param dY2    upstream gradient      [seq_len × d_model]
    // @param cache  from forward()
    // @returns      dX  [seq_len × d_model]
    [[nodiscard]] Matrix2D backward(const Matrix2D& dY2,
                                    const BlockCache& cache)
    {
        const std::size_t seq_len = dY2.rows;
        const std::size_t d      = dY2.cols;

        // ── FFN sublayer backward ─────────────────────────
        // dY2 flows two ways: through FFN and through residual

        // Through FFN: dFFN_in = FFN.backward(dY2)
        Matrix2D dLN2_out = ffn_.backward(dY2, cache.ffn_cache);

        // Through LN2: dY1_from_ffn = LN2.backward(dLN2_out)
        Matrix2D dY1 = ln2_.backward(dLN2_out, cache.ln2_cache);

        // Residual: dY1 += dY2  (identity branch)
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t j = 0; j < d; ++j)
                dY1.at(i, j) += dY2.at(i, j);

        // ── MHA sublayer backward ─────────────────────────
        // dY1 flows two ways: through MHA and through residual

        // Through MHA: dMHA_in = MHA.backward(dY1)
        Matrix2D dLN1_out = mha_.backward(dY1, cache.mha_cache);

        // Through LN1: dX_from_mha = LN1.backward(dLN1_out)
        Matrix2D dX = ln1_.backward(dLN1_out, cache.ln1_cache);

        // Residual: dX += dY1  (identity branch)
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t j = 0; j < d; ++j)
                dX.at(i, j) += dY1.at(i, j);

        return dX;   // [seq_len × d_model]
    }

    // ── Parameter update ──────────────────────────────────

    void sgd_step(float lr) {
        ln1_.sgd_step(lr);
        mha_.sgd_step(lr);
        ln2_.sgd_step(lr);
        ffn_.sgd_step(lr);
    }

    void zero_grad() {
        ln1_.zero_grad();
        mha_.zero_grad();
        ln2_.zero_grad();
        ffn_.zero_grad();
    }

    // ── Accessors ─────────────────────────────────────────
    std::size_t       d_model()  const noexcept { return d_model_; }
    LayerNorm&        ln1()      { return ln1_; }
    MultiHeadAttention& mha()   { return mha_; }
    LayerNorm&        ln2()      { return ln2_; }
    FeedForward&      ffn()      { return ffn_; }
    const LayerNorm&  ln1()      const { return ln1_; }
    const MultiHeadAttention& mha() const { return mha_; }
    const LayerNorm&  ln2()      const { return ln2_; }
    const FeedForward& ffn()     const { return ffn_; }

private:
    std::size_t        d_model_;
    LayerNorm          ln1_;   // before MHA
    MultiHeadAttention mha_;
    LayerNorm          ln2_;   // before FFN
    FeedForward        ffn_;
};

} // namespace miniGPT
