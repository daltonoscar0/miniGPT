#pragma once

// ============================================================
//  MiniGPT.hpp, Full GPT-style language model
//
//  Architecture
//  ────────────
//  Token IDs  [seq_len]
//    │  EmbeddingTable  (token embed + sinusoidal PE)
//    ▼
//  X          [seq_len × d_model]
//    │  TransformerBlock × n_layers
//    ▼
//  H          [seq_len × d_model]
//    │  LayerNorm (final)
//    ▼
//  H_norm     [seq_len × d_model]
//    │  LMHead (linear + softmax)
//    ▼
//  Probs      [seq_len × vocab_size]
//
//  Hyperparameters (defaults = project spec)
//  ──────────────────────────────────────────
//  n_layers   = 2
//  n_heads    = 4
//  d_model    = 128
//  d_ff       = 512  (4 × d_model)
//  max_len    = 64
//
//  Forward pass stores one ModelCache per forward call.
//  Backward pass walks the cache in reverse order.
// ============================================================

#include "../embed/Embedding.hpp"
#include "../block/TransformerBlock.hpp"
#include "../norm/LayerNorm.hpp"
#include "../lm_head/LMHead.hpp"
#include "../encoder/Encoder.hpp"

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <fstream>
#include <iostream>

namespace miniGPT {

// ── ModelConfig ───────────────────────────────────────────
struct ModelConfig {
    std::size_t vocab_size = 2000;
    std::size_t d_model    = 128;
    std::size_t n_heads    = 4;
    std::size_t n_layers   = 2;
    std::size_t d_ff       = 512;
    std::size_t max_len    = 64;
    uint32_t    seed       = 42;
    bool        tie_weights = true;
};

// ── ModelCache ────────────────────────────────────────────
struct ModelCache {
    std::vector<TokenID>          ids;       // [seq_len]
    std::vector<bool>             mask;      // [seq_len]
    Matrix2D                      embed_out; // [seq_len × d_model]
    std::vector<BlockCache>       block_caches;
    LayerNormCache                final_ln_cache;
    LMHeadCache                   lmhead_cache;
};

// ── MiniGPT ───────────────────────────────────────────────
class MiniGPT {
public:
    // ── Construction ──────────────────────────────────────
    explicit MiniGPT(const ModelConfig& cfg)
        : cfg_(cfg)
        , embed_(cfg.vocab_size, cfg.d_model, cfg.max_len, cfg.seed)
        , final_ln_(cfg.d_model)
        , lm_head_(cfg.d_model, cfg.vocab_size, cfg.seed + 999)
    {
        // Build transformer blocks
        blocks_.reserve(cfg.n_layers);
        for (std::size_t l = 0; l < cfg.n_layers; ++l)
            blocks_.emplace_back(cfg.d_model, cfg.n_heads,
                                 cfg.d_ff, cfg.seed + l + 1);

        // Tie LM head weights to embedding table
        if (cfg.tie_weights)
            lm_head_.tie_weights(embed_.E(), embed_.dE());
    }

    // ── Forward ───────────────────────────────────────────

    // @param ids    [seq_len]  from Encoder
    // @param mask   [seq_len]  from Encoder
    // @param cache  filled for backward
    // @returns      probs  [seq_len × vocab_size]
    [[nodiscard]] Matrix2D forward(
            const std::vector<TokenID>& ids,
            const std::vector<bool>&    mask,
            ModelCache&                 cache) const
    {
        // ── Embedding ──────────────────────────────────────
        Matrix2D X = embed_.forward(ids, mask);   // [seq_len × d_model]

        // ── Transformer blocks ─────────────────────────────
        cache.block_caches.resize(cfg_.n_layers);
        for (std::size_t l = 0; l < cfg_.n_layers; ++l)
            X = blocks_[l].forward(X, mask, /*causal=*/true,
                                   cache.block_caches[l]);

        // ── Final layer norm ───────────────────────────────
        Matrix2D H = final_ln_.forward(X, cache.final_ln_cache);

        // ── LM head ───────────────────────────────────────
        Matrix2D probs = lm_head_.forward(H, cache.lmhead_cache);

        // Save ids/mask for backward
        cache.ids      = ids;
        cache.mask     = mask;
        cache.embed_out = embed_.forward(ids, mask);  // re-run for cache
        // Note: embed_out is re-computed here to keep embed_ const.
        // In a production system you'd cache the first run.

        return probs;   // [seq_len × vocab_size]
    }

    // ── Loss ──────────────────────────────────────────────

    // Convenience: forward + cross-entropy in one call.
    // @param ids     [seq_len]  input token IDs
    // @param mask    [seq_len]  attention mask
    // @param targets [seq_len]  target token IDs (next-token)
    // @param cache   filled for backward
    // @returns       scalar loss
    float forward_loss(
            const std::vector<TokenID>& ids,
            const std::vector<bool>&    mask,
            const std::vector<TokenID>& targets,
            ModelCache&                 cache)
    {
        auto probs = forward(ids, mask, cache);
        auto lg    = LMHead::cross_entropy(probs, targets);
        cache.lmhead_cache.probs = probs;   // ensure probs stored
        last_dLogits_ = lg.dLogits;
        return lg.loss;
    }

    // ── Backward ──────────────────────────────────────────

    // Must be called after forward_loss().
    // Walks the cache in reverse, accumulating gradients.
    void backward(const ModelCache& cache)
    {
        // ── LM head backward ──────────────────────────────
        Matrix2D dH = lm_head_.backward(last_dLogits_,
                                         cache.lmhead_cache);

        // ── Final layer norm backward ──────────────────────
        Matrix2D dX = final_ln_.backward(dH, cache.final_ln_cache);

        // ── Transformer blocks backward (reverse order) ───
        for (int l = static_cast<int>(cfg_.n_layers) - 1; l >= 0; --l)
            dX = blocks_[l].backward(dX, cache.block_caches[l]);

        // ── Embedding backward ─────────────────────────────
        embed_.backward(dX, cache.ids, cache.mask);
    }

    // ── Gradient utilities ────────────────────────────────

    void zero_grad() {
        embed_.zero_grad();
        for (auto& blk : blocks_) blk.zero_grad();
        final_ln_.zero_grad();
        lm_head_.zero_grad();
    }

    // Gradient clipping: scale all gradients so their global
    // L2 norm does not exceed max_norm.
    float clip_grad_norm(float max_norm) {
        // Compute global L2 norm across all gradient tensors
        double norm_sq = 0.0;

        auto accum = [&](const Matrix2D& g) {
            for (float v : g.data) norm_sq += v * v;
        };

        accum(embed_.dE());
        for (auto& blk : blocks_) {
            accum(blk.mha().dW_Q()); accum(blk.mha().dW_K());
            accum(blk.mha().dW_V()); accum(blk.mha().dW_O());
            accum(blk.ffn().dW1());  accum(blk.ffn().dW2());
            accum(blk.ffn().db1());  accum(blk.ffn().db2());
            accum(blk.ln1().dgamma()); accum(blk.ln1().dbeta());
            accum(blk.ln2().dgamma()); accum(blk.ln2().dbeta());
        }
        accum(final_ln_.dgamma()); accum(final_ln_.dbeta());
        if (!lm_head_.is_tied()) accum(lm_head_.dW());
        accum(lm_head_.db());

        float norm = std::sqrt(static_cast<float>(norm_sq));
        if (norm > max_norm) {
            float scale = max_norm / norm;
            auto clip = [&](Matrix2D& g) {
                for (float& v : g.data) v *= scale;
            };
            clip(embed_.dE());
            for (auto& blk : blocks_) {
                clip(blk.mha().dW_Q()); clip(blk.mha().dW_K());
                clip(blk.mha().dW_V()); clip(blk.mha().dW_O());
                clip(blk.ffn().dW1());  clip(blk.ffn().dW2());
                clip(blk.ffn().db1());  clip(blk.ffn().db2());
                clip(blk.ln1().dgamma()); clip(blk.ln1().dbeta());
                clip(blk.ln2().dgamma()); clip(blk.ln2().dbeta());
            }
            clip(final_ln_.dgamma()); clip(final_ln_.dbeta());
            if (!lm_head_.is_tied()) clip(lm_head_.dW());
            clip(lm_head_.db());
        }
        return norm;
    }

    // ── Accessors ─────────────────────────────────────────
    const ModelConfig&  config()    const { return cfg_; }
    EmbeddingTable&     embed()     { return embed_; }
    TransformerBlock&   block(std::size_t l) { return blocks_[l]; }
    LayerNorm&          final_ln()  { return final_ln_; }
    LMHead&             lm_head()   { return lm_head_; }

    const EmbeddingTable&   embed()    const { return embed_; }
    const TransformerBlock& block(std::size_t l) const { return blocks_[l]; }

    std::size_t n_layers() const { return cfg_.n_layers; }

    // Count total trainable parameters
    std::size_t param_count() const {
        std::size_t n = 0;
        // Embedding (not double-counted when tied)
        n += cfg_.vocab_size * cfg_.d_model;
        // Each block: MHA (4 weight matrices) + FFN (2 weights + 2 biases)
        //           + 2 LayerNorms (gamma + beta each)
        for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
            n += 4 * cfg_.d_model * cfg_.d_model;  // W_Q/K/V/O
            n += cfg_.d_model;                      // b_O
            n += cfg_.d_model * cfg_.d_ff * 2;      // W1, W2
            n += cfg_.d_ff + cfg_.d_model;          // b1, b2
            n += cfg_.d_model * 4;                  // 2x (gamma + beta)
        }
        // Final LN
        n += cfg_.d_model * 2;
        // LM head (only bias if tied)
        if (!lm_head_.is_tied())
            n += cfg_.d_model * cfg_.vocab_size;
        n += cfg_.vocab_size;  // bias
        return n;
    }

    void print_config(std::ostream& os = std::cout) const {
        os << "MiniGPT config:\n"
           << "  vocab_size = " << cfg_.vocab_size << "\n"
           << "  d_model    = " << cfg_.d_model    << "\n"
           << "  n_heads    = " << cfg_.n_heads    << "\n"
           << "  n_layers   = " << cfg_.n_layers   << "\n"
           << "  d_ff       = " << cfg_.d_ff       << "\n"
           << "  max_len    = " << cfg_.max_len    << "\n"
           << "  tie_weights= " << cfg_.tie_weights << "\n"
           << "  ~params    = " << param_count()   << "\n";
    }

private:
    ModelConfig                  cfg_;
    EmbeddingTable               embed_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm                    final_ln_;
    LMHead                       lm_head_;
    Matrix2D                     last_dLogits_;  // set by forward_loss()
};

} // namespace miniGPT
