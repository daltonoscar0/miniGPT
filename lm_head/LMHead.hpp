#pragma once

// ============================================================
//  LMHead.hpp, Language model head
//
//  Takes the final transformer block's output and produces
//  a probability distribution over the vocabulary.
//
//  Forward pass
//  ────────────
//  Input  H        [seq_len × d_model]    (last block output)
//  W               [d_model × vocab_size] (projection weights)
//  b               [1 × vocab_size]       (bias)
//
//  Logits = H · W + b                     [seq_len × vocab_size]
//  Probs  = softmax(Logits)  row-wise     [seq_len × vocab_size]
//
//  Loss (cross-entropy, for training)
//  ────────────────────────────────────
//  Given target IDs t_i for each real position i:
//
//    loss = −(1/n_real) · Σ_i log(Probs[i, t_i])
//
//  where n_real = number of non-padding positions.
//  Padding positions (target == kPAD_ID) are excluded.
//
//  Backward pass
//  ─────────────
//  The gradient of cross-entropy + softmax combined is clean:
//
//    dLogits[i, j] = (Probs[i, j] − 1{j == t_i}) / n_real
//
//  This is the standard result, the gradient is just the
//  predicted probability minus the one-hot target, scaled.
//  Masked positions get zero gradient.
//
//  From dLogits:
//    dW += Hᵀ · dLogits              [d_model × vocab_size]
//    db += sum over seq of dLogits   [vocab_size]
//    dH  = dLogits · Wᵀ              [seq_len × d_model]
//
//  Weight tying (optional)
//  ────────────────────────
//  GPT-2 ties the LM head weights to the embedding table:
//  W = Eᵀ.  This halves the parameter count and often
//  improves performance.  We support it via tie_weights().
//  When tied, dW is accumulated into dE instead.
//  Not enabled by default, set explicitly before training.
// ============================================================

#include "../embed/Embedding.hpp"
#include "../attention/Attention.hpp"   // for matmul helpers
#include "../vocab/Vocab.hpp"

#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <stdexcept>
#include <limits>

namespace miniGPT {

// ── LMHeadCache ───────────────────────────────────────────
struct LMHeadCache {
    Matrix2D H;       // input to LM head    [seq_len × d_model]
    Matrix2D logits;  // pre-softmax         [seq_len × vocab_size]
    Matrix2D probs;   // post-softmax        [seq_len × vocab_size]
};

// ── LMHead ────────────────────────────────────────────────
class LMHead {
public:
    // ── Construction ──────────────────────────────────────

    // @param d_model    input dimension
    // @param vocab_size output dimension (number of token types)
    // @param seed       RNG seed
    LMHead(std::size_t d_model,
           std::size_t vocab_size,
           uint32_t    seed = 42)
        : d_model_(d_model)
        , vocab_size_(vocab_size)
        , W_(d_model, vocab_size, 0.f)    // [d_model × vocab_size]
        , b_(1, vocab_size, 0.f)          // [1 × vocab_size]
        , dW_(d_model, vocab_size, 0.f)
        , db_(1, vocab_size, 0.f)
        , tied_E_(nullptr)
        , tied_dE_(nullptr)
    {
        if (d_model == 0 || vocab_size == 0)
            throw std::invalid_argument("LMHead: zero dimension");
        init_weights(seed);
    }

    // ── Weight tying ──────────────────────────────────────

    // Tie projection weights to the embedding table.
    // After calling this, W_ is not used, forward() reads E
    // directly and backward() writes into dE instead of dW_.
    //
    // @param E   embedding weight matrix   [vocab_size × d_model]
    // @param dE  embedding gradient matrix [vocab_size × d_model]
    void tie_weights(Matrix2D& E, Matrix2D& dE) {
        if (E.rows != vocab_size_ || E.cols != d_model_)
            throw std::invalid_argument(
                "LMHead::tie_weights: E shape must be [vocab_size × d_model]");
        tied_E_  = &E;
        tied_dE_ = &dE;
    }

    bool is_tied() const noexcept { return tied_E_ != nullptr; }

    // ── Forward ───────────────────────────────────────────

    // Compute logits and probabilities.
    //
    // @param H      [seq_len × d_model]
    // @param cache  filled for backward
    // @returns      probs  [seq_len × vocab_size]
    [[nodiscard]] Matrix2D forward(const Matrix2D& H,
                                   LMHeadCache&    cache) const
    {
        if (H.cols != d_model_)
            throw std::invalid_argument("LMHead: H.cols != d_model");

        const std::size_t seq_len = H.rows;

        // Logits = H · Wᵀ_effective + b
        // When tied: W_eff = Eᵀ  so  H · Eᵀᵀ = H · E... wait:
        // E is [vocab_size × d_model], so H · Eᵀ gives [seq × vocab] 
        Matrix2D logits(seq_len, vocab_size_);

        if (is_tied()) {
            // H [seq × d_model] · Eᵀ [d_model × vocab] -> [seq × vocab]
            logits = matmul_bt(H, *tied_E_);   // matmul_bt(A,B) = A·Bᵀ
                                                // tied_E_ is [vocab × d_model]
                                                // so Bᵀ = [d_model × vocab] 
        } else {
            logits = matmul(H, W_);            // [seq × vocab]
        }

        // Add bias
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t v = 0; v < vocab_size_; ++v)
                logits.at(i, v) += b_.at(0, v);

        // Softmax row-wise
        Matrix2D probs = logits;
        softmax_inplace(probs);

        cache.H      = H;
        cache.logits = logits;
        cache.probs  = probs;

        return probs;   // [seq_len × vocab_size]
    }

    // ── Loss + backward (combined for numerical stability) ─

    // Compute cross-entropy loss and gradient of logits in one pass.
    //
    // @param probs    [seq_len × vocab_size]  from forward()
    // @param targets  [seq_len]  token IDs (kPAD_ID positions skipped)
    // @returns        { loss (scalar), dLogits [seq_len × vocab_size] }
    struct LossAndGrad {
        float    loss;
        Matrix2D dLogits;  // [seq_len × vocab_size]
    };

    [[nodiscard]] static LossAndGrad cross_entropy(
            const Matrix2D&              probs,
            const std::vector<TokenID>&  targets)
    {
        const std::size_t seq_len = probs.rows;
        assert(targets.size() == seq_len);

        // Count real (non-padding) positions
        std::size_t n_real = 0;
        for (TokenID t : targets)
            if (t != kPAD_ID) ++n_real;

        if (n_real == 0)
            return {0.f, Matrix2D(seq_len, probs.cols, 0.f)};

        float loss = 0.f;
        Matrix2D dLogits(seq_len, probs.cols, 0.f);

        const float scale = 1.f / static_cast<float>(n_real);

        for (std::size_t i = 0; i < seq_len; ++i) {
            TokenID t = targets[i];
            if (t == kPAD_ID) continue;   // skip padding

            // loss += -log(p[i, t])
            float p_t = std::max(probs.at(i, t), 1e-9f);  // clamp for log
            loss -= std::log(p_t);

            // dLogits[i, j] = (p[i,j] - 1{j==t}) / n_real
            for (std::size_t v = 0; v < probs.cols; ++v)
                dLogits.at(i, v) = probs.at(i, v) * scale;
            dLogits.at(i, t) -= scale;   // subtract 1/n_real at target
        }

        loss *= scale;

        return {loss, std::move(dLogits)};
    }

    // ── Backward through projection ───────────────────────

    // Given dLogits from cross_entropy(), compute dH and
    // accumulate weight gradients.
    //
    // @param dLogits  [seq_len × vocab_size]
    // @param cache    from forward()
    // @returns        dH  [seq_len × d_model]
    [[nodiscard]] Matrix2D backward(const Matrix2D&   dLogits,
                                    const LMHeadCache& cache)
    {
        const std::size_t seq_len = dLogits.rows;

        // db += sum over seq positions
        for (std::size_t i = 0; i < seq_len; ++i)
            for (std::size_t v = 0; v < vocab_size_; ++v)
                db_.at(0, v) += dLogits.at(i, v);

        if (is_tied()) {
            // dE += Hᵀ · dLogits   (tied_E_ is [vocab × d_model])
            // We need [vocab × d_model] <- dLogitsᵀ · H
            // i.e. for each vocab v, d_model d:
            //   dE[v,d] += sum_i dLogits[i,v] * H[i,d]
            // = matmul_at(dLogits, H) gives [vocab × d_model] 
            auto g = matmul_at(dLogits, cache.H);
            for (std::size_t i = 0; i < tied_dE_->data.size(); ++i)
                tied_dE_->data[i] += g.data[i];
        } else {
            // dW += Hᵀ · dLogits           [d_model × vocab_size]
            auto g = matmul_at(cache.H, dLogits);
            for (std::size_t i = 0; i < dW_.data.size(); ++i)
                dW_.data[i] += g.data[i];
        }

        // dH = dLogits · Wᵀ_effective      [seq_len × d_model]
        Matrix2D dH(seq_len, d_model_);
        if (is_tied()) {
            // dLogits [seq × vocab] · tied_E_ [vocab × d_model]
            dH = matmul(dLogits, *tied_E_);
        } else {
            // dLogits [seq × vocab] · Wᵀ [vocab × d_model]
            dH = matmul_bt(dLogits, W_);
        }

        return dH;   // [seq_len × d_model]
    }

    // ── Inference helpers ─────────────────────────────────

    // Return the token ID with the highest logit at position pos.
    // Used for greedy decoding.
    [[nodiscard]] static TokenID greedy(const Matrix2D& probs,
                                        std::size_t     pos)
    {
        std::size_t best = 0;
        float best_p = probs.at(pos, 0);
        for (std::size_t v = 1; v < probs.cols; ++v)
            if (probs.at(pos, v) > best_p) {
                best_p = probs.at(pos, v);
                best   = v;
            }
        return static_cast<TokenID>(best);
    }

    // Return top-k token IDs and their probabilities at position pos.
    [[nodiscard]] static std::vector<std::pair<TokenID, float>>
    top_k(const Matrix2D& probs, std::size_t pos, std::size_t k)
    {
        std::vector<std::pair<TokenID, float>> all;
        all.reserve(probs.cols);
        for (std::size_t v = 0; v < probs.cols; ++v)
            all.push_back({static_cast<TokenID>(v), probs.at(pos, v)});

        std::partial_sort(all.begin(),
                          all.begin() + std::min(k, all.size()),
                          all.end(),
                          [](const auto& a, const auto& b){
                              return a.second > b.second;
                          });
        all.resize(std::min(k, all.size()));
        return all;
    }

    // ── Parameter update ──────────────────────────────────

    void sgd_step(float lr) {
        if (!is_tied()) {
            for (std::size_t i = 0; i < W_.data.size(); ++i)
                W_.data[i] -= lr * dW_.data[i];
        }
        for (std::size_t i = 0; i < b_.data.size(); ++i)
            b_.data[i] -= lr * db_.data[i];
    }

    void zero_grad() {
        dW_.zero();
        db_.zero();
    }

    // ── Accessors ─────────────────────────────────────────
    std::size_t d_model()    const noexcept { return d_model_; }
    std::size_t vocab_size() const noexcept { return vocab_size_; }

          Matrix2D& W()  { return W_; }
    const Matrix2D& W()  const { return W_; }
          Matrix2D& dW() { return dW_; }
          Matrix2D& db() { return db_; }

private:
    std::size_t d_model_, vocab_size_;

    Matrix2D W_;     // [d_model × vocab_size]
    Matrix2D b_;     // [1 × vocab_size]
    Matrix2D dW_, db_;

    // Weight tying pointers (null = not tied)
    Matrix2D* tied_E_;
    Matrix2D* tied_dE_;

    void init_weights(uint32_t seed) {
        std::mt19937 rng(seed);
        float limit = std::sqrt(6.f / (static_cast<float>(d_model_) +
                                       static_cast<float>(vocab_size_)));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (float& v : W_.data) v = dist(rng);
        // bias stays zero
    }
};

} // namespace miniGPT
