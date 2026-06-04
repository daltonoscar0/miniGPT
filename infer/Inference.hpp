#pragma once

// ============================================================
//  Inference.hpp  —  Text generation from a trained MiniGPT
//
//  Takes a prompt (as analysis strings), encodes it, runs the
//  model forward pass, picks the next token, appends it to the
//  sequence, and repeats until <EOS> or max_new_tokens reached.
//
//  Two generation strategies
//  ──────────────────────────
//  Greedy:   always pick the highest-probability token.
//            Deterministic. Fast. Tends to produce repetitive
//            output for long sequences.
//
//  Top-k sampling: sample from the k highest-probability
//            tokens. Non-deterministic. More varied output.
//            k=1 is equivalent to greedy.
//
//  Autoregressive loop
//  ────────────────────
//  context = encode(prompt)             // [max_len]
//
//  for step in 0..max_new_tokens:
//    probs = model.forward(context)     // [max_len × vocab_size]
//    next  = pick(probs[last_real_pos]) // scalar TokenID
//    if next == EOS: break
//    append next to context
//    shift context left if full (sliding window)
//
//  Output: sequence of TokenIDs (including prompt)
//          decoded back to analysis strings via Vocab
// ============================================================

#include "../model/MiniGPT.hpp"
#include "../encoder/Encoder.hpp"
#include "../vocab/Vocab.hpp"
#include "../lm_head/LMHead.hpp"

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace miniGPT {

// ── GenerationConfig ──────────────────────────────────────
struct GenerationConfig {
    std::size_t max_new_tokens = 50;
    std::size_t top_k          = 1;      // 1 = greedy
    float       temperature    = 1.0f;   // scale logits before softmax
    uint32_t    seed           = 0;      // for sampling
    bool        print_tokens   = false;  // stream tokens as generated
};

// ── Inference ─────────────────────────────────────────────
class Inference {
public:
    // ── Construction ──────────────────────────────────────

    Inference(MiniGPT&       model,
              const Encoder& encoder,
              const Vocab&   vocab)
        : model_(model)
        , encoder_(encoder)
        , vocab_(vocab)
    {}

    // ── Generate ──────────────────────────────────────────

    // Generate tokens given a prompt of analysis strings.
    //
    // @param prompt   analysis strings, e.g. {"the+DET", "cat+N+SG"}
    // @param cfg      generation parameters
    // @returns        generated analysis strings (not including prompt)
    [[nodiscard]] std::vector<std::string> generate(
            const std::vector<std::string>& prompt,
            const GenerationConfig&         cfg = GenerationConfig{})
    {
        const std::size_t max_len = encoder_.max_len();

        // Encode prompt into a working context buffer
        // We maintain a sliding window of token IDs
        std::vector<TokenID> context;
        context.reserve(max_len);

        // Push BOS
        context.push_back(kBOS_ID);

        // Push prompt tokens (truncate if needed)
        for (const auto& a : prompt) {
            if (context.size() >= max_len - 1) break;  // leave room for EOS
            context.push_back(vocab_.encode(a));
        }

        std::vector<std::string> generated;
        generated.reserve(cfg.max_new_tokens);

        std::mt19937 rng(cfg.seed);

        for (std::size_t step = 0; step < cfg.max_new_tokens; ++step) {
            // Pad context to max_len
            std::vector<TokenID> padded(max_len, kPAD_ID);
            std::vector<bool>    mask(max_len, false);

            std::size_t n = std::min(context.size(), max_len);
            for (std::size_t i = 0; i < n; ++i) {
                padded[i] = context[i];
                mask[i]   = true;
            }

            // Forward pass
            ModelCache cache;
            auto probs = model_.forward(padded, mask, cache);

            // Pick generation position: last real token
            std::size_t gen_pos = n - 1;

            // Apply temperature scaling to logits at gen_pos
            // (work from probs — convert back via log, scale, re-softmax)
            // Clamp temperature to a small positive floor: temperature=0
            // would divide by zero and produce NaN/Inf logits.
            const float temp = std::max(cfg.temperature, 1e-6f);
            std::vector<float> logits(vocab_.size());
            for (std::size_t v = 0; v < vocab_.size(); ++v)
                logits[v] = std::log(std::max(probs.at(gen_pos, v), 1e-9f))
                            / temp;

            // Re-softmax after temperature scaling
            float maxl = *std::max_element(logits.begin(), logits.end());
            float sum  = 0.f;
            for (float& l : logits) { l = std::exp(l - maxl); sum += l; }
            for (float& l : logits) l /= sum;

            // Pick next token
            TokenID next;
            if (cfg.top_k <= 1) {
                // Greedy
                next = static_cast<TokenID>(
                    std::max_element(logits.begin(), logits.end())
                    - logits.begin());
            } else {
                // Top-k sampling
                next = sample_top_k(logits, cfg.top_k, rng);
            }

            // Stop at EOS
            if (next == kEOS_ID) break;

            // Skip PAD and UNK in generation
            if (next == kPAD_ID) break;

            std::string token_str = vocab_.decode(next);
            generated.push_back(token_str);

            if (cfg.print_tokens)
                std::cout << token_str << " " << std::flush;

            // Append to context, sliding window if full
            if (context.size() < max_len - 1) {
                context.push_back(next);
            } else {
                // Shift left, dropping oldest non-BOS token
                // Keep BOS at position 0
                context.erase(context.begin() + 1);
                context.push_back(next);
            }
        }

        if (cfg.print_tokens) std::cout << "\n";

        return generated;
    }

    // ── Convenience: generate as a single string ──────────

    [[nodiscard]] std::string generate_str(
            const std::vector<std::string>& prompt,
            const GenerationConfig&         cfg = GenerationConfig{})
    {
        auto tokens = generate(prompt, cfg);
        std::string out;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) out += " ";
            out += tokens[i];
        }
        return out;
    }

    // ── Perplexity ────────────────────────────────────────

    // Compute perplexity of a sequence under the model.
    // Perplexity = exp(average cross-entropy loss).
    // Lower is better — a perplexity of V means the model is
    // as uncertain as a uniform distribution over V tokens.
    //
    // @param analyses  full sequence of analysis strings
    // @returns         perplexity (scalar)
    [[nodiscard]] float perplexity(
            const std::vector<std::string>& analyses)
    {
        if (analyses.empty()) return 0.f;

        auto pair = encoder_.encode_lm_pair(analyses);

        ModelCache cache;
        float loss = model_.forward_loss(
            pair.input.ids, pair.input.mask,
            pair.target.ids, cache);

        // Don't backprop — this is inference only
        model_.zero_grad();

        return std::exp(loss);
    }

private:
    MiniGPT&       model_;
    const Encoder& encoder_;
    const Vocab&   vocab_;

    // ── Top-k sampling ────────────────────────────────────
    static TokenID sample_top_k(const std::vector<float>& probs,
                                  std::size_t               k,
                                  std::mt19937&             rng)
    {
        // Build (index, prob) pairs and partial-sort to get top-k
        std::vector<std::pair<std::size_t, float>> indexed;
        indexed.reserve(probs.size());
        for (std::size_t i = 0; i < probs.size(); ++i)
            indexed.push_back({i, probs[i]});

        k = std::min(k, probs.size());
        std::partial_sort(indexed.begin(), indexed.begin() + k,
                          indexed.end(),
                          [](const auto& a, const auto& b){
                              return a.second > b.second;
                          });
        indexed.resize(k);

        // Renormalise top-k probabilities
        float sum = 0.f;
        for (auto& [idx, p] : indexed) sum += p;
        for (auto& [idx, p] : indexed) p /= sum;

        // Sample
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        float r = dist(rng);
        float cumsum = 0.f;
        for (auto& [idx, p] : indexed) {
            cumsum += p;
            if (r <= cumsum)
                return static_cast<TokenID>(idx);
        }
        return static_cast<TokenID>(indexed.back().first);
    }
};

} // namespace miniGPT
