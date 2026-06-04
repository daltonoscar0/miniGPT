#pragma once

// ============================================================
//  Trainer.hpp  —  Training loop for MiniGPT
//
//  Wires together:
//    - MiniGPT model (forward + backward)
//    - AdamW optimizer
//    - Encoder (produces LM pairs from analysis sequences)
//    - Gradient clipping
//
//  Training step (one batch of one sequence)
//  ──────────────────────────────────────────
//  1. Encode analyses → (input_ids, target_ids, mask)
//  2. model.zero_grad()
//  3. loss = model.forward_loss(input_ids, mask, target_ids)
//  4. model.backward(cache)
//  5. model.clip_grad_norm(max_grad_norm)
//  6. optimizer.step()
// ============================================================

#include "AdamW.hpp"
#include "../model/MiniGPT.hpp"
#include "../encoder/Encoder.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

namespace miniGPT {

struct TrainerConfig {
    float       lr            = 3e-4f;
    float       weight_decay  = 0.01f;
    float       max_grad_norm = 1.0f;
    std::size_t n_epochs      = 3;
    std::size_t log_every     = 100;
    uint32_t    data_seed     = 0;
};

class Trainer {
public:
    Trainer(MiniGPT&             model,
            const Encoder&       encoder,
            const TrainerConfig& cfg = TrainerConfig{})
        : model_(model)
        , encoder_(encoder)
        , cfg_(cfg)
        , opt_(cfg.lr, 0.9f, 0.999f, 1e-8f, cfg.weight_decay)
        , step_(0)
        , running_loss_(0.f)
        , running_count_(0)
    {
        register_all_params();
    }

    // Single training step — returns loss
    float train_step(const std::vector<TokenID>& input_ids,
                     const std::vector<bool>&    mask,
                     const std::vector<TokenID>& target_ids)
    {
        model_.zero_grad();
        ModelCache cache;
        float loss = model_.forward_loss(input_ids, mask, target_ids, cache);
        model_.backward(cache);
        model_.clip_grad_norm(cfg_.max_grad_norm);
        opt_.step();
        ++step_;
        running_loss_  += loss;
        running_count_ += 1;
        return loss;
    }

    // Train on a flat list of analysis strings for n_epochs
    void train(const std::vector<std::string>& analyses)
    {
        const std::size_t win_size = encoder_.max_len();

        std::vector<std::size_t> starts;
        if (analyses.size() >= win_size)
            for (std::size_t i = 0; i + win_size <= analyses.size(); ++i)
                starts.push_back(i);

        if (starts.empty()) {
            std::cout << "Corpus too short to form any training windows.\n";
            return;
        }

        std::mt19937 rng(cfg_.data_seed);

        for (std::size_t epoch = 0; epoch < cfg_.n_epochs; ++epoch) {
            std::shuffle(starts.begin(), starts.end(), rng);

            float epoch_loss  = 0.f;
            std::size_t steps = 0;

            for (std::size_t start : starts) {
                std::vector<std::string> window(
                    analyses.begin() + start,
                    analyses.begin() + start + win_size);

                auto pair = encoder_.encode_lm_pair(window);
                float loss = train_step(pair.input.ids,
                                        pair.input.mask,
                                        pair.target.ids);
                epoch_loss += loss;
                ++steps;

                if (step_ % cfg_.log_every == 0) {
                    float avg = running_loss_ /
                                static_cast<float>(running_count_);
                    std::cout << "step " << step_
                              << "  loss " << avg
                              << "  lr "   << opt_.lr() << "\n";
                    running_loss_  = 0.f;
                    running_count_ = 0;
                }
            }

            std::cout << "── epoch " << (epoch + 1)
                      << "/" << cfg_.n_epochs
                      << "  avg_loss "
                      << epoch_loss / static_cast<float>(steps) << "\n";
        }
    }

    int    step()     const { return step_; }
    AdamW& optimizer()      { return opt_; }

private:
    MiniGPT&       model_;
    const Encoder& encoder_;
    TrainerConfig  cfg_;
    AdamW          opt_;
    int            step_;
    float          running_loss_;
    int            running_count_;

    void register_all_params() {
        auto& m = model_;

        // Embedding — no weight decay
        opt_.register_param(m.embed().E(), m.embed().dE(), false);

        // Transformer blocks
        for (std::size_t l = 0; l < m.n_layers(); ++l) {
            auto& blk = m.block(l);
            opt_.register_param(blk.mha().W_Q(), blk.mha().dW_Q(), true);
            opt_.register_param(blk.mha().W_K(), blk.mha().dW_K(), true);
            opt_.register_param(blk.mha().W_V(), blk.mha().dW_V(), true);
            opt_.register_param(blk.mha().W_O(), blk.mha().dW_O(), true);
            opt_.register_param(blk.ffn().W1(), blk.ffn().dW1(), true);
            opt_.register_param(blk.ffn().W2(), blk.ffn().dW2(), true);
            opt_.register_param(blk.ffn().db1(), blk.ffn().db1(), false);
            opt_.register_param(blk.ffn().db2(), blk.ffn().db2(), false);
            opt_.register_param(blk.ln1().gamma(), blk.ln1().dgamma(), false);
            opt_.register_param(blk.ln1().beta(),  blk.ln1().dbeta(),  false);
            opt_.register_param(blk.ln2().gamma(), blk.ln2().dgamma(), false);
            opt_.register_param(blk.ln2().beta(),  blk.ln2().dbeta(),  false);
        }

        // Final LayerNorm
        opt_.register_param(m.final_ln().gamma(), m.final_ln().dgamma(), false);
        opt_.register_param(m.final_ln().beta(),  m.final_ln().dbeta(),  false);

        // LM head bias only (W is tied to embedding)
        opt_.register_param(m.lm_head().db(), m.lm_head().db(), false);
    }
};

} // namespace miniGPT
