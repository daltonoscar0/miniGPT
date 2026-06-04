// train_test.cpp  —  Tests for AdamW, MiniGPT model, and Trainer
//
// Build (from miniGPT/):
//   make train_test && ./train_test

#include "../train/AdamW.hpp"
#include "../train/Trainer.hpp"
#include "../model/MiniGPT.hpp"
#include "../encoder/Encoder.hpp"
#include "../vocab/Vocab.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <numeric>

static int total = 0, passed = 0;

#define CHECK(expr)                                                   \
    do {                                                              \
        ++total;                                                      \
        if (!(expr)) {                                                \
            std::cerr << "FAIL  " #expr                              \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";\
        } else { ++passed; }                                          \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                         \
    do {                                                              \
        ++total;                                                      \
        float _a=(a), _b=(b);                                         \
        if (std::abs(_a - _b) > (eps)) {                              \
            std::cerr << "FAIL  " #a " ~= " #b                       \
                      << "  got " << _a << " vs " << _b              \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";\
        } else { ++passed; }                                          \
    } while(0)

#define SECTION(name) std::cout << "\n-- " name " --\n"

using namespace miniGPT;

Vocab make_vocab() {
    Vocab v;
    v.add("the+DET"); v.add("cat+N+SG"); v.add("sat+V+PAST");
    v.add("on+PREP"); v.add("mat+N+SG"); v.add("dog+N+SG");
    v.add("run+V+PAST");
    return v;
}

ModelConfig small_config(std::size_t vocab_size) {
    ModelConfig cfg;
    cfg.vocab_size  = vocab_size;
    cfg.d_model     = 16;
    cfg.n_heads     = 2;
    cfg.n_layers    = 2;
    cfg.d_ff        = 32;
    cfg.max_len     = 8;
    cfg.seed        = 42;
    cfg.tie_weights = true;
    return cfg;
}

void test_adamw_construction() {
    SECTION("AdamW construction");
    AdamW opt(1e-3f);
    CHECK(opt.step_count() == 0);
    CHECK(opt.n_params() == 0);
}

void test_adamw_invalid_lr() {
    SECTION("AdamW rejects non-positive lr");
    bool threw = false;
    try { AdamW opt(-1.f); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_adamw_step_moves_param() {
    SECTION("AdamW step moves parameter");
    Matrix2D W(1, 1, 1.f);
    Matrix2D dW(1, 1, 1.f);
    AdamW opt(0.1f, 0.9f, 0.999f, 1e-8f, 0.f);
    opt.register_param(W, dW, false);
    float before = W.at(0, 0);
    opt.step();
    CHECK(W.at(0, 0) != before);
    CHECK(opt.step_count() == 1);
}

void test_adamw_bias_correction() {
    SECTION("AdamW update magnitude at t=1 with beta=0");
    Matrix2D W(1, 1, 0.f);
    Matrix2D dW(1, 1, 1.f);
    AdamW opt(0.1f, 0.f, 0.f, 1e-8f, 0.f);
    opt.register_param(W, dW, false);
    opt.step();
    CHECK_NEAR(W.at(0, 0), -0.1f, 0.01f);
}

void test_adamw_weight_decay() {
    SECTION("AdamW weight decay shrinks parameters");
    Matrix2D W(1, 1, 1.f);
    Matrix2D dW(1, 1, 0.f);
    AdamW opt(1e-4f, 0.9f, 0.999f, 1e-8f, 0.5f);
    opt.register_param(W, dW, true);
    float before = W.at(0, 0);
    opt.step();
    CHECK(W.at(0, 0) < before);
}

void test_adamw_no_decay_for_bias() {
    SECTION("decay=false skips weight decay");
    Matrix2D W(1, 1, 10.f);
    Matrix2D dW(1, 1, 0.f);
    AdamW opt(1e-6f, 0.9f, 0.999f, 1e-8f, 100.f);
    opt.register_param(W, dW, false);
    opt.step();
    CHECK(std::abs(W.at(0, 0) - 10.f) < 0.01f);
}

void test_model_construction() {
    SECTION("MiniGPT construction");
    auto cfg = small_config(11);
    MiniGPT model(cfg);
    model.print_config();
    CHECK(model.n_layers() == 2);
    CHECK(model.param_count() > 0);
    std::cout << "  param_count = " << model.param_count() << "\n";
}

void test_model_forward_shape() {
    SECTION("Model forward output shape");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    auto seq = enc.encode({"the+DET", "cat+N+SG", "sat+V+PAST"});
    ModelCache cache;
    auto probs = model.forward(seq.ids, seq.mask, cache);
    CHECK(probs.rows == cfg.max_len);
    CHECK(probs.cols == vocab.size());
}

void test_model_probs_sum_to_one() {
    SECTION("Output probabilities sum to 1");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    auto seq = enc.encode({"the+DET", "cat+N+SG"});
    ModelCache cache;
    auto probs = model.forward(seq.ids, seq.mask, cache);
    for (std::size_t i = 0; i < seq.n_real; ++i) {
        float s = 0.f;
        for (std::size_t v = 0; v < vocab.size(); ++v) s += probs.at(i, v);
        CHECK_NEAR(s, 1.f, 1e-3f);
    }
}

void test_model_forward_loss_finite() {
    SECTION("forward_loss() returns finite positive loss");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    auto pair = enc.encode_lm_pair({"the+DET", "cat+N+SG", "sat+V+PAST"});
    ModelCache cache;
    float loss = model.forward_loss(pair.input.ids, pair.input.mask,
                                    pair.target.ids, cache);
    CHECK(std::isfinite(loss));
    CHECK(loss > 0.f);
    std::cout << "  initial loss = " << loss << "\n";
}

void test_model_backward_no_nan() {
    SECTION("backward() produces no NaN gradients");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    auto pair = enc.encode_lm_pair({"the+DET", "cat+N+SG", "sat+V+PAST"});
    ModelCache cache;
    model.zero_grad();
    model.forward_loss(pair.input.ids, pair.input.mask,
                       pair.target.ids, cache);
    model.backward(cache);
    for (float v : model.embed().dE().data)    CHECK(!std::isnan(v));
    for (float v : model.block(0).mha().dW_Q().data) CHECK(!std::isnan(v));
}

void test_model_grad_nonzero() {
    SECTION("Gradients are non-zero after backward()");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    auto pair = enc.encode_lm_pair({"the+DET", "cat+N+SG"});
    ModelCache cache;
    model.zero_grad();
    model.forward_loss(pair.input.ids, pair.input.mask,
                       pair.target.ids, cache);
    model.backward(cache);
    bool any = false;
    for (float v : model.block(0).mha().dW_Q().data)
        if (v != 0.f) { any = true; break; }
    CHECK(any);
}

void test_clip_grad_norm() {
    SECTION("clip_grad_norm() returns non-negative norm");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    auto pair = enc.encode_lm_pair({"the+DET", "cat+N+SG", "sat+V+PAST"});
    ModelCache cache;
    model.zero_grad();
    model.forward_loss(pair.input.ids, pair.input.mask,
                       pair.target.ids, cache);
    model.backward(cache);
    float norm = model.clip_grad_norm(1.0f);
    std::cout << "  grad norm = " << norm << "\n";
    CHECK(norm >= 0.f);
}

void test_trainer_loss_decreases() {
    SECTION("Trainer: loss decreases over 50 steps on fixed sequence");
    auto vocab = make_vocab();
    vocab.freeze();
    auto cfg = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    TrainerConfig tcfg;
    tcfg.lr = 1e-2f;
    tcfg.log_every = 9999;
    Trainer trainer(model, enc, tcfg);
    auto pair = enc.encode_lm_pair(
        {"the+DET","cat+N+SG","sat+V+PAST","on+PREP","the+DET","mat+N+SG"});
    float loss_first = trainer.train_step(pair.input.ids, pair.input.mask,
                                          pair.target.ids);
    float loss_last = loss_first;
    for (int i = 0; i < 50; ++i)
        loss_last = trainer.train_step(pair.input.ids, pair.input.mask,
                                       pair.target.ids);
    std::cout << "  loss: " << loss_first << " -> " << loss_last << "\n";
    CHECK(loss_last < loss_first);
}

void test_trainer_step_counter() {
    SECTION("Trainer step counter increments");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Trainer trainer(model, enc);
    auto pair = enc.encode_lm_pair({"the+DET", "cat+N+SG"});
    for (int i = 0; i < 5; ++i)
        trainer.train_step(pair.input.ids, pair.input.mask, pair.target.ids);
    CHECK(trainer.step() == 5);
}

void test_trainer_corpus() {
    SECTION("Trainer::train() runs without crash");
    auto vocab = make_vocab();
    vocab.freeze();
    auto cfg = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    TrainerConfig tcfg;
    tcfg.n_epochs  = 2;
    tcfg.log_every = 9999;
    Trainer trainer(model, enc, tcfg);
    std::vector<std::string> corpus = {
        "the+DET","cat+N+SG","sat+V+PAST","on+PREP","the+DET","mat+N+SG",
        "the+DET","dog+N+SG","run+V+PAST","on+PREP","the+DET","mat+N+SG",
        "the+DET","cat+N+SG","sat+V+PAST","on+PREP","the+DET","mat+N+SG"
    };
    bool threw = false;
    try { trainer.train(corpus); } catch (...) { threw = true; }
    CHECK(!threw);
    CHECK(trainer.step() > 0);
}

int main() {
    std::cout << "miniGPT -- Training loop tests\n";
    std::cout << "==============================\n";
    test_adamw_construction();
    test_adamw_invalid_lr();
    test_adamw_step_moves_param();
    test_adamw_bias_correction();
    test_adamw_weight_decay();
    test_adamw_no_decay_for_bias();
    test_model_construction();
    test_model_forward_shape();
    test_model_probs_sum_to_one();
    test_model_forward_loss_finite();
    test_model_backward_no_nan();
    test_model_grad_nonzero();
    test_clip_grad_norm();
    test_trainer_loss_decreases();
    test_trainer_step_counter();
    test_trainer_corpus();
    std::cout << "\n==============================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total) std::cout << "  ALL PASS\n";
    else std::cout << "  " << (total - passed) << " FAILED\n";
    return (passed == total) ? 0 : 1;
}
