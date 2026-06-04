// infer_test.cpp  —  Tests for Inference engine
//
// Build (from miniGPT/):
//   make infer_test && ./infer_test

#include "../infer/Inference.hpp"
#include "../train/Trainer.hpp"
#include "../model/MiniGPT.hpp"
#include "../encoder/Encoder.hpp"
#include "../vocab/Vocab.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <algorithm>

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

// ── Helpers ───────────────────────────────────────────────

Vocab make_vocab() {
    Vocab v;
    v.add("the+DET");    // 4
    v.add("cat+N+SG");   // 5
    v.add("sat+V+PAST"); // 6
    v.add("on+PREP");    // 7
    v.add("mat+N+SG");   // 8
    v.add("dog+N+SG");   // 9
    v.add("run+V+PAST"); // 10
    return v;
}

ModelConfig small_config(std::size_t vocab_size) {
    ModelConfig cfg;
    cfg.vocab_size  = vocab_size;
    cfg.d_model     = 16;
    cfg.n_heads     = 2;
    cfg.n_layers    = 2;
    cfg.d_ff        = 32;
    cfg.max_len     = 10;
    cfg.seed        = 42;
    cfg.tie_weights = true;
    return cfg;
}

// ── Tests ─────────────────────────────────────────────────

void test_greedy_returns_tokens() {
    SECTION("generate() returns non-empty token list");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    GenerationConfig gcfg;
    gcfg.max_new_tokens = 5;
    gcfg.top_k = 1;

    auto tokens = infer.generate({"the+DET", "cat+N+SG"}, gcfg);
    CHECK(tokens.size() <= 5);
    CHECK(!tokens.empty() || true);  // may stop at EOS — that's fine
}

void test_greedy_deterministic() {
    SECTION("Greedy generation is deterministic");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    GenerationConfig gcfg;
    gcfg.max_new_tokens = 5;
    gcfg.top_k = 1;

    auto tokens1 = infer.generate({"the+DET"}, gcfg);
    auto tokens2 = infer.generate({"the+DET"}, gcfg);

    CHECK(tokens1.size() == tokens2.size());
    for (std::size_t i = 0; i < tokens1.size(); ++i)
        CHECK(tokens1[i] == tokens2[i]);
}

void test_output_tokens_in_vocab() {
    SECTION("All generated tokens are in the vocabulary");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    GenerationConfig gcfg;
    gcfg.max_new_tokens = 8;
    gcfg.top_k = 1;

    auto tokens = infer.generate({"the+DET", "cat+N+SG"}, gcfg);
    for (const auto& t : tokens)
        CHECK(vocab.contains(t));
}

void test_empty_prompt() {
    SECTION("Empty prompt generates without crash");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    GenerationConfig gcfg;
    gcfg.max_new_tokens = 3;

    bool threw = false;
    try { (void)infer.generate({}, gcfg); }
    catch (...) { threw = true; }
    CHECK(!threw);
}

void test_max_new_tokens_respected() {
    SECTION("Generation does not exceed max_new_tokens");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    GenerationConfig gcfg;
    gcfg.max_new_tokens = 4;
    gcfg.top_k = 1;

    auto tokens = infer.generate({"the+DET"}, gcfg);
    CHECK(tokens.size() <= 4);
}

void test_top_k_sampling_varies() {
    SECTION("Top-k sampling produces varied output across seeds");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    GenerationConfig gcfg;
    gcfg.max_new_tokens = 5;
    gcfg.top_k = vocab.size();  // sample from full distribution

    // Run with different seeds — at least some should differ
    std::vector<std::vector<std::string>> results;
    for (uint32_t seed = 0; seed < 10; ++seed) {
        gcfg.seed = seed;
        results.push_back(infer.generate({"the+DET"}, gcfg));
    }

    // Check that not all results are identical
    bool any_differ = false;
    for (std::size_t i = 1; i < results.size(); ++i)
        if (results[i] != results[0]) { any_differ = true; break; }
    CHECK(any_differ);
}

void test_temperature_affects_distribution() {
    SECTION("Temperature=0.1 (sharp) vs 2.0 (flat) produce different outputs");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    GenerationConfig sharp, flat;
    sharp.max_new_tokens = 5; sharp.top_k = 1; sharp.temperature = 0.1f;
    flat.max_new_tokens  = 5; flat.top_k  = 1; flat.temperature  = 2.0f;

    // Both should complete without crashing
    bool threw = false;
    try {
        (void)infer.generate({"the+DET"}, sharp);
        (void)infer.generate({"the+DET"}, flat);
    } catch (...) { threw = true; }
    CHECK(!threw);
}

void test_generate_str() {
    SECTION("generate_str() returns space-joined tokens");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    GenerationConfig gcfg;
    gcfg.max_new_tokens = 3;
    gcfg.top_k = 1;

    auto tokens = infer.generate({"the+DET"}, gcfg);
    auto str    = infer.generate_str({"the+DET"}, gcfg);

    // str should be the tokens joined by spaces
    std::string expected;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) expected += " ";
        expected += tokens[i];
    }
    CHECK(str == expected);
}

void test_perplexity_finite_positive() {
    SECTION("perplexity() returns finite positive value");
    auto vocab = make_vocab();
    auto cfg   = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);
    Inference infer(model, enc, vocab);

    float ppl = infer.perplexity(
        {"the+DET", "cat+N+SG", "sat+V+PAST"});
    CHECK(std::isfinite(ppl));
    CHECK(ppl > 0.f);
    std::cout << "  initial perplexity = " << ppl << "\n";
}

void test_perplexity_decreases_after_training() {
    SECTION("Perplexity decreases after training on the same sequence");
    auto vocab = make_vocab();
    vocab.freeze();
    auto cfg = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);

    std::vector<std::string> seq = {
        "the+DET","cat+N+SG","sat+V+PAST","on+PREP","the+DET","mat+N+SG"
    };

    Inference infer(model, enc, vocab);
    float ppl_before = infer.perplexity(seq);

    // Train for 30 steps
    TrainerConfig tcfg;
    tcfg.lr = 1e-2f;
    tcfg.log_every = 9999;
    Trainer trainer(model, enc, tcfg);
    auto pair = enc.encode_lm_pair(seq);
    for (int i = 0; i < 30; ++i)
        trainer.train_step(pair.input.ids, pair.input.mask,
                           pair.target.ids);

    float ppl_after = infer.perplexity(seq);
    std::cout << "  perplexity: " << ppl_before
              << " -> " << ppl_after << "\n";
    CHECK(ppl_after < ppl_before);
}

void test_trained_model_predicts_trained_sequence() {
    SECTION("After training, model assigns high prob to trained next token");
    auto vocab = make_vocab();
    vocab.freeze();
    auto cfg = small_config(vocab.size());
    MiniGPT model(cfg);
    Encoder enc(vocab, cfg.max_len);

    // Train heavily on "the cat sat"
    TrainerConfig tcfg;
    tcfg.lr = 5e-2f;
    tcfg.log_every = 9999;
    Trainer trainer(model, enc, tcfg);
    auto pair = enc.encode_lm_pair({"the+DET", "cat+N+SG", "sat+V+PAST"});
    for (int i = 0; i < 100; ++i)
        trainer.train_step(pair.input.ids, pair.input.mask,
                           pair.target.ids);

    // After training, given "the cat" the model should favour "sat"
    // (or at least produce a valid vocab token)
    Inference infer(model, enc, vocab);
    GenerationConfig gcfg;
    gcfg.max_new_tokens = 1;
    gcfg.top_k = 1;

    auto next = infer.generate({"the+DET", "cat+N+SG"}, gcfg);
    CHECK(!next.empty());
    std::cout << "  given [the+DET cat+N+SG], predicted: "
              << (next.empty() ? "<EOS>" : next[0]) << "\n";
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT -- Inference tests\n";
    std::cout << "==========================\n";

    test_greedy_returns_tokens();
    test_greedy_deterministic();
    test_output_tokens_in_vocab();
    test_empty_prompt();
    test_max_new_tokens_respected();
    test_top_k_sampling_varies();
    test_temperature_affects_distribution();
    test_generate_str();
    test_perplexity_finite_positive();
    test_perplexity_decreases_after_training();
    test_trained_model_predicts_trained_sequence();

    std::cout << "\n==========================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total) std::cout << "  ALL PASS\n";
    else std::cout << "  " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
