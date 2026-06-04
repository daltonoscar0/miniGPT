// lmhead_test.cpp  —  Unit tests for LMHead
//
// Build (from miniGPT/):
//   make lmhead_test && ./lmhead_test

#include "../lm_head/LMHead.hpp"
#include "../embed/Embedding.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cassert>

// ── Tiny test harness ─────────────────────────────────────
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
            std::cerr << "FAIL  " #a " ≈ " #b                        \
                      << "  got " << _a << " vs " << _b              \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";\
        } else { ++passed; }                                          \
    } while(0)

#define SECTION(name) std::cout << "\n── " name " ──\n"

using namespace miniGPT;

// ── Helpers ───────────────────────────────────────────────

float row_sum(const Matrix2D& M, std::size_t r) {
    float s = 0.f;
    for (std::size_t c = 0; c < M.cols; ++c) s += M.at(r, c);
    return s;
}

bool mat_any_nonzero(const Matrix2D& M) {
    for (float v : M.data) if (v != 0.f) return true;
    return false;
}

Matrix2D make_H(std::size_t seq, std::size_t d) {
    Matrix2D H(seq, d);
    for (std::size_t i = 0; i < H.data.size(); ++i)
        H.data[i] = static_cast<float>((i * 7 + 3) % 13) * 0.1f - 0.6f;
    return H;
}

// ── Tests ─────────────────────────────────────────────────

void test_construction() {
    SECTION("Construction");
    LMHead lm(128, 1000, 42);
    CHECK(lm.d_model()    == 128);
    CHECK(lm.vocab_size() == 1000);
    CHECK(lm.W().rows == 128 && lm.W().cols == 1000);
}

void test_invalid_construction() {
    SECTION("Zero dimension throws");
    bool threw = false;
    try { LMHead lm(0, 100); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { LMHead lm(128, 0); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_forward_output_shape() {
    SECTION("forward() output shape [seq_len × vocab_size]");
    LMHead lm(16, 50, 42);
    auto H = make_H(4, 16);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    CHECK(probs.rows == 4);
    CHECK(probs.cols == 50);
}

void test_probs_sum_to_one() {
    SECTION("Probability rows sum to 1");
    LMHead lm(16, 50, 42);
    auto H = make_H(5, 16);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    for (std::size_t i = 0; i < 5; ++i)
        CHECK_NEAR(row_sum(probs, i), 1.f, 1e-4f);
}

void test_probs_nonnegative() {
    SECTION("All probabilities are non-negative");
    LMHead lm(16, 50, 42);
    auto H = make_H(4, 16);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    for (float v : probs.data)
        CHECK(v >= 0.f);
}

void test_cache_shapes() {
    SECTION("Cache stores H, logits, probs with correct shapes");
    LMHead lm(8, 20, 42);
    auto H = make_H(3, 8);
    LMHeadCache cache;
    (void)lm.forward(H, cache);
    CHECK(cache.H.rows == 3      && cache.H.cols == 8);
    CHECK(cache.logits.rows == 3 && cache.logits.cols == 20);
    CHECK(cache.probs.rows == 3  && cache.probs.cols == 20);
}

void test_cross_entropy_perfect_prediction() {
    SECTION("Cross-entropy is zero for perfect prediction");
    // Build a probs matrix where target token has prob ~1
    std::size_t seq = 3, vocab = 10;
    Matrix2D probs(seq, vocab, 1e-9f);
    std::vector<TokenID> targets = {4, 7, 2};
    for (std::size_t i = 0; i < seq; ++i)
        probs.at(i, targets[i]) = 1.f - (vocab - 1) * 1e-9f;

    auto result = LMHead::cross_entropy(probs, targets);
    CHECK_NEAR(result.loss, 0.f, 0.01f);
}

void test_cross_entropy_uniform_prediction() {
    SECTION("Cross-entropy for uniform dist = log(vocab_size)");
    std::size_t seq = 2, vocab = 8;
    Matrix2D probs(seq, vocab, 1.f / vocab);
    std::vector<TokenID> targets = {3, 5};

    auto result = LMHead::cross_entropy(probs, targets);
    float expected = std::log(static_cast<float>(vocab));
    CHECK_NEAR(result.loss, expected, 0.01f);
}

void test_cross_entropy_skips_padding() {
    SECTION("Cross-entropy skips padding positions");
    std::size_t seq = 4, vocab = 8;
    Matrix2D probs(seq, vocab, 1.f / vocab);

    // Two real tokens, two padding
    std::vector<TokenID> targets = {3, kPAD_ID, 5, kPAD_ID};
    auto result_with_pad = LMHead::cross_entropy(probs, targets);

    // All real
    std::vector<TokenID> targets_all = {3, 3, 5, 5};
    auto result_all = LMHead::cross_entropy(probs, targets_all);

    // Both should give log(vocab) since distribution is uniform
    CHECK_NEAR(result_with_pad.loss, result_all.loss, 0.01f);
}

void test_cross_entropy_dlogits_shape() {
    SECTION("dLogits shape is [seq_len × vocab_size]");
    LMHead lm(8, 20, 42);
    auto H = make_H(4, 8);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    std::vector<TokenID> targets = {3, 7, 1, 5};
    auto result = LMHead::cross_entropy(probs, targets);
    CHECK(result.dLogits.rows == 4);
    CHECK(result.dLogits.cols == 20);
}

void test_cross_entropy_dlogits_sum_zero() {
    SECTION("dLogits rows sum to 0 for real positions");
    // d/dlogits CE+softmax = p - one_hot, which sums to 0
    LMHead lm(8, 10, 42);
    auto H = make_H(3, 8);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    std::vector<TokenID> targets = {2, 5, 8};
    auto result = LMHead::cross_entropy(probs, targets);

    for (std::size_t i = 0; i < 3; ++i)
        CHECK_NEAR(row_sum(result.dLogits, i), 0.f, 1e-5f);
}

void test_cross_entropy_dlogits_pad_is_zero() {
    SECTION("dLogits is zero at padding positions");
    LMHead lm(8, 10, 42);
    auto H = make_H(4, 8);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    std::vector<TokenID> targets = {2, kPAD_ID, 8, kPAD_ID};
    auto result = LMHead::cross_entropy(probs, targets);

    // Positions 1 and 3 are padding — dLogits must be zero
    for (std::size_t v = 0; v < 10; ++v) {
        CHECK(result.dLogits.at(1, v) == 0.f);
        CHECK(result.dLogits.at(3, v) == 0.f);
    }
}

void test_backward_dH_shape() {
    SECTION("backward() dH shape [seq_len × d_model]");
    LMHead lm(8, 20, 42);
    auto H = make_H(4, 8);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    std::vector<TokenID> targets = {3, 7, 1, 5};
    auto result = LMHead::cross_entropy(probs, targets);
    auto dH = lm.backward(result.dLogits, cache);
    CHECK(dH.rows == 4 && dH.cols == 8);
}

void test_backward_accumulates_dW() {
    SECTION("backward() accumulates dW");
    LMHead lm(8, 20, 42);
    lm.zero_grad();
    auto H = make_H(3, 8);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    std::vector<TokenID> targets = {3, 7, 1};
    auto result = LMHead::cross_entropy(probs, targets);
    (void)lm.backward(result.dLogits, cache);
    CHECK(mat_any_nonzero(lm.dW()));
    CHECK(mat_any_nonzero(lm.db()));
}

void test_sgd_step_updates_W() {
    SECTION("sgd_step() updates W and b");
    LMHead lm(8, 20, 42);
    lm.zero_grad();
    float w_before = lm.W().at(0, 0);

    auto H = make_H(3, 8);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    std::vector<TokenID> targets = {3, 7, 1};
    auto result = LMHead::cross_entropy(probs, targets);
    (void)lm.backward(result.dLogits, cache);
    lm.sgd_step(0.01f);

    CHECK(std::abs(lm.W().at(0, 0) - w_before) > 1e-7f);
}

void test_greedy_returns_argmax() {
    SECTION("greedy() returns the index with highest probability");
    std::size_t seq = 2, vocab = 5;
    Matrix2D probs(seq, vocab, 0.1f);
    probs.at(0, 3) = 0.7f;   // highest at position 0 is vocab ID 3
    probs.at(1, 1) = 0.6f;   // highest at position 1 is vocab ID 1

    CHECK(LMHead::greedy(probs, 0) == 3);
    CHECK(LMHead::greedy(probs, 1) == 1);
}

void test_top_k() {
    SECTION("top_k() returns k highest-probability tokens in order");
    std::size_t seq = 1, vocab = 6;
    Matrix2D probs(seq, vocab, 0.f);
    probs.at(0, 0) = 0.05f;
    probs.at(0, 1) = 0.30f;
    probs.at(0, 2) = 0.10f;
    probs.at(0, 3) = 0.40f;
    probs.at(0, 4) = 0.10f;
    probs.at(0, 5) = 0.05f;

    auto topk = LMHead::top_k(probs, 0, 3);
    CHECK(topk.size() == 3);
    CHECK(topk[0].first == 3);   // highest: ID 3, p=0.40
    CHECK(topk[1].first == 1);   // second:  ID 1, p=0.30
    CHECK(topk[2].first == 2 || topk[2].first == 4);  // tied at 0.10
}

void test_weight_tying() {
    SECTION("Weight tying: forward uses E instead of W");
    std::size_t d = 8, vocab = 20;
    LMHead lm(d, vocab, 42);

    // Create embedding table
    Matrix2D E(vocab, d);
    Matrix2D dE(vocab, d, 0.f);
    for (std::size_t i = 0; i < E.data.size(); ++i)
        E.data[i] = static_cast<float>(i % 7) * 0.1f;

    lm.tie_weights(E, dE);
    CHECK(lm.is_tied());

    auto H = make_H(3, d);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);

    CHECK(probs.rows == 3 && probs.cols == vocab);
    for (std::size_t i = 0; i < 3; ++i)
        CHECK_NEAR(row_sum(probs, i), 1.f, 1e-4f);
}

void test_weight_tying_backward_updates_dE() {
    SECTION("Tied backward accumulates into dE not dW");
    std::size_t d = 8, vocab = 20;
    LMHead lm(d, vocab, 42);
    lm.zero_grad();

    Matrix2D E(vocab, d);
    Matrix2D dE(vocab, d, 0.f);
    for (std::size_t i = 0; i < E.data.size(); ++i)
        E.data[i] = static_cast<float>(i % 5) * 0.2f;

    lm.tie_weights(E, dE);

    auto H = make_H(3, d);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    std::vector<TokenID> targets = {3, 7, 1};
    auto result = LMHead::cross_entropy(probs, targets);
    (void)lm.backward(result.dLogits, cache);

    // dE should be updated, dW should stay zero
    CHECK(mat_any_nonzero(dE));
    CHECK(!mat_any_nonzero(lm.dW()));
}

void test_numerical_gradient_loss() {
    SECTION("Numerical gradient check: dLogits vs finite diff on loss");
    const float eps = 1e-3f;
    LMHead lm(4, 8, 42);

    auto H = make_H(2, 4);
    LMHeadCache cache;
    auto probs = lm.forward(H, cache);
    std::vector<TokenID> targets = {3, 6};
    auto result = LMHead::cross_entropy(probs, targets);
    (void)result.dLogits.at(0, 0);

    // Perturb logits[0,0] via W[0,0] tweak
    // (easier: perturb H[0,0] and measure loss change)
    float h_orig = H.at(0, 0);

    H.at(0, 0) = h_orig + eps;
    LMHeadCache c1; auto p1 = lm.forward(H, c1);
    auto r1 = LMHead::cross_entropy(p1, targets);

    H.at(0, 0) = h_orig - eps;
    LMHeadCache c2; auto p2 = lm.forward(H, c2);
    auto r2 = LMHead::cross_entropy(p2, targets);

    H.at(0, 0) = h_orig;
    float numerical_dH = (r1.loss - r2.loss) / (2.f * eps);

    // Get analytical dH[0,0]
    auto dH = lm.backward(result.dLogits, cache);
    CHECK_NEAR(dH.at(0, 0), numerical_dH, 5e-2f);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — LMHead tests\n";
    std::cout << "======================\n";

    test_construction();
    test_invalid_construction();
    test_forward_output_shape();
    test_probs_sum_to_one();
    test_probs_nonnegative();
    test_cache_shapes();
    test_cross_entropy_perfect_prediction();
    test_cross_entropy_uniform_prediction();
    test_cross_entropy_skips_padding();
    test_cross_entropy_dlogits_shape();
    test_cross_entropy_dlogits_sum_zero();
    test_cross_entropy_dlogits_pad_is_zero();
    test_backward_dH_shape();
    test_backward_accumulates_dW();
    test_sgd_step_updates_W();
    test_greedy_returns_argmax();
    test_top_k();
    test_weight_tying();
    test_weight_tying_backward_updates_dE();
    test_numerical_gradient_loss();

    std::cout << "\n======================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
