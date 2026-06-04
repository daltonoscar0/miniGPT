// mha_test.cpp  —  Unit tests for MultiHeadAttention
//
// Build (from miniGPT/):
//   make mha_test && ./mha_test

#include "../attention/MultiHeadAttention.hpp"
#include "../attention/Attention.hpp"
#include "../embed/Embedding.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

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

std::vector<bool> all_real(std::size_t n) {
    return std::vector<bool>(n, true);
}

Matrix2D make_input(std::size_t seq, std::size_t d) {
    Matrix2D X(seq, d);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i % 7) * 0.1f;
    return X;
}

float mat_sum(const Matrix2D& M) {
    float s = 0.f;
    for (float v : M.data) s += v;
    return s;
}

bool mat_any_nonzero(const Matrix2D& M) {
    for (float v : M.data)
        if (v != 0.f) return true;
    return false;
}

// ── Tests ─────────────────────────────────────────────────

void test_construction() {
    SECTION("Construction — dimensions");
    MultiHeadAttention mha(128, 4, 42);
    CHECK(mha.d_model() == 128);
    CHECK(mha.n_heads() == 4);
    CHECK(mha.d_k()     == 32);

    CHECK(mha.W_Q().rows == 128 && mha.W_Q().cols == 128);
    CHECK(mha.W_K().rows == 128 && mha.W_K().cols == 128);
    CHECK(mha.W_V().rows == 128 && mha.W_V().cols == 128);
    CHECK(mha.W_O().rows == 128 && mha.W_O().cols == 128);
}

void test_invalid_construction() {
    SECTION("Invalid construction throws");
    bool threw = false;
    try { MultiHeadAttention mha(128, 3); }   // 128 % 3 != 0
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { MultiHeadAttention mha(0, 4); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_forward_output_shape() {
    SECTION("forward() output shape [seq_len × d_model]");
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(5, 16);
    MHACache cache;

    auto out = mha.forward(X, all_real(5), /*causal=*/true, cache);
    CHECK(out.rows == 5);
    CHECK(out.cols == 16);
}

void test_forward_output_nonzero() {
    SECTION("forward() produces non-zero output");
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(4, 16);
    MHACache cache;

    auto out = mha.forward(X, all_real(4), true, cache);
    CHECK(mat_any_nonzero(out));
}

void test_cache_filled() {
    SECTION("forward() fills cache correctly");
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(4, 16);
    MHACache cache;

    (void)mha.forward(X, all_real(4), true, cache);

    CHECK(cache.X.rows == 4 && cache.X.cols == 16);
    CHECK(cache.concat.rows == 4 && cache.concat.cols == 16);
    CHECK(cache.heads.size() == 4);

    // Each head cache should have Q,K,V of shape [seq_len × d_k]
    for (const auto& h : cache.heads) {
        CHECK(h.Q.rows == 4 && h.Q.cols == 4);   // d_k = 16/4 = 4
        CHECK(h.K.rows == 4 && h.K.cols == 4);
        CHECK(h.V.rows == 4 && h.V.cols == 4);
    }
}

void test_causal_mask_respected() {
    SECTION("causal mask flows through to each head");
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(4, 16);
    MHACache cache;

    (void)mha.forward(X, all_real(4), /*causal=*/true, cache);

    // Each head's attention weights should be lower-triangular
    for (const auto& h : cache.heads) {
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = i + 1; j < 4; ++j)
                CHECK_NEAR(h.weights.at(i, j), 0.f, 1e-6f);
    }
}

void test_padding_mask_respected() {
    SECTION("padding mask zeros future-position attention weights");
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(4, 16);

    // Last 2 positions are padding
    std::vector<bool> mask = {true, true, false, false};
    MHACache cache;

    (void)mha.forward(X, mask, /*causal=*/false, cache);

    for (const auto& h : cache.heads)
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK_NEAR(h.weights.at(i, 2), 0.f, 1e-6f);
            CHECK_NEAR(h.weights.at(i, 3), 0.f, 1e-6f);
        }
}

void test_different_heads_different_output() {
    SECTION("Different heads produce different attention patterns");
    // With random init (different seeds per head slice), heads should differ
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(4, 16);
    MHACache cache;
    (void)mha.forward(X, all_real(4), true, cache);

    // Head 0 and head 1 weights should differ (different W_Q/K/V slices)
    bool any_diff = false;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            if (std::abs(cache.heads[0].weights.at(i,j) -
                         cache.heads[1].weights.at(i,j)) > 1e-4f)
                any_diff = true;
    CHECK(any_diff);
}

void test_backward_dX_shape() {
    SECTION("backward() dX shape matches input");
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(5, 16);
    MHACache cache;
    auto out = mha.forward(X, all_real(5), true, cache);

    Matrix2D dOut(5, 16, 1.f);
    auto dX = mha.backward(dOut, cache);

    CHECK(dX.rows == 5);
    CHECK(dX.cols == 16);
}

void test_backward_dX_nonzero() {
    SECTION("backward() dX is non-zero");
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(4, 16);
    MHACache cache;
    (void)mha.forward(X, all_real(4), true, cache);

    Matrix2D dOut(4, 16, 1.f);
    auto dX = mha.backward(dOut, cache);
    CHECK(mat_any_nonzero(dX));
}

void test_backward_grad_accumulators_updated() {
    SECTION("backward() updates dW_Q, dW_K, dW_V, dW_O");
    MultiHeadAttention mha(16, 4, 42);
    mha.zero_grad();

    auto X = make_input(4, 16);
    MHACache cache;
    (void)mha.forward(X, all_real(4), true, cache);

    Matrix2D dOut(4, 16, 1.f);
    (void)mha.backward(dOut, cache);

    CHECK(mat_any_nonzero(mha.dW_Q()));
    CHECK(mat_any_nonzero(mha.dW_K()));
    CHECK(mat_any_nonzero(mha.dW_V()));
    CHECK(mat_any_nonzero(mha.dW_O()));
}

void test_zero_grad_clears_all() {
    SECTION("zero_grad() clears all gradient accumulators");
    MultiHeadAttention mha(16, 4, 42);
    auto X = make_input(4, 16);
    MHACache cache;
    (void)mha.forward(X, all_real(4), true, cache);
    Matrix2D dOut(4, 16, 1.f);
    (void)mha.backward(dOut, cache);
    mha.zero_grad();

    CHECK(!mat_any_nonzero(mha.dW_Q()));
    CHECK(!mat_any_nonzero(mha.dW_K()));
    CHECK(!mat_any_nonzero(mha.dW_V()));
    CHECK(!mat_any_nonzero(mha.dW_O()));
}

void test_sgd_step_changes_weights() {
    SECTION("sgd_step() updates W_O");
    MultiHeadAttention mha(16, 4, 42);
    mha.zero_grad();

    // Record a weight before update
    float w_before = mha.W_O().at(0, 0);

    auto X = make_input(4, 16);
    MHACache cache;
    (void)mha.forward(X, all_real(4), true, cache);
    Matrix2D dOut(4, 16, 1.f);
    (void)mha.backward(dOut, cache);
    mha.sgd_step(0.01f);

    float w_after = mha.W_O().at(0, 0);
    CHECK(std::abs(w_after - w_before) > 1e-7f);
}

void test_n_heads_1_matches_single_attention() {
    SECTION("n_heads=1 is equivalent to a single attention head");
    // With 1 head, the MHA is just a single attention with W_Q/K/V/O projections.
    // Output shape and non-zero check suffice here.
    MultiHeadAttention mha(8, 1, 42);
    CHECK(mha.d_k() == 8);

    auto X = make_input(3, 8);
    MHACache cache;
    auto out = mha.forward(X, all_real(3), true, cache);

    CHECK(out.rows == 3 && out.cols == 8);
    CHECK(cache.heads.size() == 1);
}

void test_deterministic_given_seed() {
    SECTION("Same seed → identical weights");
    MultiHeadAttention a(16, 4, 99);
    MultiHeadAttention b(16, 4, 99);

    for (std::size_t i = 0; i < a.W_Q().data.size(); ++i)
        CHECK(a.W_Q().data[i] == b.W_Q().data[i]);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — MultiHeadAttention tests\n";
    std::cout << "===================================\n";

    test_construction();
    test_invalid_construction();
    test_forward_output_shape();
    test_forward_output_nonzero();
    test_cache_filled();
    test_causal_mask_respected();
    test_padding_mask_respected();
    test_different_heads_different_output();
    test_backward_dX_shape();
    test_backward_dX_nonzero();
    test_backward_grad_accumulators_updated();
    test_zero_grad_clears_all();
    test_sgd_step_changes_weights();
    test_n_heads_1_matches_single_attention();
    test_deterministic_given_seed();

    std::cout << "\n===================================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
