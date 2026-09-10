// ffn_test.cpp, Unit tests for FeedForward sublayer
//
// Build (from miniGPT/):
//   make ffn_test && ./ffn_test

#include "../ffn/FeedForward.hpp"
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

Matrix2D make_input(std::size_t seq, std::size_t d, float scale = 0.1f) {
    Matrix2D X(seq, d);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i % 13) * scale - 0.5f;
    return X;
}

bool mat_any_nonzero(const Matrix2D& M) {
    for (float v : M.data) if (v != 0.f) return true;
    return false;
}

float mat_max_abs(const Matrix2D& M) {
    float m = 0.f;
    for (float v : M.data) m = std::max(m, std::abs(v));
    return m;
}

// ── Tests ─────────────────────────────────────────────────

void test_construction_default_d_ff() {
    SECTION("Default d_ff = 4 * d_model");
    FeedForward ffn(128);
    CHECK(ffn.d_model() == 128);
    CHECK(ffn.d_ff()    == 512);
    CHECK(ffn.W1().rows == 128 && ffn.W1().cols == 512);
    CHECK(ffn.W2().rows == 512 && ffn.W2().cols == 128);
}

void test_construction_custom_d_ff() {
    SECTION("Custom d_ff");
    FeedForward ffn(16, 64);
    CHECK(ffn.d_ff() == 64);
    CHECK(ffn.W1().cols == 64);
    CHECK(ffn.W2().rows == 64);
}

void test_invalid_construction() {
    SECTION("Zero d_model throws");
    bool threw = false;
    try { FeedForward ffn(0); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_forward_output_shape() {
    SECTION("forward() output shape [seq_len × d_model]");
    FeedForward ffn(16, 64, 42);
    auto X = make_input(5, 16);
    FFNCache cache;
    auto out = ffn.forward(X, cache);
    CHECK(out.rows == 5);
    CHECK(out.cols == 16);
}

void test_forward_output_nonzero() {
    SECTION("forward() produces non-zero output");
    FeedForward ffn(16, 64, 42);
    auto X = make_input(4, 16);
    FFNCache cache;
    auto out = ffn.forward(X, cache);
    CHECK(mat_any_nonzero(out));
}

void test_relu_mask_binary() {
    SECTION("ReLU mask is binary (0 or 1)");
    FeedForward ffn(8, 32, 42);
    auto X = make_input(3, 8);
    FFNCache cache;
    (void)ffn.forward(X, cache);

    for (float v : cache.relu_mask.data) {
        CHECK(v == 0.f || v == 1.f);
    }
}

void test_relu_mask_matches_hidden() {
    SECTION("ReLU mask is 1 exactly where H > 0");
    FeedForward ffn(8, 32, 42);
    auto X = make_input(4, 8);
    FFNCache cache;
    (void)ffn.forward(X, cache);

    for (std::size_t i = 0; i < cache.H.rows; ++i)
        for (std::size_t d = 0; d < cache.H.cols; ++d) {
            float h = cache.H.at(i, d);
            float m = cache.relu_mask.at(i, d);
            if (h > 0.f) CHECK(m == 1.f);
            else         CHECK(m == 0.f);
        }
}

void test_hidden_nonnegative() {
    SECTION("Post-ReLU hidden H has no negative values");
    FeedForward ffn(8, 32, 7);
    auto X = make_input(5, 8);
    FFNCache cache;
    (void)ffn.forward(X, cache);
    for (float v : cache.H.data)
        CHECK(v >= 0.f);
}

void test_forward_all_negative_input() {
    SECTION("All-negative pre-activation → all-zero H");
    // Force W1 and b1 such that all pre-activations are negative
    FeedForward ffn(4, 8, 42);
    // Set W1 and b1 to large negatives
    for (float& v : ffn.W1().data) v = -10.f;
    // Input all positive -> W1 negative -> H_pre all negative -> H=0
    Matrix2D X(2, 4, 1.f);
    FFNCache cache;
    (void)ffn.forward(X, cache);
    for (float v : cache.H.data)
        CHECK(v == 0.f);
}

void test_cache_shapes() {
    SECTION("Cache stores X and H with correct shapes");
    FeedForward ffn(16, 64, 42);
    auto X = make_input(4, 16);
    FFNCache cache;
    (void)ffn.forward(X, cache);

    CHECK(cache.X.rows == 4 && cache.X.cols == 16);
    CHECK(cache.H.rows == 4 && cache.H.cols == 64);
    CHECK(cache.relu_mask.rows == 4 && cache.relu_mask.cols == 64);
}

void test_backward_dX_shape() {
    SECTION("backward() dX shape matches input");
    FeedForward ffn(16, 64, 42);
    auto X = make_input(4, 16);
    FFNCache cache;
    (void)ffn.forward(X, cache);

    Matrix2D dOut(4, 16, 1.f);
    auto dX = ffn.backward(dOut, cache);
    CHECK(dX.rows == 4 && dX.cols == 16);
}

void test_backward_dX_nonzero() {
    SECTION("backward() dX is non-zero");
    FeedForward ffn(16, 64, 42);
    auto X = make_input(4, 16);
    FFNCache cache;
    (void)ffn.forward(X, cache);

    Matrix2D dOut(4, 16, 1.f);
    auto dX = ffn.backward(dOut, cache);
    CHECK(mat_any_nonzero(dX));
}

void test_backward_grad_accumulators() {
    SECTION("backward() updates dW1, dW2, db1, db2");
    FeedForward ffn(8, 32, 42);
    ffn.zero_grad();

    auto X = make_input(3, 8);
    FFNCache cache;
    (void)ffn.forward(X, cache);

    Matrix2D dOut(3, 8, 1.f);
    (void)ffn.backward(dOut, cache);

    CHECK(mat_any_nonzero(ffn.dW1()));
    CHECK(mat_any_nonzero(ffn.dW2()));
    // db1/db2 are nonzero as long as some ReLU units fired
    // (they will with random init)
    CHECK(mat_any_nonzero(ffn.db2()));
}

void test_backward_zero_when_relu_dead() {
    SECTION("dW1 is zero when all ReLU units are dead");
    FeedForward ffn(4, 8, 42);
    ffn.zero_grad();

    // Kill all ReLU units by setting W1 to large negatives
    for (float& v : ffn.W1().data) v = -100.f;

    Matrix2D X(2, 4, 1.f);
    FFNCache cache;
    (void)ffn.forward(X, cache);

    // All H == 0 -> relu_mask == 0 -> dH_pre == 0 -> dW1 == 0
    Matrix2D dOut(2, 4, 1.f);
    (void)ffn.backward(dOut, cache);

    for (float v : ffn.dW1().data)
        CHECK(v == 0.f);
}

void test_zero_grad() {
    SECTION("zero_grad() clears all accumulators");
    FeedForward ffn(8, 32, 42);
    auto X = make_input(3, 8);
    FFNCache cache;
    (void)ffn.forward(X, cache);
    Matrix2D dOut(3, 8, 1.f);
    (void)ffn.backward(dOut, cache);
    ffn.zero_grad();

    CHECK(!mat_any_nonzero(ffn.dW1()));
    CHECK(!mat_any_nonzero(ffn.dW2()));
    CHECK(!mat_any_nonzero(ffn.db1()));
    CHECK(!mat_any_nonzero(ffn.db2()));
}

void test_sgd_step_updates_weights() {
    SECTION("sgd_step() moves W1 and W2");
    FeedForward ffn(8, 32, 42);
    ffn.zero_grad();

    (void)ffn.W1().at(0, 0);
    float w2_before = ffn.W2().at(0, 0);

    auto X = make_input(3, 8);
    FFNCache cache;
    (void)ffn.forward(X, cache);
    Matrix2D dOut(3, 8, 1.f);
    (void)ffn.backward(dOut, cache);
    ffn.sgd_step(0.01f);

    // At least one weight should have changed
    CHECK(std::abs(ffn.W2().at(0, 0) - w2_before) > 1e-7f);
    // Verify W1 gradient is non-zero (ReLU fired with this input)
    CHECK(mat_any_nonzero(ffn.dW1()));
}

void test_numerical_gradient_W2() {
    SECTION("Numerical gradient check for W2");
    // Finite difference: df/dw ≈ (f(w+ε) - f(w-ε)) / (2ε)
    const float eps = 1e-3f;
    FeedForward ffn(4, 8, 42);
    ffn.zero_grad();

    Matrix2D X(2, 4);
    X.data = {0.1f, -0.2f, 0.3f, -0.4f,
              0.5f,  0.6f,-0.7f,  0.8f};

    // Forward + backward to get analytical gradient
    FFNCache cache;
    auto out = ffn.forward(X, cache);
    Matrix2D dOut(2, 4, 1.f);   // loss = sum(out)
    (void)ffn.backward(dOut, cache);
    float analytical = ffn.dW2().at(0, 0);

    // Numerical gradient for W2[0,0]
    float w_orig = ffn.W2().at(0, 0);

    ffn.W2().at(0, 0) = w_orig + eps;
    FFNCache c1; auto out1 = ffn.forward(X, c1);
    float loss1 = 0.f; for (float v : out1.data) loss1 += v;

    ffn.W2().at(0, 0) = w_orig - eps;
    FFNCache c2; auto out2 = ffn.forward(X, c2);
    float loss2 = 0.f; for (float v : out2.data) loss2 += v;

    ffn.W2().at(0, 0) = w_orig;  // restore

    float numerical = (loss1 - loss2) / (2.f * eps);
    CHECK_NEAR(analytical, numerical, 1e-2f);
}

void test_numerical_gradient_W1() {
    SECTION("Numerical gradient check for W1");
    const float eps = 1e-3f;
    FeedForward ffn(4, 8, 42);
    ffn.zero_grad();

    Matrix2D X(2, 4);
    X.data = {0.1f, -0.2f, 0.3f, -0.4f,
              0.5f,  0.6f,-0.7f,  0.8f};

    FFNCache cache;
    (void)ffn.forward(X, cache);
    Matrix2D dOut(2, 4, 1.f);
    (void)ffn.backward(dOut, cache);
    float analytical = ffn.dW1().at(0, 0);

    float w_orig = ffn.W1().at(0, 0);

    ffn.W1().at(0, 0) = w_orig + eps;
    FFNCache c1; auto out1 = ffn.forward(X, c1);
    float loss1 = 0.f; for (float v : out1.data) loss1 += v;

    ffn.W1().at(0, 0) = w_orig - eps;
    FFNCache c2; auto out2 = ffn.forward(X, c2);
    float loss2 = 0.f; for (float v : out2.data) loss2 += v;

    ffn.W1().at(0, 0) = w_orig;

    float numerical = (loss1 - loss2) / (2.f * eps);
    CHECK_NEAR(analytical, numerical, 1e-2f);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — FeedForward tests\n";
    std::cout << "===========================\n";

    test_construction_default_d_ff();
    test_construction_custom_d_ff();
    test_invalid_construction();
    test_forward_output_shape();
    test_forward_output_nonzero();
    test_relu_mask_binary();
    test_relu_mask_matches_hidden();
    test_hidden_nonnegative();
    test_forward_all_negative_input();
    test_cache_shapes();
    test_backward_dX_shape();
    test_backward_dX_nonzero();
    test_backward_grad_accumulators();
    test_backward_zero_when_relu_dead();
    test_zero_grad();
    test_sgd_step_updates_weights();
    test_numerical_gradient_W2();
    test_numerical_gradient_W1();

    std::cout << "\n===========================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
