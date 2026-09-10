// layernorm_test.cpp, Unit tests for LayerNorm
//
// Build (from miniGPT/):
//   make layernorm_test && ./layernorm_test

#include "../norm/LayerNorm.hpp"
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

float row_mean(const Matrix2D& M, std::size_t r) {
    float s = 0.f;
    for (std::size_t c = 0; c < M.cols; ++c) s += M.at(r, c);
    return s / static_cast<float>(M.cols);
}

float row_var(const Matrix2D& M, std::size_t r) {
    float mu = row_mean(M, r);
    float v  = 0.f;
    for (std::size_t c = 0; c < M.cols; ++c) {
        float d = M.at(r, c) - mu;
        v += d * d;
    }
    return v / static_cast<float>(M.cols);
}

bool mat_any_nonzero(const Matrix2D& M) {
    for (float v : M.data) if (v != 0.f) return true;
    return false;
}

// ── Tests ─────────────────────────────────────────────────

void test_construction() {
    SECTION("Construction and initial parameters");
    LayerNorm ln(128);
    CHECK(ln.d_model() == 128);

    // γ initialised to 1
    for (std::size_t d = 0; d < 128; ++d)
        CHECK(ln.gamma().at(0, d) == 1.f);

    // β initialised to 0
    for (std::size_t d = 0; d < 128; ++d)
        CHECK(ln.beta().at(0, d) == 0.f);
}

void test_invalid_construction() {
    SECTION("Zero d_model throws");
    bool threw = false;
    try { LayerNorm ln(0); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_output_shape() {
    SECTION("forward() output shape [seq_len × d_model]");
    LayerNorm ln(16);
    Matrix2D X(4, 16);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i) * 0.1f;
    LayerNormCache cache;
    auto out = ln.forward(X, cache);
    CHECK(out.rows == 4 && out.cols == 16);
}

void test_output_zero_mean() {
    SECTION("Output has zero mean per position (with γ=1, β=0)");
    LayerNorm ln(32);
    Matrix2D X(5, 32);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i % 17) * 0.3f - 2.f;

    LayerNormCache cache;
    auto out = ln.forward(X, cache);

    for (std::size_t i = 0; i < 5; ++i)
        CHECK_NEAR(row_mean(out, i), 0.f, 1e-4f);
}

void test_output_unit_variance() {
    SECTION("Output has unit variance per position (with γ=1, β=0)");
    LayerNorm ln(32);
    Matrix2D X(5, 32);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i % 11) - 5.f;

    LayerNormCache cache;
    auto out = ln.forward(X, cache);

    for (std::size_t i = 0; i < 5; ++i)
        CHECK_NEAR(row_var(out, i), 1.f, 1e-3f);
}

void test_gamma_scales_output() {
    SECTION("γ=2 doubles the output scale");
    LayerNorm ln(8);
    for (std::size_t d = 0; d < 8; ++d)
        ln.gamma().at(0, d) = 2.f;

    Matrix2D X(2, 8);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i) * 0.5f;

    LayerNormCache cache1, cache2;

    // Compare γ=1 vs γ=2
    LayerNorm ln1(8);
    auto out1 = ln1.forward(X, cache1);
    auto out2 = ln.forward(X, cache2);

    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t d = 0; d < 8; ++d)
            CHECK_NEAR(out2.at(i, d), 2.f * out1.at(i, d), 1e-5f);
}

void test_beta_shifts_output() {
    SECTION("β=1 shifts all outputs by 1");
    LayerNorm ln(8);
    for (std::size_t d = 0; d < 8; ++d)
        ln.beta().at(0, d) = 1.f;

    Matrix2D X(2, 8);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i) * 0.5f;

    LayerNormCache cache1, cache2;
    LayerNorm ln1(8);
    auto out1 = ln1.forward(X, cache1);
    auto out2 = ln.forward(X, cache2);

    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t d = 0; d < 8; ++d)
            CHECK_NEAR(out2.at(i, d), out1.at(i, d) + 1.f, 1e-5f);
}

void test_constant_input_no_nan() {
    SECTION("Constant input row (zero variance) does not produce NaN");
    LayerNorm ln(8);
    Matrix2D X(2, 8, 5.f);   // all same value
    LayerNormCache cache;
    auto out = ln.forward(X, cache);

    for (float v : out.data)
        CHECK(!std::isnan(v));

    // Output should be ~0 since x - mean = 0 for all elements
    for (float v : out.data)
        CHECK_NEAR(v, 0.f, 1e-4f);
}

void test_cache_shapes() {
    SECTION("Cache stores x_hat and sigma with correct shapes");
    LayerNorm ln(16);
    Matrix2D X(3, 16);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i) * 0.1f;

    LayerNormCache cache;
    (void)ln.forward(X, cache);

    CHECK(cache.x_hat.rows == 3 && cache.x_hat.cols == 16);
    CHECK(cache.sigma.rows == 3 && cache.sigma.cols == 1);

    // sigma values should be positive
    for (std::size_t i = 0; i < 3; ++i)
        CHECK(cache.sigma.at(i, 0) > 0.f);
}

void test_x_hat_zero_mean() {
    SECTION("Cached x_hat has zero mean per position");
    LayerNorm ln(16);
    Matrix2D X(3, 16);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>((i * 7 + 3) % 19) - 9.f;

    LayerNormCache cache;
    (void)ln.forward(X, cache);

    for (std::size_t i = 0; i < 3; ++i)
        CHECK_NEAR(row_mean(cache.x_hat, i), 0.f, 1e-4f);
}

void test_backward_dX_shape() {
    SECTION("backward() dX shape matches input");
    LayerNorm ln(16);
    Matrix2D X(4, 16);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i) * 0.1f - 3.f;

    LayerNormCache cache;
    (void)ln.forward(X, cache);

    Matrix2D dOut(4, 16, 1.f);
    auto dX = ln.backward(dOut, cache);
    CHECK(dX.rows == 4 && dX.cols == 16);
}

void test_backward_dgamma_dbeta_nonzero() {
    SECTION("backward() accumulates dgamma and dbeta");
    LayerNorm ln(8);
    ln.zero_grad();

    Matrix2D X(3, 8);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i) * 0.3f - 1.f;

    LayerNormCache cache;
    (void)ln.forward(X, cache);

    Matrix2D dOut(3, 8, 1.f);
    (void)ln.backward(dOut, cache);

    CHECK(mat_any_nonzero(ln.dgamma()));
    CHECK(mat_any_nonzero(ln.dbeta()));
}

void test_backward_dbeta_equals_sum_dout() {
    SECTION("dβ = sum of dOut rows (chain rule through β add)");
    LayerNorm ln(4);
    ln.zero_grad();

    Matrix2D X(3, 4);
    X.data = {1.f, 2.f, 3.f, 4.f,
              5.f, 6.f, 7.f, 8.f,
              9.f,10.f,11.f,12.f};

    LayerNormCache cache;
    (void)ln.forward(X, cache);

    Matrix2D dOut(3, 4, 1.f);
    (void)ln.backward(dOut, cache);

    // dβ[d] = sum_i dOut[i,d] = 3 (three positions, each contributing 1)
    for (std::size_t d = 0; d < 4; ++d)
        CHECK_NEAR(ln.dbeta().at(0, d), 3.f, 1e-5f);
}

void test_backward_dX_zero_mean() {
    SECTION("dX rows have zero mean (LayerNorm backward property)");
    // The LayerNorm backward ensures sum(dX[i,:]) = 0 for each i
    // because normalisation removes the mean direction
    LayerNorm ln(16);
    ln.zero_grad();

    Matrix2D X(4, 16);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>((i * 3 + 1) % 13) * 0.5f - 3.f;

    LayerNormCache cache;
    (void)ln.forward(X, cache);

    Matrix2D dOut(4, 16);
    for (std::size_t i = 0; i < dOut.data.size(); ++i)
        dOut.data[i] = static_cast<float>(i % 7) * 0.1f;

    auto dX = ln.backward(dOut, cache);

    for (std::size_t i = 0; i < 4; ++i)
        CHECK_NEAR(row_mean(dX, i), 0.f, 1e-4f);
}

void test_zero_grad() {
    SECTION("zero_grad() clears dgamma and dbeta");
    LayerNorm ln(8);
    Matrix2D X(2, 8);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i);
    LayerNormCache cache;
    (void)ln.forward(X, cache);
    Matrix2D dOut(2, 8, 1.f);
    (void)ln.backward(dOut, cache);
    ln.zero_grad();

    CHECK(!mat_any_nonzero(ln.dgamma()));
    CHECK(!mat_any_nonzero(ln.dbeta()));
}

void test_sgd_step_updates_gamma_beta() {
    SECTION("sgd_step() updates γ and β");
    LayerNorm ln(8);
    ln.zero_grad();

    Matrix2D X(2, 8);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>(i) * 0.5f - 2.f;

    LayerNormCache cache;
    (void)ln.forward(X, cache);
    Matrix2D dOut(2, 8, 1.f);
    (void)ln.backward(dOut, cache);

    float gamma_before = ln.gamma().at(0, 0);
    float beta_before  = ln.beta().at(0, 0);
    ln.sgd_step(0.1f);

    CHECK(std::abs(ln.gamma().at(0, 0) - gamma_before) > 1e-7f);
    CHECK(std::abs(ln.beta().at(0, 0)  - beta_before)  > 1e-7f);
}

void test_numerical_gradient_gamma() {
    SECTION("Numerical gradient check for γ");
    const float eps = 1e-3f;
    LayerNorm ln(4);
    ln.zero_grad();

    Matrix2D X(2, 4);
    X.data = {1.f, -2.f, 3.f, -1.f,
              2.f,  1.f,-1.f,  0.5f};

    LayerNormCache cache;
    (void)ln.forward(X, cache);
    Matrix2D dOut(2, 4, 1.f);
    (void)ln.backward(dOut, cache);
    float analytical = ln.dgamma().at(0, 0);

    float g_orig = ln.gamma().at(0, 0);

    ln.gamma().at(0, 0) = g_orig + eps;
    LayerNormCache c1; auto out1 = ln.forward(X, c1);
    float loss1 = 0.f; for (float v : out1.data) loss1 += v;

    ln.gamma().at(0, 0) = g_orig - eps;
    LayerNormCache c2; auto out2 = ln.forward(X, c2);
    float loss2 = 0.f; for (float v : out2.data) loss2 += v;

    ln.gamma().at(0, 0) = g_orig;
    float numerical = (loss1 - loss2) / (2.f * eps);
    CHECK_NEAR(analytical, numerical, 1e-2f);
}

void test_numerical_gradient_input() {
    SECTION("Numerical gradient check for input X");
    const float eps = 1e-3f;
    LayerNorm ln(4);
    ln.zero_grad();

    Matrix2D X(2, 4);
    X.data = {1.f, -2.f, 3.f, -1.f,
              2.f,  1.f,-1.f,  0.5f};

    LayerNormCache cache;
    (void)ln.forward(X, cache);
    Matrix2D dOut(2, 4, 1.f);
    auto dX = ln.backward(dOut, cache);
    float analytical = dX.at(0, 0);

    float x_orig = X.at(0, 0);

    X.at(0, 0) = x_orig + eps;
    LayerNormCache c1; auto out1 = ln.forward(X, c1);
    float loss1 = 0.f; for (float v : out1.data) loss1 += v;

    X.at(0, 0) = x_orig - eps;
    LayerNormCache c2; auto out2 = ln.forward(X, c2);
    float loss2 = 0.f; for (float v : out2.data) loss2 += v;

    X.at(0, 0) = x_orig;
    float numerical = (loss1 - loss2) / (2.f * eps);
    CHECK_NEAR(analytical, numerical, 1e-2f);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — LayerNorm tests\n";
    std::cout << "=========================\n";

    test_construction();
    test_invalid_construction();
    test_output_shape();
    test_output_zero_mean();
    test_output_unit_variance();
    test_gamma_scales_output();
    test_beta_shifts_output();
    test_constant_input_no_nan();
    test_cache_shapes();
    test_x_hat_zero_mean();
    test_backward_dX_shape();
    test_backward_dgamma_dbeta_nonzero();
    test_backward_dbeta_equals_sum_dout();
    test_backward_dX_zero_mean();
    test_zero_grad();
    test_sgd_step_updates_gamma_beta();
    test_numerical_gradient_gamma();
    test_numerical_gradient_input();

    std::cout << "\n=========================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
