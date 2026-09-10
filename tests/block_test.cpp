// block_test.cpp, Unit tests for TransformerBlock
//
// Build (from miniGPT/):
//   make block_test && ./block_test

#include "../block/TransformerBlock.hpp"

#include <iostream>
#include <vector>
#include <cmath>
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

std::vector<bool> all_real(std::size_t n) {
    return std::vector<bool>(n, true);
}

Matrix2D make_input(std::size_t seq, std::size_t d) {
    Matrix2D X(seq, d);
    for (std::size_t i = 0; i < X.data.size(); ++i)
        X.data[i] = static_cast<float>((i * 7 + 3) % 19) * 0.1f - 0.9f;
    return X;
}

bool mat_any_nonzero(const Matrix2D& M) {
    for (float v : M.data) if (v != 0.f) return true;
    return false;
}

bool mat_any_nan(const Matrix2D& M) {
    for (float v : M.data) if (std::isnan(v)) return true;
    return false;
}

// ── Tests ─────────────────────────────────────────────────

void test_construction() {
    SECTION("Construction");
    TransformerBlock blk(128, 4);
    CHECK(blk.d_model() == 128);
    CHECK(blk.mha().n_heads() == 4);
    CHECK(blk.ffn().d_ff() == 512);   // 4 * 128
    CHECK(blk.ln1().d_model() == 128);
    CHECK(blk.ln2().d_model() == 128);
}

void test_forward_output_shape() {
    SECTION("forward() output shape [seq_len × d_model]");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(5, 16);
    BlockCache cache;
    auto out = blk.forward(X, all_real(5), true, cache);
    CHECK(out.rows == 5);
    CHECK(out.cols == 16);
}

void test_forward_no_nan() {
    SECTION("forward() produces no NaN values");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    BlockCache cache;
    auto out = blk.forward(X, all_real(4), true, cache);
    CHECK(!mat_any_nan(out));
}

void test_forward_nonzero_output() {
    SECTION("forward() output is non-zero");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    BlockCache cache;
    auto out = blk.forward(X, all_real(4), true, cache);
    CHECK(mat_any_nonzero(out));
}

void test_residual_connection() {
    SECTION("Residual: output differs from pure MHA+FFN (identity path exists)");
    // With residual, output = input + transform(input)
    // So output should differ from both pure input and pure transform
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(3, 16);
    BlockCache cache;
    auto out = blk.forward(X, all_real(3), true, cache);

    // Output should not equal input exactly (transform is non-zero)
    bool differs_from_input = false;
    for (std::size_t i = 0; i < X.data.size(); ++i)
        if (std::abs(out.data[i] - X.data[i]) > 1e-6f)
            differs_from_input = true;
    CHECK(differs_from_input);
}

void test_cache_filled() {
    SECTION("Cache stores X and y1 with correct shapes");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    BlockCache cache;
    (void)blk.forward(X, all_real(4), true, cache);

    CHECK(cache.X.rows  == 4 && cache.X.cols  == 16);
    CHECK(cache.y1.rows == 4 && cache.y1.cols == 16);
}

void test_causal_mask_flows_through() {
    SECTION("Causal mask reaches attention heads");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    BlockCache cache;
    (void)blk.forward(X, all_real(4), /*causal=*/true, cache);

    // Each head's weights should be lower-triangular
    for (const auto& h : cache.mha_cache.heads)
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = i + 1; j < 4; ++j)
                CHECK_NEAR(h.weights.at(i, j), 0.f, 1e-6f);
}

void test_padding_mask_flows_through() {
    SECTION("Padding mask reaches attention heads");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    std::vector<bool> mask = {true, true, false, false};
    BlockCache cache;
    (void)blk.forward(X, mask, false, cache);

    for (const auto& h : cache.mha_cache.heads)
        for (std::size_t i = 0; i < 4; ++i) {
            CHECK_NEAR(h.weights.at(i, 2), 0.f, 1e-6f);
            CHECK_NEAR(h.weights.at(i, 3), 0.f, 1e-6f);
        }
}

void test_backward_dX_shape() {
    SECTION("backward() dX shape matches input");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    BlockCache cache;
    (void)blk.forward(X, all_real(4), true, cache);

    Matrix2D dY2(4, 16, 1.f);
    auto dX = blk.backward(dY2, cache);
    CHECK(dX.rows == 4 && dX.cols == 16);
}

void test_backward_dX_nonzero() {
    SECTION("backward() dX is non-zero");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    BlockCache cache;
    (void)blk.forward(X, all_real(4), true, cache);

    Matrix2D dY2(4, 16, 1.f);
    auto dX = blk.backward(dY2, cache);
    CHECK(mat_any_nonzero(dX));
}

void test_backward_no_nan() {
    SECTION("backward() produces no NaN values");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    BlockCache cache;
    (void)blk.forward(X, all_real(4), true, cache);

    Matrix2D dY2(4, 16, 1.f);
    auto dX = blk.backward(dY2, cache);
    CHECK(!mat_any_nan(dX));
}

void test_backward_updates_all_params() {
    SECTION("backward() accumulates gradients in all subcomponents");
    TransformerBlock blk(16, 4, 64, 42);
    blk.zero_grad();

    auto X = make_input(4, 16);
    BlockCache cache;
    (void)blk.forward(X, all_real(4), true, cache);

    Matrix2D dY2(4, 16, 1.f);
    (void)blk.backward(dY2, cache);

    CHECK(mat_any_nonzero(blk.mha().dW_Q()));
    CHECK(mat_any_nonzero(blk.mha().dW_O()));
    CHECK(mat_any_nonzero(blk.ffn().dW1()));
    CHECK(mat_any_nonzero(blk.ffn().dW2()));
    CHECK(mat_any_nonzero(blk.ln1().dgamma()));
    CHECK(mat_any_nonzero(blk.ln2().dgamma()));
}

void test_zero_grad_clears_everything() {
    SECTION("zero_grad() clears all subcomponent gradients");
    TransformerBlock blk(16, 4, 64, 42);
    auto X = make_input(4, 16);
    BlockCache cache;
    (void)blk.forward(X, all_real(4), true, cache);
    Matrix2D dY2(4, 16, 1.f);
    (void)blk.backward(dY2, cache);
    blk.zero_grad();

    CHECK(!mat_any_nonzero(blk.mha().dW_Q()));
    CHECK(!mat_any_nonzero(blk.ffn().dW1()));
    CHECK(!mat_any_nonzero(blk.ln1().dgamma()));
    CHECK(!mat_any_nonzero(blk.ln2().dgamma()));
}

void test_sgd_step_changes_weights() {
    SECTION("sgd_step() updates weights in all subcomponents");
    TransformerBlock blk(16, 4, 64, 42);
    blk.zero_grad();

    float w_before = blk.mha().W_O().at(0, 0);

    auto X = make_input(4, 16);
    BlockCache cache;
    (void)blk.forward(X, all_real(4), true, cache);
    Matrix2D dY2(4, 16, 1.f);
    (void)blk.backward(dY2, cache);
    blk.sgd_step(0.01f);

    CHECK(std::abs(blk.mha().W_O().at(0, 0) - w_before) > 1e-7f);
}

void test_two_blocks_stack() {
    SECTION("Two blocks stacked produce correct shapes");
    TransformerBlock blk1(16, 4, 64, 42);
    TransformerBlock blk2(16, 4, 64, 99);

    auto X = make_input(4, 16);
    BlockCache cache1, cache2;

    auto y1 = blk1.forward(X,  all_real(4), true, cache1);
    auto y2 = blk2.forward(y1, all_real(4), true, cache2);

    CHECK(y2.rows == 4 && y2.cols == 16);
    CHECK(!mat_any_nan(y2));

    // Backward through both blocks
    Matrix2D dY2(4, 16, 1.f);
    auto dY1 = blk2.backward(dY2,  cache2);
    auto dX  = blk1.backward(dY1,  cache1);

    CHECK(dX.rows == 4 && dX.cols == 16);
    CHECK(!mat_any_nan(dX));
    CHECK(mat_any_nonzero(dX));
}

void test_numerical_gradient() {
    SECTION("Numerical gradient check through full block");
    const float eps = 1e-3f;
    TransformerBlock blk(4, 2, 8, 42);
    blk.zero_grad();

    Matrix2D X(2, 4);
    X.data = {0.1f, -0.2f, 0.3f, -0.1f,
              0.4f,  0.1f,-0.2f,  0.3f};

    // Analytical
    BlockCache cache;
    (void)blk.forward(X, all_real(2), true, cache);
    Matrix2D dOut(2, 4, 1.f);
    auto dX = blk.backward(dOut, cache);
    float analytical = dX.at(0, 0);

    // Numerical
    float x_orig = X.at(0, 0);

    X.at(0, 0) = x_orig + eps;
    BlockCache c1; auto out1 = blk.forward(X, all_real(2), true, c1);
    float loss1 = 0.f; for (float v : out1.data) loss1 += v;

    X.at(0, 0) = x_orig - eps;
    BlockCache c2; auto out2 = blk.forward(X, all_real(2), true, c2);
    float loss2 = 0.f; for (float v : out2.data) loss2 += v;

    X.at(0, 0) = x_orig;
    float numerical = (loss1 - loss2) / (2.f * eps);
    CHECK_NEAR(analytical, numerical, 5e-2f);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — TransformerBlock tests\n";
    std::cout << "================================\n";

    test_construction();
    test_forward_output_shape();
    test_forward_no_nan();
    test_forward_nonzero_output();
    test_residual_connection();
    test_cache_filled();
    test_causal_mask_flows_through();
    test_padding_mask_flows_through();
    test_backward_dX_shape();
    test_backward_dX_nonzero();
    test_backward_no_nan();
    test_backward_updates_all_params();
    test_zero_grad_clears_everything();
    test_sgd_step_changes_weights();
    test_two_blocks_stack();
    test_numerical_gradient();

    std::cout << "\n================================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
