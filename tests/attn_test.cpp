// attn_test.cpp, Unit tests for scaled dot-product attention
//
// Build (from miniGPT/):
//   make attn_test && ./attn_test

#include "../attention/Attention.hpp"
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
        if (std::abs(_a-_b) > (eps)) {                                \
            std::cerr << "FAIL  " #a " ≈ " #b                        \
                      << "  got " << _a << " vs " << _b              \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";\
        } else { ++passed; }                                          \
    } while(0)

#define SECTION(name) std::cout << "\n── " name " ──\n"

using namespace miniGPT;

// ── Helpers ───────────────────────────────────────────────

// Fill matrix with a constant
Matrix2D fill(std::size_t r, std::size_t c, float v) {
    return Matrix2D(r, c, v);
}

// Fill matrix with sequential values 0,1,2,...
Matrix2D iota(std::size_t r, std::size_t c) {
    Matrix2D M(r, c);
    for (std::size_t i = 0; i < r * c; ++i)
        M.data[i] = static_cast<float>(i);
    return M;
}

// All-true mask
std::vector<bool> all_real(std::size_t n) {
    return std::vector<bool>(n, true);
}

// Row sum
float row_sum(const Matrix2D& M, std::size_t r) {
    float s = 0.f;
    for (std::size_t c = 0; c < M.cols; ++c) s += M.at(r, c);
    return s;
}

// ── Matrix operation tests ────────────────────────────────

void test_matmul() {
    SECTION("matmul basic");
    // [2×3] · [3×2] = [2×2]
    Matrix2D A(2, 3), B(3, 2);
    // A = [[1,2,3],[4,5,6]]
    A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3;
    A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;
    // B = [[7,8],[9,10],[11,12]]
    B.at(0,0)=7;  B.at(0,1)=8;
    B.at(1,0)=9;  B.at(1,1)=10;
    B.at(2,0)=11; B.at(2,1)=12;

    auto C = matmul(A, B);
    // C[0,0] = 1*7+2*9+3*11 = 7+18+33 = 58
    CHECK_NEAR(C.at(0,0), 58.f, 1e-4f);
    // C[0,1] = 1*8+2*10+3*12 = 8+20+36 = 64
    CHECK_NEAR(C.at(0,1), 64.f, 1e-4f);
    // C[1,0] = 4*7+5*9+6*11 = 28+45+66 = 139
    CHECK_NEAR(C.at(1,0), 139.f, 1e-4f);
}

void test_matmul_bt() {
    SECTION("matmul_bt: A · Bᵀ");
    // A [2×3], B [2×3] -> C [2×2]  (C = A · Bᵀ)
    Matrix2D A(2,3), B(2,3);
    A.at(0,0)=1; A.at(0,1)=0; A.at(0,2)=0;
    A.at(1,0)=0; A.at(1,1)=1; A.at(1,2)=0;
    B.at(0,0)=1; B.at(0,1)=0; B.at(0,2)=0;
    B.at(1,0)=0; B.at(1,1)=1; B.at(1,2)=0;

    auto C = matmul_bt(A, B);
    // Identity-like: C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=1
    CHECK_NEAR(C.at(0,0), 1.f, 1e-5f);
    CHECK_NEAR(C.at(0,1), 0.f, 1e-5f);
    CHECK_NEAR(C.at(1,0), 0.f, 1e-5f);
    CHECK_NEAR(C.at(1,1), 1.f, 1e-5f);
}

void test_transpose() {
    SECTION("transpose");
    Matrix2D A(2, 3);
    A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3;
    A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;

    auto B = transpose(A);
    CHECK(B.rows == 3 && B.cols == 2);
    CHECK_NEAR(B.at(0,0), 1.f, 1e-5f);
    CHECK_NEAR(B.at(2,1), 6.f, 1e-5f);
    CHECK_NEAR(B.at(1,0), 2.f, 1e-5f);
}

void test_softmax() {
    SECTION("softmax rows sum to 1");
    Matrix2D M(3, 4);
    M.at(0,0)=1; M.at(0,1)=2; M.at(0,2)=3; M.at(0,3)=4;
    M.at(1,0)=0; M.at(1,1)=0; M.at(1,2)=0; M.at(1,3)=0;
    M.at(2,0)=-1;M.at(2,1)=5; M.at(2,2)=2; M.at(2,3)=-3;

    softmax_inplace(M);

    for (std::size_t r = 0; r < 3; ++r)
        CHECK_NEAR(row_sum(M, r), 1.f, 1e-5f);

    // All values in [0,1]
    for (float v : M.data) {
        CHECK(v >= 0.f);
        CHECK(v <= 1.f + 1e-5f);
    }
}

void test_softmax_uniform() {
    SECTION("softmax of identical values = uniform distribution");
    Matrix2D M(2, 4, 1.f);
    softmax_inplace(M);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            CHECK_NEAR(M.at(r,c), 0.25f, 1e-5f);
}

void test_softmax_numerical_stability() {
    SECTION("softmax handles large values without overflow");
    Matrix2D M(1, 3);
    M.at(0,0) = 1000.f; M.at(0,1) = 1001.f; M.at(0,2) = 1002.f;
    softmax_inplace(M);
    CHECK_NEAR(row_sum(M, 0), 1.f, 1e-5f);
    // Largest value should dominate
    CHECK(M.at(0,2) > M.at(0,1));
    CHECK(M.at(0,1) > M.at(0,0));
}

// ── Attention forward tests ───────────────────────────────

void test_attention_output_shape() {
    SECTION("attention forward output shape [seq_len × d_v]");
    std::size_t seq=4, dk=8, dv=8;
    auto Q = iota(seq, dk);
    auto K = iota(seq, dk);
    auto V = iota(seq, dv);

    AttentionCache cache;
    auto out = ScaledDotProductAttention::forward(
        Q, K, V, all_real(seq), /*causal=*/true, cache);

    CHECK(out.rows == seq);
    CHECK(out.cols == dv);
}

void test_attention_weights_sum_to_1() {
    SECTION("attention weights (rows) sum to 1 after softmax");
    std::size_t seq=5, dk=4;
    auto Q = iota(seq, dk);
    auto K = iota(seq, dk);
    auto V = iota(seq, dk);

    AttentionCache cache;
    (void)ScaledDotProductAttention::forward(
        Q, K, V, all_real(seq), /*causal=*/false, cache);

    for (std::size_t i = 0; i < seq; ++i)
        CHECK_NEAR(row_sum(cache.weights, i), 1.f, 1e-4f);
}

void test_causal_mask_blocks_future() {
    SECTION("causal mask: no attention to future positions");
    std::size_t seq=4, dk=4;
    auto Q = fill(seq, dk, 1.f);
    auto K = fill(seq, dk, 1.f);
    auto V = fill(seq, dk, 1.f);

    AttentionCache cache;
    (void)ScaledDotProductAttention::forward(
        Q, K, V, all_real(seq), /*causal=*/true, cache);

    // weights[i,j] must be 0 for j > i
    for (std::size_t i = 0; i < seq; ++i)
        for (std::size_t j = i+1; j < seq; ++j)
            CHECK_NEAR(cache.weights.at(i,j), 0.f, 1e-6f);

    // weights[i, 0..i] must be positive and sum to 1
    for (std::size_t i = 0; i < seq; ++i) {
        float s = 0.f;
        for (std::size_t j = 0; j <= i; ++j) {
            CHECK(cache.weights.at(i,j) > 0.f);
            s += cache.weights.at(i,j);
        }
        CHECK_NEAR(s, 1.f, 1e-4f);
    }
}

void test_padding_mask_blocks_pad_keys() {
    SECTION("padding mask zeros out attention to pad positions");
    std::size_t seq=4, dk=4;
    auto Q = fill(seq, dk, 1.f);
    auto K = fill(seq, dk, 1.f);
    auto V = fill(seq, dk, 1.f);

    // Last two positions are padding
    std::vector<bool> mask = {true, true, false, false};

    AttentionCache cache;
    (void)ScaledDotProductAttention::forward(
        Q, K, V, mask, /*causal=*/false, cache);

    // Columns 2 and 3 of weights must be zero
    for (std::size_t i = 0; i < seq; ++i) {
        CHECK_NEAR(cache.weights.at(i,2), 0.f, 1e-6f);
        CHECK_NEAR(cache.weights.at(i,3), 0.f, 1e-6f);
    }
}

void test_attention_identity_v() {
    SECTION("attention with uniform weights returns mean of V rows");
    // If Q=K (all equal), causal=false, then weights are uniform (1/seq).
    // Output row i = mean of all V rows.
    std::size_t seq=3, dk=2, dv=2;

    auto Q = fill(seq, dk, 1.f);
    auto K = fill(seq, dk, 1.f);

    // V: each row is [row_index, row_index]
    Matrix2D V(seq, dv);
    for (std::size_t i = 0; i < seq; ++i)
        for (std::size_t d = 0; d < dv; ++d)
            V.at(i,d) = static_cast<float>(i);

    AttentionCache cache;
    auto out = ScaledDotProductAttention::forward(
        Q, K, V, all_real(seq), /*causal=*/false, cache);

    // mean of V rows = (0+1+2)/3 = 1.0 for each dim
    for (std::size_t i = 0; i < seq; ++i)
        for (std::size_t d = 0; d < dv; ++d)
            CHECK_NEAR(out.at(i,d), 1.f, 1e-4f);
}

void test_scaling_by_sqrt_dk() {
    SECTION("scores are scaled by 1/√d_k");
    std::size_t seq=2, dk=4;
    // Q = K = identity-like so QKᵀ is known
    Matrix2D Q(seq, dk, 0.f), K(seq, dk, 0.f);
    Q.at(0,0) = 1.f;   // Q[0] = [1,0,0,0]
    Q.at(1,1) = 1.f;   // Q[1] = [0,1,0,0]
    K.at(0,0) = 1.f;
    K.at(1,1) = 1.f;

    // QKᵀ[0,0] = 1, QKᵀ[0,1] = 0 -> after scaling: 1/√4=0.5, 0
    auto V = fill(seq, dk, 1.f);
    AttentionCache cache;
    (void)ScaledDotProductAttention::forward(
        Q, K, V, all_real(seq), /*causal=*/false, cache);

    // scores[0,0] should be 1/√4 = 0.5 before softmax
    CHECK_NEAR(cache.scores.at(0,0), 0.5f, 1e-5f);
    CHECK_NEAR(cache.scores.at(0,1), 0.f,  1e-5f);
}

// ── Attention backward tests ──────────────────────────────

void test_backward_shapes() {
    SECTION("backward() gradient shapes match Q/K/V");
    std::size_t seq=4, dk=8, dv=8;
    auto Q = iota(seq, dk);
    auto K = iota(seq, dk);
    auto V = iota(seq, dv);

    AttentionCache cache;
    (void)ScaledDotProductAttention::forward(
        Q, K, V, all_real(seq), true, cache);

    auto dOut = fill(seq, dv, 1.f);
    auto grads = ScaledDotProductAttention::backward(dOut, cache);

    CHECK(grads.dQ.rows == seq && grads.dQ.cols == dk);
    CHECK(grads.dK.rows == seq && grads.dK.cols == dk);
    CHECK(grads.dV.rows == seq && grads.dV.cols == dv);
}

void test_backward_dv_correctness() {
    SECTION("backward dV = weightsᵀ · dOut");
    std::size_t seq=3, dk=4, dv=4;
    auto Q = fill(seq, dk, 1.f);
    auto K = fill(seq, dk, 1.f);
    auto V = iota(seq, dv);

    AttentionCache cache;
    (void)ScaledDotProductAttention::forward(
        Q, K, V, all_real(seq), /*causal=*/false, cache);

    auto dOut = fill(seq, dv, 1.f);
    auto grads = ScaledDotProductAttention::backward(dOut, cache);

    // dV[i,j] = sum_k weights[k,i] * dOut[k,j]
    // Since dOut is all-ones: dV[i,j] = sum_k weights[k,i]
    // = column i sum of weights
    for (std::size_t i = 0; i < seq; ++i) {
        float col_sum = 0.f;
        for (std::size_t k = 0; k < seq; ++k)
            col_sum += cache.weights.at(k, i);
        for (std::size_t d = 0; d < dv; ++d)
            CHECK_NEAR(grads.dV.at(i,d), col_sum, 1e-4f);
    }
}

void test_softmax_backward_correctness() {
    SECTION("softmax_backward satisfies Jacobian property");
    // For softmax: sum_j dScores[i,j] * 1 = 0 (when dW = const)
    // because d/dx softmax sums to 0 along output dimension
    Matrix2D W(2, 3);
    W.at(0,0)=0.2f; W.at(0,1)=0.5f; W.at(0,2)=0.3f;
    W.at(1,0)=0.1f; W.at(1,1)=0.7f; W.at(1,2)=0.2f;

    Matrix2D dW(2, 3, 1.f);  // uniform upstream gradient
    auto dS = softmax_backward(dW, W);

    // sum of each row of dS should be 0
    for (std::size_t i = 0; i < 2; ++i)
        CHECK_NEAR(row_sum(dS, i), 0.f, 1e-5f);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — Attention tests\n";
    std::cout << "=========================\n";

    test_matmul();
    test_matmul_bt();
    test_transpose();
    test_softmax();
    test_softmax_uniform();
    test_softmax_numerical_stability();

    test_attention_output_shape();
    test_attention_weights_sum_to_1();
    test_causal_mask_blocks_future();
    test_padding_mask_blocks_pad_keys();
    test_attention_identity_v();
    test_scaling_by_sqrt_dk();

    test_backward_shapes();
    test_backward_dv_correctness();
    test_softmax_backward_correctness();

    std::cout << "\n=========================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
