#pragma once

// ============================================================
//  Attention.hpp  —  Scaled dot-product attention
//
//  Implements the core attention operation from
//  "Attention Is All You Need" (Vaswani et al., 2017):
//
//    Attention(Q, K, V) = softmax(QKᵀ / √d_k + M) · V
//
//  where M is an additive causal mask (−∞ for future positions).
//
//  Shapes (all row-major Matrix2D)
//  ────────────────────────────────
//  Q        [seq_len × d_k]
//  K        [seq_len × d_k]
//  V        [seq_len × d_v]
//  scores   [seq_len × seq_len]   QKᵀ / √d_k
//  weights  [seq_len × seq_len]   softmax(scores + mask)
//  output   [seq_len × d_v]       weights · V
//
//  Causal mask
//  ───────────
//  For position (i, j):
//    j <= i  →  0       (can attend: past and present)
//    j >  i  →  -∞      (cannot attend: future)
//  Applied as additive bias before softmax so -∞ → exp(-∞) = 0.
//  We use -1e9 as a practical stand-in for -∞.
//
//  Padding mask
//  ────────────
//  Positions where mask[j] == false (padding) are also blocked
//  by adding -∞ to column j of the score matrix.
//
//  Backward pass
//  ─────────────
//  Given dOut = dL/dOutput  [seq_len × d_v]:
//
//  1. dV      = weightsᵀ · dOut            [seq_len × d_v]
//  2. dW      = dOut · Vᵀ                  [seq_len × seq_len]
//     (gradient through the weighted sum)
//  3. dScores = softmax_backward(dW, W)    [seq_len × seq_len]
//     where softmax_backward[i,j] =
//       W[i,j] * (dW[i,j] - sum_k(dW[i,k] * W[i,k]))
//  4. dQ      = dScores · K / √d_k         [seq_len × d_k]
//     dK      = dScoresᵀ · Q / √d_k        [seq_len × d_k]
//
//  The causal and padding mask positions have zero gradient
//  (their softmax output is 0 so they vanish in step 3).
// ============================================================

#include "../embed/Embedding.hpp"   // for Matrix2D

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cassert>

namespace miniGPT {

// ── Constants ─────────────────────────────────────────────
constexpr float kNegInf = -1e9f;   // practical −∞ for masking

// ── Free matrix operations ────────────────────────────────
// These are used by attention and will be reused by FFN, LayerNorm, etc.

// Matrix multiply: C = A · B
// A [m × k], B [k × n] → C [m × n]
inline Matrix2D matmul(const Matrix2D& A, const Matrix2D& B) {
    if (A.cols != B.rows)
        throw std::invalid_argument(
            "matmul: dimension mismatch " +
            std::to_string(A.cols) + " vs " + std::to_string(B.rows));

    Matrix2D C(A.rows, B.cols, 0.f);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t k = 0; k < A.cols; ++k) {
            float aik = A.at(i, k);
            if (aik == 0.f) continue;   // skip zero (sparse speedup)
            for (std::size_t j = 0; j < B.cols; ++j)
                C.at(i, j) += aik * B.at(k, j);
        }
    return C;
}

// Matrix multiply with transpose on right: C = A · Bᵀ
// A [m × k], B [n × k] → C [m × n]
inline Matrix2D matmul_bt(const Matrix2D& A, const Matrix2D& B) {
    if (A.cols != B.cols)
        throw std::invalid_argument(
            "matmul_bt: dimension mismatch " +
            std::to_string(A.cols) + " vs " + std::to_string(B.cols));

    Matrix2D C(A.rows, B.rows, 0.f);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < B.rows; ++j) {
            float sum = 0.f;
            for (std::size_t k = 0; k < A.cols; ++k)
                sum += A.at(i, k) * B.at(j, k);
            C.at(i, j) = sum;
        }
    return C;
}

// Matrix multiply with transpose on left: C = Aᵀ · B
// A [k × m], B [k × n] → C [m × n]
inline Matrix2D matmul_at(const Matrix2D& A, const Matrix2D& B) {
    if (A.rows != B.rows)
        throw std::invalid_argument(
            "matmul_at: dimension mismatch " +
            std::to_string(A.rows) + " vs " + std::to_string(B.rows));

    Matrix2D C(A.cols, B.cols, 0.f);
    for (std::size_t k = 0; k < A.rows; ++k)
        for (std::size_t i = 0; i < A.cols; ++i) {
            float aki = A.at(k, i);
            if (aki == 0.f) continue;
            for (std::size_t j = 0; j < B.cols; ++j)
                C.at(i, j) += aki * B.at(k, j);
        }
    return C;
}

// Transpose: B = Aᵀ
// A [m × n] → B [n × m]
inline Matrix2D transpose(const Matrix2D& A) {
    Matrix2D B(A.cols, A.rows);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < A.cols; ++j)
            B.at(j, i) = A.at(i, j);
    return B;
}

// Row-wise softmax in place
// Each row r: out[r,j] = exp(in[r,j] - max_j) / sum_j exp(...)
inline void softmax_inplace(Matrix2D& M) {
    for (std::size_t i = 0; i < M.rows; ++i) {
        float* row = M.row_ptr(i);
        // Numerically stable: subtract row max before exp
        float maxval = row[0];
        for (std::size_t j = 1; j < M.cols; ++j)
            maxval = std::max(maxval, row[j]);

        float sum = 0.f;
        for (std::size_t j = 0; j < M.cols; ++j) {
            row[j] = std::exp(row[j] - maxval);
            sum += row[j];
        }
        for (std::size_t j = 0; j < M.cols; ++j)
            row[j] /= sum;
    }
}

// Softmax backward for one row
// dScores[i,j] = W[i,j] * (dW[i,j] - dot(dW[i,:], W[i,:]))
// Applied row-wise to the full matrix.
inline Matrix2D softmax_backward(const Matrix2D& dW,
                                  const Matrix2D& W)
{
    assert(dW.rows == W.rows && dW.cols == W.cols);
    Matrix2D dS(dW.rows, dW.cols);

    for (std::size_t i = 0; i < W.rows; ++i) {
        // dot product: sum_k dW[i,k] * W[i,k]
        float dot = 0.f;
        for (std::size_t k = 0; k < W.cols; ++k)
            dot += dW.at(i, k) * W.at(i, k);

        for (std::size_t j = 0; j < W.cols; ++j)
            dS.at(i, j) = W.at(i, j) * (dW.at(i, j) - dot);
    }
    return dS;
}

// ── AttentionCache ────────────────────────────────────────
// Stores intermediate values needed for the backward pass.
// The forward pass fills this; backward pass reads it.
struct AttentionCache {
    Matrix2D Q;        // [seq_len × d_k]
    Matrix2D K;        // [seq_len × d_k]
    Matrix2D V;        // [seq_len × d_v]
    Matrix2D scores;   // [seq_len × seq_len]  pre-softmax
    Matrix2D weights;  // [seq_len × seq_len]  post-softmax
    float    scale;    // 1 / sqrt(d_k)
};

// ── ScaledDotProductAttention ─────────────────────────────
class ScaledDotProductAttention {
public:
    // ── Forward ───────────────────────────────────────────

    // Compute attention output and fill cache for backward.
    //
    // @param Q      [seq_len × d_k]
    // @param K      [seq_len × d_k]
    // @param V      [seq_len × d_v]
    // @param mask   [seq_len] — true = real token, false = padding
    // @param causal  true for LM (can't see future positions)
    // @param cache   filled with intermediates for backward pass
    // @returns       [seq_len × d_v]
    [[nodiscard]] static Matrix2D forward(
            const Matrix2D&        Q,
            const Matrix2D&        K,
            const Matrix2D&        V,
            const std::vector<bool>& mask,
            bool                   causal,
            AttentionCache&        cache)
    {
        const std::size_t seq_len = Q.rows;
        const std::size_t d_k     = Q.cols;

        if (K.rows != seq_len || V.rows != seq_len)
            throw std::invalid_argument("Attention: Q/K/V seq_len mismatch");
        if (K.cols != d_k)
            throw std::invalid_argument("Attention: Q/K d_k mismatch");

        const float scale = 1.f / std::sqrt(static_cast<float>(d_k));

        // ── Step 1: scores = Q · Kᵀ / √d_k  [seq_len × seq_len] ──
        Matrix2D scores = matmul_bt(Q, K);   // [seq_len × seq_len]
        for (float& v : scores.data) v *= scale;

        // ── Step 2: apply causal mask ─────────────────────────────
        if (causal) {
            for (std::size_t i = 0; i < seq_len; ++i)
                for (std::size_t j = i + 1; j < seq_len; ++j)
                    scores.at(i, j) = kNegInf;
        }

        // ── Step 3: apply padding mask ────────────────────────────
        // Block attention TO any padding key position
        for (std::size_t j = 0; j < seq_len; ++j) {
            if (!mask[j]) {
                for (std::size_t i = 0; i < seq_len; ++i)
                    scores.at(i, j) = kNegInf;
            }
        }

        // ── Step 4: softmax row-wise ──────────────────────────────
        Matrix2D weights = scores;           // copy before softmax
        softmax_inplace(weights);            // [seq_len × seq_len]

        // ── Step 5: output = weights · V  [seq_len × d_v] ────────
        Matrix2D output = matmul(weights, V);

        // ── Save to cache ─────────────────────────────────────────
        cache.Q       = Q;
        cache.K       = K;
        cache.V       = V;
        cache.scores  = scores;
        cache.weights = weights;
        cache.scale   = scale;

        return output;  // [seq_len × d_v]
    }

    // ── Backward ──────────────────────────────────────────

    // Compute gradients w.r.t. Q, K, V given upstream gradient dOut.
    //
    // @param dOut   [seq_len × d_v]  upstream gradient
    // @param cache  filled by forward()
    // @returns      {dQ, dK, dV}  each matching Q/K/V shapes
    struct Gradients {
        Matrix2D dQ;  // [seq_len × d_k]
        Matrix2D dK;  // [seq_len × d_k]
        Matrix2D dV;  // [seq_len × d_v]
    };

    [[nodiscard]] static Gradients backward(
            const Matrix2D&      dOut,
            const AttentionCache& cache)
    {
        // Step 1: dV = weightsᵀ · dOut       [seq_len × d_v]
        Matrix2D dV = matmul_at(cache.weights, dOut);

        // Step 2: dWeights = dOut · Vᵀ        [seq_len × seq_len]
        Matrix2D dWeights = matmul_bt(dOut, cache.V);

        // Step 3: dScores = softmax_backward  [seq_len × seq_len]
        Matrix2D dScores = softmax_backward(dWeights, cache.weights);

        // Step 4: scale by 1/√d_k
        for (float& v : dScores.data) v *= cache.scale;

        // Step 5: dQ = dScores · K            [seq_len × d_k]
        Matrix2D dQ = matmul(dScores, cache.K);

        // Step 6: dK = dScoresᵀ · Q           [seq_len × d_k]
        Matrix2D dK = matmul_at(dScores, cache.Q);

        return {std::move(dQ), std::move(dK), std::move(dV)};
    }
};

} // namespace miniGPT
