// embed_test.cpp  —  Unit tests for miniGPT::EmbeddingTable
//
// Build (from miniGPT/):
//   make embed_test && ./embed_test

#include "../embed/Embedding.hpp"
#include "../vocab/Vocab.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>
#include <filesystem>

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
        float _a = (a), _b = (b);                                     \
        if (std::abs(_a - _b) > (eps)) {                              \
            std::cerr << "FAIL  " #a " ≈ " #b                        \
                      << "  got " << _a << " vs " << _b              \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";\
        } else { ++passed; }                                          \
    } while (0)

#define SECTION(name) std::cout << "\n── " name " ──\n"

using namespace miniGPT;

// ── Helpers ───────────────────────────────────────────────

// All-true mask of length n
std::vector<bool> all_real(std::size_t n) {
    return std::vector<bool>(n, true);
}

// Mask with trailing padding
std::vector<bool> with_padding(std::size_t real, std::size_t total) {
    std::vector<bool> m(total, false);
    for (std::size_t i = 0; i < real; ++i) m[i] = true;
    return m;
}

// L2 norm of a row
float row_norm(const Matrix2D& M, std::size_t r) {
    float sum = 0.f;
    for (std::size_t c = 0; c < M.cols; ++c)
        sum += M.at(r, c) * M.at(r, c);
    return std::sqrt(sum);
}

// ── Tests ─────────────────────────────────────────────────

void test_construction() {
    SECTION("Construction and dimensions");
    EmbeddingTable emb(/*vocab_size=*/100, /*d_model=*/16,
                       /*max_len=*/32, /*seed=*/42);

    CHECK(emb.vocab_size() == 100);
    CHECK(emb.d_model()    == 16);
    CHECK(emb.max_len()    == 32);
    CHECK(emb.E().rows     == 100);
    CHECK(emb.E().cols     == 16);
    CHECK(emb.PE().rows    == 32);
    CHECK(emb.PE().cols    == 16);
}

void test_pad_row_is_zero() {
    SECTION("PAD embedding (ID 0) is all zeros");
    EmbeddingTable emb(50, 16, 32, 42);

    const float* pad_row = emb.E().row_ptr(kPAD_ID);
    for (std::size_t d = 0; d < emb.d_model(); ++d)
        CHECK(pad_row[d] == 0.f);
}

void test_non_pad_rows_nonzero() {
    SECTION("Non-PAD embeddings are non-zero after Xavier init");
    EmbeddingTable emb(50, 16, 32, 42);

    // Check a handful of rows
    for (TokenID id = 1; id < 10; ++id) {
        float norm = row_norm(emb.E(), id);
        CHECK(norm > 0.f);
    }
}

void test_xavier_range() {
    SECTION("Xavier init values are within expected range");
    // limit = sqrt(6 / (1 + d_model))
    const std::size_t d = 128;
    EmbeddingTable emb(200, d, 64, 7);
    float limit = std::sqrt(6.f / (1.f + static_cast<float>(d)));

    for (std::size_t i = 1; i < 200; ++i) {
        const float* row = emb.E().row_ptr(i);
        for (std::size_t j = 0; j < d; ++j) {
            CHECK(row[j] >= -limit - 1e-5f);
            CHECK(row[j] <=  limit + 1e-5f);
        }
    }
}

void test_positional_encoding_values() {
    SECTION("Sinusoidal PE values are correct");
    EmbeddingTable emb(10, 8, 16, 1);
    const Matrix2D& PE = emb.PE();

    // PE[0] should be all sin(0)=0 and cos(0)=1 alternating
    CHECK_NEAR(PE.at(0, 0), 0.f,  1e-5f);  // sin(0) = 0
    CHECK_NEAR(PE.at(0, 1), 1.f,  1e-5f);  // cos(0) = 1
    CHECK_NEAR(PE.at(0, 2), 0.f,  1e-5f);  // sin(0) = 0
    CHECK_NEAR(PE.at(0, 3), 1.f,  1e-5f);  // cos(0) = 1

    // PE[1, 0] = sin(1 / 10000^0) = sin(1)
    CHECK_NEAR(PE.at(1, 0), std::sin(1.f), 1e-5f);
    // PE[1, 1] = cos(1 / 10000^0) = cos(1)
    CHECK_NEAR(PE.at(1, 1), std::cos(1.f), 1e-5f);

    // PE values must be in [-1, 1]
    for (std::size_t p = 0; p < PE.rows; ++p)
        for (std::size_t d = 0; d < PE.cols; ++d)
            CHECK(std::abs(PE.at(p, d)) <= 1.f + 1e-5f);
}

void test_pe_position_0_matches_all_positions() {
    SECTION("PE position 0 is identical for all d_model sizes");
    // At p=0: sin(0)=0 for even dims, cos(0)=1 for odd dims
    EmbeddingTable emb(10, 128, 64, 1);
    const Matrix2D& PE = emb.PE();
    for (std::size_t d = 0; d < 128; d += 2) {
        CHECK_NEAR(PE.at(0, d),     0.f, 1e-5f);  // sin(0)
        CHECK_NEAR(PE.at(0, d + 1), 1.f, 1e-5f);  // cos(0)
    }
}

void test_forward_output_shape() {
    SECTION("forward() output shape is [seq_len × d_model]");
    EmbeddingTable emb(20, 16, 32, 42);

    std::vector<TokenID> ids  = {2, 5, 6, 7, 3};   // BOS + 3 content + EOS
    std::vector<bool>    mask = all_real(5);

    auto X = emb.forward(ids, mask);   // [5 × 16]

    CHECK(X.rows == 5);
    CHECK(X.cols == 16);
}

void test_forward_is_embedding_plus_pe() {
    SECTION("forward() = E[id] + PE[pos] for each position");
    EmbeddingTable emb(20, 8, 16, 42);

    std::vector<TokenID> ids  = {2, 5, 6};
    std::vector<bool>    mask = all_real(3);

    auto X = emb.forward(ids, mask);   // [3 × 8]

    for (std::size_t p = 0; p < 3; ++p) {
        TokenID tid = ids[p];
        for (std::size_t d = 0; d < 8; ++d) {
            float expected = emb.E().at(tid, d) + emb.PE().at(p, d);
            CHECK_NEAR(X.at(p, d), expected, 1e-5f);
        }
    }
}

void test_forward_padding_is_zero() {
    SECTION("Padding positions produce zero vectors");
    EmbeddingTable emb(20, 8, 16, 42);

    // [BOS, tok, EOS, PAD, PAD]
    std::vector<TokenID> ids  = {kBOS_ID, 5, kEOS_ID, kPAD_ID, kPAD_ID};
    std::vector<bool>    mask = with_padding(3, 5);

    auto X = emb.forward(ids, mask);   // [5 × 8]

    // Positions 3 and 4 are padding — must be zero
    for (std::size_t d = 0; d < 8; ++d) {
        CHECK(X.at(3, d) == 0.f);
        CHECK(X.at(4, d) == 0.f);
    }

    // Position 0 (BOS) must be non-zero (E[BOS] + PE[0])
    float norm_bos = row_norm(X, 0);
    CHECK(norm_bos > 0.f);
}

void test_forward_same_id_different_position() {
    SECTION("Same token ID at different positions → different output (due to PE)");
    EmbeddingTable emb(20, 16, 32, 42);

    // token ID 5 at position 0 vs position 2
    std::vector<TokenID> ids  = {5, 6, 5};
    std::vector<bool>    mask = all_real(3);

    auto X = emb.forward(ids, mask);

    // X[0] and X[2] should differ because PE[0] != PE[2]
    bool any_diff = false;
    for (std::size_t d = 0; d < 16; ++d)
        if (std::abs(X.at(0, d) - X.at(2, d)) > 1e-6f)
            any_diff = true;
    CHECK(any_diff);
}

void test_backward_accumulates_grad() {
    SECTION("backward() accumulates gradients into dE");
    EmbeddingTable emb(20, 4, 16, 42);
    emb.zero_grad();

    std::vector<TokenID> ids  = {2, 5, 3};   // BOS, tok5, EOS
    std::vector<bool>    mask = all_real(3);

    // Upstream gradient: all ones  [3 × 4]
    Matrix2D dX(3, 4, 1.f);

    emb.backward(dX, ids, mask);

    // dE[2] (BOS) and dE[5] (tok5) and dE[3] (EOS) should be 1.0
    // dE[0] (PAD) must remain zero
    for (std::size_t d = 0; d < 4; ++d) {
        CHECK_NEAR(emb.dE().at(kBOS_ID, d), 1.f, 1e-6f);
        CHECK_NEAR(emb.dE().at(5,       d), 1.f, 1e-6f);
        CHECK_NEAR(emb.dE().at(kEOS_ID, d), 1.f, 1e-6f);
        CHECK(emb.dE().at(kPAD_ID, d) == 0.f);
    }
}

void test_backward_accumulates_across_positions() {
    SECTION("backward() accumulates when same token appears twice");
    EmbeddingTable emb(20, 4, 16, 42);
    emb.zero_grad();

    // Token 5 appears at positions 0 and 2
    std::vector<TokenID> ids  = {5, 6, 5};
    std::vector<bool>    mask = all_real(3);

    Matrix2D dX(3, 4, 1.f);
    emb.backward(dX, ids, mask);

    // dE[5] should be 2.0 (accumulated from both positions)
    for (std::size_t d = 0; d < 4; ++d)
        CHECK_NEAR(emb.dE().at(5, d), 2.f, 1e-6f);

    // dE[6] should be 1.0
    for (std::size_t d = 0; d < 4; ++d)
        CHECK_NEAR(emb.dE().at(6, d), 1.f, 1e-6f);
}

void test_zero_grad() {
    SECTION("zero_grad() clears gradient accumulator");
    EmbeddingTable emb(20, 4, 16, 42);

    std::vector<TokenID> ids  = {5, 6};
    std::vector<bool>    mask = all_real(2);
    Matrix2D dX(2, 4, 1.f);

    emb.backward(dX, ids, mask);
    emb.zero_grad();

    for (std::size_t i = 0; i < 20; ++i)
        for (std::size_t d = 0; d < 4; ++d)
            CHECK(emb.dE().at(i, d) == 0.f);
}

void test_sgd_step_updates_weights() {
    SECTION("sgd_step() updates E weights, leaves PAD zero");
    EmbeddingTable emb(20, 4, 16, 42);

    // Record original weight for token 5
    float orig = emb.E().at(5, 0);

    std::vector<TokenID> ids  = {5};
    std::vector<bool>    mask = all_real(1);
    Matrix2D dX(1, 4, 1.f);

    emb.zero_grad();
    emb.backward(dX, ids, mask);
    emb.sgd_step(0.1f);

    // Weight should have moved by -lr * grad = -0.1 * 1.0
    CHECK_NEAR(emb.E().at(5, 0), orig - 0.1f, 1e-5f);

    // PAD row must stay zero
    for (std::size_t d = 0; d < 4; ++d)
        CHECK(emb.E().at(kPAD_ID, d) == 0.f);
}

void test_save_load_roundtrip() {
    SECTION("save() / load() round-trip");
    const std::string path = "/tmp/test_embeddings.bin";

    EmbeddingTable orig(30, 8, 16, 99);
    orig.save(path);

    EmbeddingTable loaded(30, 8, 16, 0);  // different seed
    loaded.load(path);

    // All weights must match exactly
    for (std::size_t i = 0; i < 30; ++i)
        for (std::size_t d = 0; d < 8; ++d)
            CHECK_NEAR(loaded.E().at(i, d), orig.E().at(i, d), 1e-7f);

    std::filesystem::remove(path);
}

void test_invalid_construction() {
    SECTION("Zero dimension throws");
    bool threw = false;
    try { EmbeddingTable emb(0, 16, 32); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { EmbeddingTable emb(10, 0, 32); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

void test_print_stats_smoke() {
    SECTION("print_stats() smoke test");
    EmbeddingTable emb(20, 8, 16, 42);
    std::ostringstream oss;
    emb.print_stats(oss);
    CHECK(oss.str().find("EmbeddingTable") != std::string::npos);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — EmbeddingTable tests\n";
    std::cout << "==============================\n";

    test_construction();
    test_pad_row_is_zero();
    test_non_pad_rows_nonzero();
    test_xavier_range();
    test_positional_encoding_values();
    test_pe_position_0_matches_all_positions();
    test_forward_output_shape();
    test_forward_is_embedding_plus_pe();
    test_forward_padding_is_zero();
    test_forward_same_id_different_position();
    test_backward_accumulates_grad();
    test_backward_accumulates_across_positions();
    test_zero_grad();
    test_sgd_step_updates_weights();
    test_save_load_roundtrip();
    test_invalid_construction();
    test_print_stats_smoke();

    std::cout << "\n==============================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
