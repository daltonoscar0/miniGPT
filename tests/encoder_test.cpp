// encoder_test.cpp, Unit tests for miniGPT::Encoder
//
// Tests the Encoder in isolation using hand-crafted analysis
// strings, no Scalpel or Morphology dependency needed here.
//
// Build (from miniGPT/):
//   make encoder_test && ./encoder_test

#include "../encoder/Encoder.hpp"
#include "../vocab/Vocab.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <string>

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

#define SECTION(name) std::cout << "\n── " name " ──\n"

using namespace miniGPT;

// ── Helpers ───────────────────────────────────────────────

// Build a small vocab seeded with a few morphological analyses
Vocab make_vocab() {
    Vocab v;
    v.add("the+DET");       // ID 4
    v.add("cat+N+SG");      // ID 5
    v.add("cat+N+PL");      // ID 6
    v.add("sit+V+PAST");    // ID 7
    v.add("run+V+INF");     // ID 8
    v.add("quick+ADJ");     // ID 9
    return v;
}

// ── Tests ─────────────────────────────────────────────────

void test_basic_encode() {
    SECTION("Basic encode — short sequence");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/8);

    auto seq = enc.encode({"the+DET", "cat+N+SG", "sit+V+PAST"});

    // Length is always exactly max_len
    CHECK(seq.ids.size() == 8);
    CHECK(seq.mask.size() == 8);

    // [BOS, the+DET, cat+N+SG, sit+V+PAST, EOS, PAD, PAD, PAD]
    CHECK(seq.ids[0] == kBOS_ID);          // BOS
    CHECK(seq.ids[1] == v.encode("the+DET"));
    CHECK(seq.ids[2] == v.encode("cat+N+SG"));
    CHECK(seq.ids[3] == v.encode("sit+V+PAST"));
    CHECK(seq.ids[4] == kEOS_ID);          // EOS
    CHECK(seq.ids[5] == kPAD_ID);          // PAD
    CHECK(seq.ids[6] == kPAD_ID);
    CHECK(seq.ids[7] == kPAD_ID);

    // n_real = BOS + 3 content + EOS = 5
    CHECK(seq.n_real == 5);

    // Mask: first 5 true, last 3 false
    for (std::size_t i = 0; i < 5; ++i) CHECK(seq.mask[i] == true);
    for (std::size_t i = 5; i < 8; ++i) CHECK(seq.mask[i] == false);
}

void test_exact_fit() {
    SECTION("Sequence fills max_len exactly (no padding)");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/6);

    // 4 content tokens + BOS + EOS = 6 = max_len
    auto seq = enc.encode({"the+DET", "cat+N+SG", "sit+V+PAST", "quick+ADJ"});

    CHECK(seq.ids.size() == 6);
    CHECK(seq.ids[0] == kBOS_ID);
    CHECK(seq.ids[5] == kEOS_ID);
    CHECK(seq.n_real == 6);
    // No padding
    for (std::size_t i = 0; i < 6; ++i) CHECK(seq.mask[i] == true);
    // No PAD tokens
    for (std::size_t i = 0; i < 6; ++i) CHECK(seq.ids[i] != kPAD_ID);
}

void test_truncation() {
    SECTION("Truncation — sequence longer than max_len");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/5);

    // 6 tokens but only room for 3 content (max_len=5: BOS+3+EOS)
    // Tail truncation -> keep last 3: cat+N+PL sit+V+PAST quick+ADJ
    std::vector<std::string> long_seq = {
        "the+DET",   // dropped
        "run+V+INF", // dropped
        "the+DET",   // dropped
        "cat+N+PL",  // kept (3rd from end)
        "sit+V+PAST",// kept
        "quick+ADJ"  // kept
    };

    auto seq = enc.encode(long_seq);

    CHECK(seq.ids.size() == 5);
    CHECK(seq.ids[0] == kBOS_ID);
    CHECK(seq.ids[1] == v.encode("cat+N+PL"));
    CHECK(seq.ids[2] == v.encode("sit+V+PAST"));
    CHECK(seq.ids[3] == v.encode("quick+ADJ"));
    CHECK(seq.ids[4] == kEOS_ID);
    CHECK(seq.n_real == 5);
}

void test_unknown_token() {
    SECTION("Unknown analysis → kUNK_ID");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/6);

    auto seq = enc.encode({"cat+N+SG", "xylophone+N+SG"}); // xylophone not in vocab

    CHECK(seq.ids[0] == kBOS_ID);
    CHECK(seq.ids[1] == v.encode("cat+N+SG"));
    CHECK(seq.ids[2] == kUNK_ID);   // unknown maps to UNK
    CHECK(seq.ids[3] == kEOS_ID);
}

void test_empty_sequence() {
    SECTION("Empty analysis sequence");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/6);

    auto seq = enc.encode({});

    CHECK(seq.ids.size() == 6);
    CHECK(seq.ids[0] == kBOS_ID);
    CHECK(seq.ids[1] == kEOS_ID);   // EOS immediately after BOS
    CHECK(seq.ids[2] == kPAD_ID);
    CHECK(seq.n_real == 2);         // just BOS + EOS
    CHECK(seq.mask[0] == true);
    CHECK(seq.mask[1] == true);
    CHECK(seq.mask[2] == false);
}

void test_single_token() {
    SECTION("Single analysis token");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/6);

    auto seq = enc.encode({"cat+N+SG"});

    CHECK(seq.ids[0] == kBOS_ID);
    CHECK(seq.ids[1] == v.encode("cat+N+SG"));
    CHECK(seq.ids[2] == kEOS_ID);
    CHECK(seq.ids[3] == kPAD_ID);
    CHECK(seq.n_real == 3);
}

void test_lm_pair_normal() {
    SECTION("encode_lm_pair — normal sequence");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/6);

    // analyses: [the+DET, cat+N+SG, sit+V+PAST]
    // input  = [BOS, the+DET, cat+N+SG, sit+V+PAST, PAD, PAD]
    // target = [the+DET, cat+N+SG, sit+V+PAST, EOS, PAD, PAD]
    auto pair = enc.encode_lm_pair({"the+DET", "cat+N+SG", "sit+V+PAST"});

    auto& in  = pair.input;
    auto& tgt = pair.target;

    CHECK(in.ids.size()  == 6);
    CHECK(tgt.ids.size() == 6);

    // Input starts with BOS
    CHECK(in.ids[0] == kBOS_ID);
    CHECK(in.ids[1] == v.encode("the+DET"));
    CHECK(in.ids[2] == v.encode("cat+N+SG"));
    CHECK(in.ids[3] == v.encode("sit+V+PAST"));
    CHECK(in.ids[4] == kPAD_ID);

    // Target is shifted: first token is the+DET, last real is EOS
    CHECK(tgt.ids[0] == v.encode("the+DET"));
    CHECK(tgt.ids[1] == v.encode("cat+N+SG"));
    CHECK(tgt.ids[2] == v.encode("sit+V+PAST"));
    CHECK(tgt.ids[3] == kEOS_ID);
    CHECK(tgt.ids[4] == kPAD_ID);

    // Masks align: 4 real tokens each
    CHECK(in.n_real  == 4);
    CHECK(tgt.n_real == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(in.mask[i]  == true);
        CHECK(tgt.mask[i] == true);
    }
    CHECK(in.mask[4]  == false);
    CHECK(tgt.mask[4] == false);
}

void test_lm_pair_empty() {
    SECTION("encode_lm_pair — empty sequence");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/4);

    auto pair = enc.encode_lm_pair({});

    // input  = [BOS, PAD, PAD, PAD]
    // target = [EOS, PAD, PAD, PAD]
    CHECK(pair.input.ids[0]  == kBOS_ID);
    CHECK(pair.target.ids[0] == kEOS_ID);
    CHECK(pair.input.n_real  == 1);
    CHECK(pair.target.n_real == 1);
}

void test_min_length_constructor() {
    SECTION("max_len=2 (minimum valid)");
    auto v = make_vocab();
    Encoder enc(v, /*max_len=*/2);

    auto seq = enc.encode({"cat+N+SG", "run+V+INF"});
    // No room for content, BOS + EOS only, everything truncated
    CHECK(seq.ids.size() == 2);
    CHECK(seq.ids[0] == kBOS_ID);
    CHECK(seq.ids[1] == kEOS_ID);
    CHECK(seq.n_real == 2);
}

void test_invalid_max_len() {
    SECTION("max_len < 2 throws");
    auto v = make_vocab();
    bool threw = false;
    try {
        Encoder enc(v, /*max_len=*/1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_accessor() {
    SECTION("max_len() and vocab() accessors");
    auto v = make_vocab();
    Encoder enc(v, 32);
    CHECK(enc.max_len() == 32);
    CHECK(enc.vocab().size() == v.size());
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — Encoder tests\n";
    std::cout << "=======================\n";

    test_basic_encode();
    test_exact_fit();
    test_truncation();
    test_unknown_token();
    test_empty_sequence();
    test_single_token();
    test_lm_pair_normal();
    test_lm_pair_empty();
    test_min_length_constructor();
    test_invalid_max_len();
    test_accessor();

    std::cout << "\n=======================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
