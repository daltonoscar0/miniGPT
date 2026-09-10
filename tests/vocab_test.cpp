// vocab_test.cpp, Unit tests for miniGPT::Vocab
//
// Build:   g++ -std=c++17 -Wall -Wextra -o vocab_test vocab_test.cpp && ./vocab_test
// (no other dependencies)

#include "../vocab/Vocab.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

// ── Tiny test harness ─────────────────────────────────────
static int total = 0, passed = 0;

#define CHECK(expr)                                              \
    do {                                                         \
        ++total;                                                 \
        if (!(expr)) {                                           \
            std::cerr << "FAIL  " #expr                         \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        } else {                                                 \
            ++passed;                                            \
        }                                                        \
    } while (0)

#define SECTION(name) std::cout << "\n── " name " ──\n"

// ── Tests ─────────────────────────────────────────────────
using namespace miniGPT;

void test_specials() {
    SECTION("Special token IDs");
    Vocab v;
    // Size starts at 4 (PAD UNK BOS EOS)
    CHECK(v.size() == 4);

    // Reserved IDs are exactly 0-3
    CHECK(v.encode(kPAD) == kPAD_ID);
    CHECK(v.encode(kUNK) == kUNK_ID);
    CHECK(v.encode(kBOS) == kBOS_ID);
    CHECK(v.encode(kEOS) == kEOS_ID);

    // Re-adding specials must not grow the vocab
    v.add(kPAD);
    v.add("<UNK>");
    CHECK(v.size() == 4);

    // Decode round-trips
    CHECK(v.decode(kPAD_ID) == "<PAD>");
    CHECK(v.decode(kUNK_ID) == "<UNK>");
    CHECK(v.decode(kBOS_ID) == "<BOS>");
    CHECK(v.decode(kEOS_ID) == "<EOS>");
}

void test_add_and_encode() {
    SECTION("add() and encode()");
    Vocab v;

    // First insertion -> new ID (starts at 4)
    TokenID cat_id = v.add("cat+N+SG");
    CHECK(cat_id == 4);
    CHECK(v.size() == 5);

    // Duplicate insertion -> same ID
    TokenID cat_id2 = v.add("cat+N+SG");
    CHECK(cat_id2 == cat_id);
    CHECK(v.size() == 5);

    // Different type -> next ID
    TokenID cats_id = v.add("cat+N+PL");
    CHECK(cats_id == 5);

    // encode() matches add()
    CHECK(v.encode("cat+N+SG") == cat_id);
    CHECK(v.encode("cat+N+PL") == cats_id);

    // Unknown type -> kUNK_ID
    CHECK(v.encode("nonexistent+X") == kUNK_ID);

    // contains() works
    CHECK(v.contains("cat+N+SG"));
    CHECK(!v.contains("dog+N+SG"));
}

void test_whitespace_normalisation() {
    SECTION("Whitespace normalisation");
    Vocab v;
    TokenID id1 = v.add("  run+V+INF  ");
    TokenID id2 = v.add("run+V+INF");
    CHECK(id1 == id2);
    CHECK(v.contains("run+V+INF"));
    CHECK(v.encode("  run+V+INF") == id1);
}

void test_morphological_analyses() {
    SECTION("Morphological analysis strings (Scalpel/Morphology format)");
    Vocab v;

    // Simulate what Morphology produces
    std::vector<std::string> analyses = {
        "run+V+INF",       // verb, infinitive
        "run+V+PAST",      // verb, past
        "runner+N+SG",     // noun, singular
        "quick+ADJ",       // adjective
        "quickly+ADV",     // adverb (derived)
        "cat+N+SG",
        "cat+N+PL",
        "the+DET",
        "a+DET",
        "be+V+PRES+1SG",
    };

    v.add_all(analyses);
    CHECK(v.size() == 4 + analyses.size());

    // Every analysis is retrievable
    for (const auto& a : analyses) {
        CHECK(v.contains(a));
        TokenID id = v.encode(a);
        CHECK(v.decode(id) == a);
    }

    // Morphologically distinct -> distinct IDs
    CHECK(v.encode("run+V+INF") != v.encode("run+V+PAST"));
    CHECK(v.encode("cat+N+SG") != v.encode("cat+N+PL"));
}

void test_decode_out_of_range() {
    SECTION("decode() out-of-range throws");
    Vocab v;
    bool threw = false;
    try {
        [[maybe_unused]] const auto& s = v.decode(9999);
        (void)s;
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

void test_build_from_corpus() {
    SECTION("build_from_corpus()");

    // Simulate a tiny training corpus: "the cat sat on the mat"
    // repeated, so "the+DET" should have highest frequency
    std::vector<std::string> corpus;
    for (int i = 0; i < 10; ++i) corpus.push_back("the+DET");
    for (int i = 0; i < 5;  ++i) corpus.push_back("cat+N+SG");
    for (int i = 0; i < 5;  ++i) corpus.push_back("sat+V+PAST");
    for (int i = 0; i < 5;  ++i) corpus.push_back("on+PREP");
    for (int i = 0; i < 5;  ++i) corpus.push_back("mat+N+SG");
    // One rare token
    corpus.push_back("xylophone+N+SG");

    // Build with no cap
    Vocab v = Vocab::build_from_corpus(corpus);
    CHECK(v.size() == 4 + 6);  // 4 specials + 6 unique types

    // Most frequent type should be first non-special (ID 4)
    CHECK(v.decode(4) == "the+DET");

    // Rare token should still be present (no cutoff)
    CHECK(v.contains("xylophone+N+SG"));

    // Build with vocab cap of 5 (4 specials + 1 content type).
    // Only "the+DET" (freq=10) is strictly more frequent than all others (freq=5),
    // so it alone is guaranteed to survive the frequency cutoff.
    Vocab v2 = Vocab::build_from_corpus(corpus, /*max_vocab=*/5);
    CHECK(v2.size() == 5);
    CHECK(v2.contains("the+DET"));        // must be top-1 content
    CHECK(v2.decode(4) == "the+DET");     // first non-special slot
    // Rare type should be pruned
    CHECK(!v2.contains("xylophone+N+SG"));
    // Unknown falls back to <UNK>
    CHECK(v2.encode("xylophone+N+SG") == kUNK_ID);
}

void test_save_load_roundtrip() {
    SECTION("save() / load() round-trip");
    const std::string path = "/tmp/test_vocab.tsv";

    Vocab orig;
    orig.add("cat+N+SG");
    orig.add("cat+N+PL");
    orig.add("run+V+INF");
    orig.save(path);

    Vocab loaded;
    loaded.load(path);

    CHECK(loaded.size() == orig.size());
    CHECK(loaded.encode("cat+N+SG") == orig.encode("cat+N+SG"));
    CHECK(loaded.encode("cat+N+PL") == orig.encode("cat+N+PL"));
    CHECK(loaded.encode("run+V+INF") == orig.encode("run+V+INF"));
    CHECK(loaded.decode(4) == "cat+N+SG");

    // Specials survive the round-trip
    CHECK(loaded.encode(kPAD) == kPAD_ID);
    CHECK(loaded.encode(kUNK) == kUNK_ID);

    std::filesystem::remove(path);
}

void test_freeze_flag() {
    SECTION("freeze() flag (advisory)");
    Vocab v;
    CHECK(!v.is_frozen());
    v.freeze();
    CHECK(v.is_frozen());
    // freeze() doesn't actually block add(), it's advisory
    // (the training loop uses it as a sentinel, not a hard lock)
    v.add("cat+N+SG");  // should still work
    CHECK(v.size() == 5);
}

void test_print_stats() {
    SECTION("print_stats() smoke test");
    Vocab v;
    v.add("cat+N+SG");
    v.add("run+V+INF");
    std::ostringstream oss;
    v.print_stats(oss);
    std::string s = oss.str();
    CHECK(s.find("Vocab size: 6") != std::string::npos);
    CHECK(s.find("<PAD>") != std::string::npos);
    CHECK(s.find("<UNK>") != std::string::npos);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — Vocab tests\n";
    std::cout << "=====================\n";

    test_specials();
    test_add_and_encode();
    test_whitespace_normalisation();
    test_morphological_analyses();
    test_decode_out_of_range();
    test_build_from_corpus();
    test_save_load_roundtrip();
    test_freeze_flag();
    test_print_stats();

    std::cout << "\n=====================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
