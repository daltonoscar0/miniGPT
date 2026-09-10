// pipeline_test.cpp, Integration test for TextPipeline + Encoder
//
// Requires Scalpel and Morphology compiled in.
// Build and run from miniGPT/:
//   make pipeline_test && ./pipeline_test

#include "../pipeline/TextPipeline.hpp"
#include "../encoder/Encoder.hpp"
#include "../vocab/Vocab.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

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

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

bool any_contains(const std::vector<std::string>& v, const std::string& sub) {
    return std::any_of(v.begin(), v.end(),
        [&](const std::string& s){ return s.find(sub) != std::string::npos; });
}

// ── Tests ─────────────────────────────────────────────────

void test_canonicalise() {
    SECTION("canonicalise() helper");

    // Inflected noun
    CHECK(canonicalise("cat +N +PL")     == "cat+N+PL");
    // Base verb
    CHECK(canonicalise("run +V")         == "run+V");
    // Prefixed
    CHECK(canonicalise("un- + kind +ADJ") == "un+kind+ADJ");
    // Derivational
    CHECK(canonicalise("dark +ADJ [-ness ->N]") == "dark+ADJ+ness>N");
    // Past tense
    CHECK(canonicalise("walk +V +PAST")  == "walk+V+PAST");
    // Comparative
    CHECK(canonicalise("tall +ADJ +COMP") == "tall+ADJ+COMP");
}

void test_process_simple(TextPipeline& pipeline) {
    SECTION("process() — simple sentence");

    auto sentences = pipeline.process("The cats sat.");

    // Should produce exactly one sentence
    CHECK(sentences.size() == 1);
    if (sentences.empty()) return;

    auto& s = sentences[0];

    // "cats" -> "cat+N+PL"
    CHECK(any_contains(s, "cat+N+PL"));

    // "sat" -> "sit+V+PAST"  (irregular: sit -> sat)
    CHECK(any_contains(s, "sit+V+PAST"));

    // Print what we got for visibility
    std::cout << "  Tokens: ";
    for (const auto& t : s) std::cout << "[" << t << "] ";
    std::cout << "\n";
}

void test_process_two_sentences(TextPipeline& pipeline) {
    SECTION("process() — two sentences");

    auto sentences = pipeline.process("Dogs run. Cats sleep.");

    CHECK(sentences.size() == 2);
    if (sentences.size() < 2) return;

    CHECK(any_contains(sentences[0], "dog+N+PL"));
    CHECK(any_contains(sentences[1], "cat+N+PL"));
}

void test_process_morphological_variety(TextPipeline& pipeline) {
    SECTION("process() — morphological variety");

    // Covers: irregular verb, Y_FINAL noun, derivational suffix
    auto flat = pipeline.process_flat("mice walk quickly");

    // mice -> mouse+N+PL  (irregular plural)
    CHECK(any_contains(flat, "mouse+N+PL"));

    // walk -> walk+V  (base form, no suffix)
    CHECK(any_contains(flat, "walk+V"));

    // quickly -> quick+ADJ+ly>ADV  (derivational)
    CHECK(any_contains(flat, "quick"));

    std::cout << "  Tokens: ";
    for (const auto& t : flat) std::cout << "[" << t << "] ";
    std::cout << "\n";
}

void test_process_flat(TextPipeline& pipeline) {
    SECTION("process_flat() — multi-sentence flattening");

    auto flat = pipeline.process_flat("cats run. dogs walk.");
    CHECK(!flat.empty());
    CHECK(any_contains(flat, "cat+N+PL"));
    CHECK(any_contains(flat, "dog+N+PL"));
}

void test_pipeline_to_encoder(TextPipeline& pipeline) {
    SECTION("TextPipeline → Vocab → Encoder full chain");

    // Build vocab from a small corpus
    auto corpus = pipeline.process_flat(
        "the cat sat on the mat. the dog ran fast.");

    Vocab vocab = Vocab::build_from_corpus(corpus);
    vocab.freeze();

    std::cout << "  Vocab size: " << vocab.size() << "\n";
    CHECK(vocab.size() > 4); // more than just specials

    // Encode a fresh sentence through the full chain
    auto analyses = pipeline.process_flat("the cat ran.");
    Encoder enc(vocab, /*max_len=*/16);
    auto seq = enc.encode(analyses);

    CHECK(seq.ids.size() == 16);
    CHECK(seq.ids[0] == kBOS_ID);
    CHECK(seq.ids[seq.n_real - 1] == kEOS_ID);
    CHECK(seq.n_real >= 2); // at least BOS + EOS

    // All non-padding slots should have a valid token ID
    for (std::size_t i = 0; i < seq.n_real; ++i)
        CHECK(seq.ids[i] < vocab.size());

    std::cout << "  Encoded (" << seq.n_real << " real tokens): ";
    for (std::size_t i = 0; i < seq.n_real; ++i)
        std::cout << "[" << vocab.decode(seq.ids[i]) << "] ";
    std::cout << "\n";
}

void test_lm_pair_chain(TextPipeline& pipeline) {
    SECTION("LM pair from pipeline — input/target alignment");

    auto analyses = pipeline.process_flat("cats run quickly");
    Vocab vocab = Vocab::build_from_corpus(analyses);

    Encoder enc(vocab, /*max_len=*/12);
    auto pair = enc.encode_lm_pair(analyses);

    auto& in  = pair.input;
    auto& tgt = pair.target;

    CHECK(in.ids.size()  == 12);
    CHECK(tgt.ids.size() == 12);
    CHECK(in.ids[0] == kBOS_ID);
    CHECK(tgt.ids[in.n_real - 1] == kEOS_ID);

    // input[i+1] == target[i] for all real positions
    for (std::size_t i = 0; i + 1 < in.n_real; ++i)
        CHECK(in.ids[i + 1] == tgt.ids[i]);
}

// ── main ──────────────────────────────────────────────────
int main() {
    std::cout << "miniGPT — Pipeline integration tests\n";
    std::cout << "=====================================\n";

    // Initialise once, this loads the full English lexicon
    std::cout << "\nInitialising TextPipeline (loading English lexicon)...\n";
    TextPipeline pipeline;
    std::cout << "Loaded " << pipeline.lexeme_count() << " lexemes.\n";

    test_canonicalise();
    test_process_simple(pipeline);
    test_process_two_sentences(pipeline);
    test_process_morphological_variety(pipeline);
    test_process_flat(pipeline);
    test_pipeline_to_encoder(pipeline);
    test_lm_pair_chain(pipeline);

    std::cout << "\n=====================================\n";
    std::cout << passed << " / " << total << " tests passed";
    if (passed == total)
        std::cout << "  ✓ ALL PASS\n";
    else
        std::cout << "  ✗ " << (total - passed) << " FAILED\n";

    return (passed == total) ? 0 : 1;
}
