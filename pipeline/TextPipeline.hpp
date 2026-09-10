#pragma once

// ============================================================
//  TextPipeline.hpp, Scalpel + Morphology integration layer
//
//  This is the only file in miniGPT that touches the upstream
//  libraries.  Everything downstream (Encoder, Vocab, training
//  loop) works purely with std::string analysis keys and
//  TokenIDs, they never see a Token struct or an Analysis.
//
//  Pipeline
//  ────────
//  Raw string
//    │  Scalpel (via Morphology's text_pipeline.h)
//    ▼
//  vector<vector<string>>   <- sentences of surface word strings
//    │  Morphology Analyzer::analyze() per word
//    ▼
//  vector<vector<string>>   <- sentences of canonical analysis keys
//
//  Canonicalisation rules
//  ──────────────────────
//  Morphology's Analysis::to_string() can produce several
//  formats.  We normalise them all to a single key style so
//  the Vocab sees consistent atoms:
//
//    Inflected:    "cat +N +PL"        -> "cat+N+PL"
//    Base form:    "run +V"            -> "run+V"
//    Prefixed:     "un- + kind +ADJ"   -> "un+kind+ADJ"
//    Derivational: "dark +ADJ [-ness ->N]" -> "dark+ADJ+ness>N"
//
//  Non-word tokens (PUNCT, NUMBER, ABBREVIATION, CONTRACTION,
//  HYPHENATED) pass through as surface-form tags:
//
//    "."   -> "<PUNCT:.>"
//    "42"  -> "<NUM:42>"
//    "Dr." -> "<ABBR:Dr.>"
//    "can't" -> "<CONTR:can't>"
//    "state-of-the-art" -> "<HYPH:state-of-the-art>"
//
//  Ambiguity
//  ─────────
//  Morphology returns analyses sorted by weight descending.
//  We take analyses[0], the highest-weight (most likely)
//  reading.  Words with no analysis fall back to:
//    "<WORD:surface>"
//
//  Usage
//  ─────
//  TextPipeline pipeline;
//  auto sentences = pipeline.process("The cats sat.");
//  // sentences[0] == {"the+DET", "cat+N+PL", "sit+V+PAST", "<PUNCT:.>"}
// ============================================================

// ── Upstream includes ─────────────────────────────────────
// Adjust these paths if your directory layout differs.
// Expected sibling layout:
//   parent_dir/
//     Scalpel/
//     morphology/
//     miniGPT/

#include "../../morphology/PIPELINE/text_pipeline.h"
#include "../../morphology/SYMBOLS/symbol.h"
#include "../../morphology/LEXICON/lexicon_fsa.h"
#include "../../morphology/ANALYSIS/analyzer.h"
#include "../../morphology/OUTPUT/pretty_print.h"
#include "../../morphology/LANGUAGES/english_configuration.h"

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>

namespace miniGPT {

// ── Canonicalise an Analysis::to_string() output ──────────
//
// Input examples:
//   "cat +N +PL"
//   "run +V"
//   "un- + kind +ADJ"
//   "dark +ADJ [-ness ->N]"
//   "walk +V +PAST"
//
// Output examples:
//   "cat+N+PL"
//   "run+V"
//   "un+kind+ADJ"
//   "dark+ADJ+ness>N"
//   "walk+V+PAST"
inline std::string canonicalise(const std::string& raw) {
    // Work on a copy, stripping spaces around '+' and '->'
    std::string s = raw;

    // 1. Handle prefixed form: "un- + kind +ADJ"
    //    Detect "- + " pattern -> replace with "+"
    {
        auto pos = s.find("- + ");
        if (pos != std::string::npos) {
            s.replace(pos, 4, "+");
        }
    }

    // 2. Handle derivational: "dark +ADJ [-ness ->N]"
    //    Turn "[-ness ->N]" into "+ness>N"
    {
        auto lb = s.find(" [-");
        if (lb != std::string::npos) {
            // Find closing ']'
            auto rb = s.find(']', lb);
            std::string deriv = s.substr(lb + 3, rb - lb - 3); // e.g. "ness ->N"
            // Replace " ->" with ">"
            auto arrow = deriv.find(" ->");
            if (arrow != std::string::npos)
                deriv.replace(arrow, 3, ">");
            s = s.substr(0, lb) + "+" + deriv;
        }
    }

    // 3. Remove all spaces around '+' signs
    //    "cat +N +PL" -> "cat+N+PL"
    std::string result;
    result.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == ' ') {
            // Skip spaces before/after '+'
            if ((i + 1 < s.size() && s[i+1] == '+') ||
                (i > 0           && s[i-1] == '+'))
                continue;
            // Also skip lone spaces (shouldn't be any left)
            continue;
        }
        result += s[i];
    }

    // 4. Strip any leading '+' (can appear if prefix handling left one)
    while (!result.empty() && result.front() == '+')
        result.erase(result.begin());

    return result;
}

// ── TextPipeline class ────────────────────────────────────
class TextPipeline {
public:
    // ── Construction ──────────────────────────────────────
    // Initialises Morphology's full English configuration.
    // This is somewhat expensive (~1825 lexemes loaded) so
    // create one instance and reuse it.
    TextPipeline()
        : lexicon_(&symbols_), analyzer_(&symbols_, &lexicon_)
    {
        EnglishConfig::initialize(symbols_, lexicon_, analyzer_);
    }

    // Non-copyable (owns non-copyable Morphology objects)
    TextPipeline(const TextPipeline&)            = delete;
    TextPipeline& operator=(const TextPipeline&) = delete;

    // ── Primary API ───────────────────────────────────────

    // Process raw text through Scalpel + Morphology.
    // Returns one vector<string> per sentence, where each
    // string is a canonical analysis key ready for Vocab lookup.
    //
    // @param text  Any raw English text (multiple sentences OK)
    // @returns     Sentences of canonical analysis strings
    [[nodiscard]] std::vector<std::vector<std::string>>
    process(const std::string& text) const
    {
        // Phase 1: Scalpel tokenises + segments via Morphology bridge
        // Returns vector<vector<string>>, sentences of surface words
        auto sentences = segment_into_sentences(text);

        std::vector<std::vector<std::string>> result;
        result.reserve(sentences.size());

        for (const auto& words : sentences) {
            std::vector<std::string> analyses;
            analyses.reserve(words.size());

            for (const auto& word : words) {
                analyses.push_back(analyse_token(word));
            }

            if (!analyses.empty())
                result.push_back(std::move(analyses));
        }

        return result;
    }

    // Convenience: process text and flatten all sentences into
    // one sequence (useful for building the training corpus).
    [[nodiscard]] std::vector<std::string>
    process_flat(const std::string& text) const
    {
        auto sentences = process(text);
        std::vector<std::string> flat;
        for (auto& s : sentences)
            for (auto& tok : s)
                flat.push_back(std::move(tok));
        return flat;
    }

    // ── Stats ─────────────────────────────────────────────
    std::size_t lexeme_count() const { return lexicon_.num_lexemes(); }

private:
    // ── Morphology objects ────────────────────────────────
    // Initialisation order matters: symbols_ must outlive everything.
    SymbolTable symbols_;
    LexiconFSA  lexicon_;
    mutable Analyzer analyzer_;  // mutable: analyze() is not const in Morphology

    // ── Analyse a single surface token ────────────────────
    // Called once per token from process().
    // Returns a canonical analysis key string.
    std::string analyse_token(const std::string& word) const {
        // Non-alphabetic surface forms get tagged directly
        // (Morphology's text_pipeline already filters punctuation
        //  but may still pass through numbers/contractions/etc.)
        if (word.empty()) return "<UNK>";

        // Check if purely non-alpha (punctuation / number)
        bool all_alpha = std::all_of(word.begin(), word.end(),
            [](unsigned char c){ return std::isalpha(c) || c == '\''; });

        if (!all_alpha) {
            // Punctuation / number / symbol
            bool all_digit = std::all_of(word.begin(), word.end(),
                [](unsigned char c){ return std::isdigit(c); });
            if (all_digit)
                return "<NUM:" + word + ">";
            // Treat as punctuation
            return "<PUNCT:" + word + ">";
        }

        // Contraction heuristic (contains apostrophe)
        if (word.find('\'') != std::string::npos)
            return "<CONTR:" + word + ">";

        // Run morphological analysis
        auto analyses = analyzer_.analyze(word);

        if (analyses.empty()) {
            // No analysis found, fall back to lowercased surface form
            std::string lower = word;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                [](unsigned char c){ return std::tolower(c); });
            return "<WORD:" + lower + ">";
        }

        // Take the highest-weight (first) analysis and canonicalise
        return canonicalise(analyses[0].to_string());
    }
};

} // namespace miniGPT
