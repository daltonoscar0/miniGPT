#pragma once

// ============================================================
//  Vocab.hpp  —  Vocabulary builder for miniGPT
//  Maps morphological analysis strings to integer token IDs.
//
//  Design notes
//  ────────────
//  Instead of raw subword tokens (BPE/WordPiece), we use the
//  output of Scalpel + Morphology as our vocabulary atoms.
//  Each "type" is a canonical analysis string, e.g.:
//
//    "cat+N+SG"   "run+V+INF"   "quick+ADJ"   "<UNK>"
//
//  A token ID is just a 32-bit integer index into the ordered
//  vocabulary list.  The Vocab object owns the bidirectional
//  maps: string→id and id→string.
//
//  Special tokens
//  ──────────────
//  ID 0  →  <PAD>   padding (zero embedding, masked in attn)
//  ID 1  →  <UNK>   unknown analysis
//  ID 2  →  <BOS>   beginning of sequence
//  ID 3  →  <EOS>   end of sequence
//
//  The first call to add() with any of these strings returns
//  the reserved ID rather than allocating a new one.
// ============================================================

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <iostream>

namespace miniGPT {

// ── Token ID type ─────────────────────────────────────────
using TokenID = uint32_t;

// ── Reserved special tokens ────────────────────────────────
constexpr std::string_view kPAD = "<PAD>";
constexpr std::string_view kUNK = "<UNK>";
constexpr std::string_view kBOS = "<BOS>";
constexpr std::string_view kEOS = "<EOS>";

constexpr TokenID kPAD_ID = 0;
constexpr TokenID kUNK_ID = 1;
constexpr TokenID kBOS_ID = 2;
constexpr TokenID kEOS_ID = 3;

// ── Vocab class ────────────────────────────────────────────
class Vocab {
public:
    // ── Construction ──────────────────────────────────────

    // Build an empty vocab pre-seeded with the four specials.
    Vocab() { init_specials(); }

    // ── Core mutating API ─────────────────────────────────

    // Add a canonical analysis string and return its ID.
    // If the type already exists, returns the existing ID.
    // Strings matching a special token return the reserved ID.
    TokenID add(std::string_view analysis) {
        // Normalise: strip leading/trailing whitespace
        std::string key = normalise(analysis);

        auto it = str_to_id_.find(key);
        if (it != str_to_id_.end()) return it->second;

        TokenID new_id = static_cast<TokenID>(id_to_str_.size());
        str_to_id_.emplace(key, new_id);
        id_to_str_.push_back(key);
        return new_id;
    }

    // Convenience: add every analysis in a container.
    template <typename Range>
    void add_all(const Range& analyses) {
        for (const auto& a : analyses) add(a);
    }

    // ── Query API ─────────────────────────────────────────

    // Convert a string to its ID, returning kUNK_ID if absent.
    [[nodiscard]] TokenID encode(std::string_view analysis) const {
        auto it = str_to_id_.find(normalise(analysis));
        return (it != str_to_id_.end()) ? it->second : kUNK_ID;
    }

    // Convert an ID to its string; throws if out of range.
    [[nodiscard]] const std::string& decode(TokenID id) const {
        if (id >= id_to_str_.size())
            throw std::out_of_range("TokenID " + std::to_string(id) +
                                    " out of vocab range " +
                                    std::to_string(id_to_str_.size()));
        return id_to_str_[id];
    }

    [[nodiscard]] bool contains(std::string_view analysis) const {
        return str_to_id_.count(normalise(analysis)) > 0;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return id_to_str_.size();
    }

    // True once the vocab is frozen (no more add() calls intended).
    // Nothing enforces this at runtime; it's advisory for the caller.
    void freeze() noexcept { frozen_ = true; }
    [[nodiscard]] bool is_frozen() const noexcept { return frozen_; }

    // ── Persistence ───────────────────────────────────────

    // Save vocab to a plain-text file: one "ID\tanalysis" per line.
    void save(const std::string& path) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Cannot write vocab to: " + path);
        for (TokenID id = 0; id < id_to_str_.size(); ++id)
            f << id << '\t' << id_to_str_[id] << '\n';
    }

    // Load vocab from a file written by save().  Resets current state.
    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open vocab file: " + path);

        str_to_id_.clear();
        id_to_str_.clear();

        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto tab = line.find('\t');
            if (tab == std::string::npos)
                throw std::runtime_error("Malformed vocab line: " + line);
            TokenID id;
            try {
                unsigned long parsed = std::stoul(line.substr(0, tab));
                if (parsed > std::numeric_limits<TokenID>::max())
                    throw std::out_of_range("id too large");
                id = static_cast<TokenID>(parsed);
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid token ID in vocab line: " + line);
            }
            std::string sym = line.substr(tab + 1);
            if (id != id_to_str_.size())
                throw std::runtime_error("Non-sequential ID in vocab file");
            str_to_id_.emplace(sym, id);
            id_to_str_.push_back(std::move(sym));
        }
    }

    // ── Build from a corpus of analyses ───────────────────

    // Given a stream of analysis strings (one per token in the
    // training corpus), build a vocab sorted by frequency, keeping
    // only the top-k most frequent types (0 = keep all).
    //
    // Usage:
    //   Vocab v;
    //   v.build_from_corpus(all_analyses, /*max_vocab=*/8000);
    static Vocab build_from_corpus(
            const std::vector<std::string>& analyses,
            std::size_t max_vocab = 0)
    {
        // Count frequencies (skip specials — they're pre-inserted)
        std::unordered_map<std::string, uint64_t> freq;
        for (const auto& a : analyses) {
            std::string key = normalise_s(a);
            if (key == kPAD || key == kUNK || key == kBOS || key == kEOS)
                continue;
            freq[key]++;
        }

        // Sort descending by frequency
        std::vector<std::pair<std::string, uint64_t>> sorted(freq.begin(), freq.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b){ return a.second > b.second; });

        if (max_vocab > 0 && sorted.size() > max_vocab - 4 /*specials*/)
            sorted.resize(max_vocab - 4);

        Vocab v;
        for (auto& [sym, _] : sorted)
            v.add(sym);
        return v;
    }

    // ── Debug ─────────────────────────────────────────────
    void print_stats(std::ostream& os = std::cout) const {
        os << "Vocab size: " << size() << " types\n"
           << "  [" << kPAD_ID << "] " << id_to_str_[kPAD_ID] << "\n"
           << "  [" << kUNK_ID << "] " << id_to_str_[kUNK_ID] << "\n"
           << "  [" << kBOS_ID << "] " << id_to_str_[kBOS_ID] << "\n"
           << "  [" << kEOS_ID << "] " << id_to_str_[kEOS_ID] << "\n";
        if (size() > 8) {
            os << "  ... (first non-special entries)\n";
            for (TokenID id = 4; id < std::min<TokenID>(12, size()); ++id)
                os << "  [" << id << "] " << id_to_str_[id] << "\n";
        }
    }

private:
    // ── Implementation ────────────────────────────────────

    std::unordered_map<std::string, TokenID> str_to_id_;
    std::vector<std::string>                 id_to_str_;
    bool                                     frozen_ = false;

    void init_specials() {
        // Insert in exact reserved-ID order
        for (std::string_view sv : {kPAD, kUNK, kBOS, kEOS}) {
            std::string s(sv);
            str_to_id_.emplace(s, static_cast<TokenID>(id_to_str_.size()));
            id_to_str_.push_back(s);
        }
    }

    // Instance method (forward to static for ADL)
    static std::string normalise_s(std::string_view sv) {
        std::string s(sv);
        // Strip leading whitespace
        s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            [](unsigned char c){ return !std::isspace(c); }));
        // Strip trailing whitespace
        s.erase(std::find_if(s.rbegin(), s.rend(),
            [](unsigned char c){ return !std::isspace(c); }).base(), s.end());
        return s;
    }

    static std::string normalise(std::string_view sv) {
        return normalise_s(sv);
    }
};

} // namespace miniGPT
