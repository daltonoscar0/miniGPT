#pragma once

// ============================================================
//  Encoder.hpp, Text -> TokenID sequence converter
//
//  Takes a sequence of morphological analysis strings (already
//  produced by the TextPipeline or supplied directly) and maps
//  them to a fixed-length vector<TokenID> suitable for feeding
//  into the Embedding table.
//
//  Responsibilities
//  ────────────────
//  1. Wrap the sequence with <BOS> and <EOS>
//  2. Look up each analysis string in the Vocab
//     (unknown -> kUNK_ID)
//  3. Truncate to max_len if too long  (tail-truncation:
//     keep the most recent tokens so context is preserved)
//  4. Pad to exactly max_len with <PAD> if too short
//
//  Output shape: vector<TokenID> of length exactly max_len
//                // [max_len]
//
//  The encoder also produces a boolean attention mask of the
//  same length:
//    mask[i] = true  -> real token (attend to it)
//    mask[i] = false -> padding    (ignore in attention)
//
//  Truncation policy
//  ─────────────────
//  For a language model, losing the *end* of a sequence
//  discards the most recently seen context.  We therefore
//  perform TAIL truncation: keep the last (max_len - 2) content
//  tokens so that <EOS> always appears and the model sees the
//  most recent context.
//
//    max_len = 6, analyses = [A B C D E F G H]
//    -> keep last 4: [E F G H]
//    -> wrap:        [<BOS> E F G H <EOS>]    length = 6 
//
//  Encoding a batch
//  ────────────────
//  Call encode() once per sentence.  The training loop
//  stacks the resulting vectors into a matrix row-by-row.
// ============================================================

#include "../vocab/Vocab.hpp"

#include <vector>
#include <string>
#include <string_view>
#include <stdexcept>

namespace miniGPT {

// ── EncodedSequence ───────────────────────────────────────
// Plain struct so the training loop can destructure easily.
struct EncodedSequence {
    std::vector<TokenID> ids;   // [max_len], always exactly max_len long
    std::vector<bool>    mask;  // [max_len], true = real token
    std::size_t          n_real; // number of non-padding tokens
};

// ── Encoder class ─────────────────────────────────────────
class Encoder {
public:
    // ── Construction ──────────────────────────────────────

    // max_len: fixed sequence length (including BOS + EOS).
    // Must be >= 2 (at minimum one BOS + one EOS).
    explicit Encoder(const Vocab& vocab, std::size_t max_len = 64)
        : vocab_(vocab), max_len_(max_len)
    {
        if (max_len_ < 2)
            throw std::invalid_argument(
                "Encoder max_len must be >= 2 (need room for BOS + EOS)");
    }

    // ── Primary API ───────────────────────────────────────

    // Encode a sequence of analysis strings into a fixed-length
    // TokenID vector with attention mask.
    //
    // @param analyses  Morphological analysis strings, e.g.
    //                  {"cat+N+SG", "run+V+INF"}
    // @returns         EncodedSequence of length exactly max_len_
    [[nodiscard]] EncodedSequence encode(
            const std::vector<std::string>& analyses) const
    {
        // How many content slots are available after BOS + EOS
        const std::size_t content_cap = max_len_ - 2;

        // Tail-truncate: take the last content_cap analyses
        std::size_t start = 0;
        if (analyses.size() > content_cap)
            start = analyses.size() - content_cap;

        // Build the raw id sequence: BOS + content + EOS
        std::vector<TokenID> ids;
        ids.reserve(max_len_);

        ids.push_back(kBOS_ID);                        // position 0

        for (std::size_t i = start; i < analyses.size(); ++i)
            ids.push_back(vocab_.encode(analyses[i])); // positions 1..n

        ids.push_back(kEOS_ID);                        // position n+1

        // Number of real tokens before padding
        const std::size_t n_real = ids.size();         // <= max_len_

        // Pad to exactly max_len_
        while (ids.size() < max_len_)
            ids.push_back(kPAD_ID);

        // Build mask: true for real tokens, false for padding
        std::vector<bool> mask(max_len_, false);
        for (std::size_t i = 0; i < n_real; ++i)
            mask[i] = true;

        return EncodedSequence{std::move(ids), std::move(mask), n_real};
    }

    // Convenience overload: encode a single analysis string
    // (wraps in a one-element vector).
    [[nodiscard]] EncodedSequence encode_one(std::string_view analysis) const {
        return encode({std::string(analysis)});
    }

    // ── Training helpers ──────────────────────────────────

    // For language-model training we need (input, target) pairs
    // where target[i] = input[i+1] (next-token prediction).
    //
    // Returns two EncodedSequences:
    //   input:  [<BOS> t0 t1 ... t_{n-1} <PAD>...]
    //   target: [t0    t1 ... t_{n-1} <EOS> <PAD>...]
    //
    // Both are length max_len_.  Loss is computed only where
    // target mask is true.
    struct LMPair {
        EncodedSequence input;
        EncodedSequence target;
    };

    [[nodiscard]] LMPair encode_lm_pair(
            const std::vector<std::string>& analyses) const
    {
        const std::size_t content_cap = max_len_ - 2;

        std::size_t start = 0;
        if (analyses.size() > content_cap)
            start = analyses.size() - content_cap;

        // Collect content token IDs (after truncation)
        std::vector<TokenID> content;
        content.reserve(content_cap);
        for (std::size_t i = start; i < analyses.size(); ++i)
            content.push_back(vocab_.encode(analyses[i]));

        // input  = [BOS, c0, c1, ..., c_{n-1}, PAD, PAD, ...]
        // target = [c0,  c1, ..., c_{n-1}, EOS, PAD, PAD, ...]
        std::vector<TokenID> in_ids, tgt_ids;
        std::vector<bool>    in_mask, tgt_mask;
        in_ids.reserve(max_len_);
        tgt_ids.reserve(max_len_);
        in_mask.reserve(max_len_);
        tgt_mask.reserve(max_len_);

        // Position 0: BOS in input, c0 in target
        in_ids.push_back(kBOS_ID);
        in_mask.push_back(true);

        if (!content.empty()) {
            tgt_ids.push_back(content[0]);
            tgt_mask.push_back(true);
        } else {
            tgt_ids.push_back(kEOS_ID);
            tgt_mask.push_back(true);
        }

        // Positions 1..n-1
        for (std::size_t i = 0; i + 1 < content.size(); ++i) {
            in_ids.push_back(content[i]);
            in_mask.push_back(true);
            tgt_ids.push_back(content[i + 1]);
            tgt_mask.push_back(true);
        }

        // Position n: last content token in input, EOS in target
        if (!content.empty()) {
            in_ids.push_back(content.back());
            in_mask.push_back(true);
            tgt_ids.push_back(kEOS_ID);
            tgt_mask.push_back(true);
        }

        std::size_t n_real_in  = in_ids.size();
        std::size_t n_real_tgt = tgt_ids.size();

        // Pad both to max_len_
        while (in_ids.size()  < max_len_) { in_ids.push_back(kPAD_ID);  in_mask.push_back(false);  }
        while (tgt_ids.size() < max_len_) { tgt_ids.push_back(kPAD_ID); tgt_mask.push_back(false); }

        EncodedSequence in_seq  {std::move(in_ids),  std::move(in_mask),  n_real_in};
        EncodedSequence tgt_seq {std::move(tgt_ids), std::move(tgt_mask), n_real_tgt};

        return LMPair{std::move(in_seq), std::move(tgt_seq)};
    }

    // ── Accessors ─────────────────────────────────────────
    [[nodiscard]] std::size_t max_len()   const noexcept { return max_len_; }
    [[nodiscard]] const Vocab& vocab()    const noexcept { return vocab_; }

private:
    const Vocab& vocab_;
    std::size_t  max_len_;
};

} // namespace miniGPT
