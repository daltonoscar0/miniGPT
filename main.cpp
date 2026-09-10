// ============================================================
//  main.cpp, miniGPT driver program
//
//  Build:
//    make run
//
//  Before running, download training data:
//    curl -o training.txt "https://www.gutenberg.org/files/74/74-0.txt"
// ============================================================

#include "pipeline/TextPipeline.hpp"
#include "vocab/Vocab.hpp"
#include "encoder/Encoder.hpp"
#include "model/MiniGPT.hpp"
#include "train/Trainer.hpp"
#include "infer/Inference.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() {

    // ── 1. Load training corpus ───────────────────────────
    const std::string training_file = "training.txt";
    std::string training_text;
    {
        std::ifstream f(training_file);
        if (!f) {
            std::cerr << "Error: could not open " << training_file << "\n\n";
            std::cerr << "Download a training corpus with:\n";
            std::cerr << "  curl -o training.txt "
                         "\"https://www.gutenberg.org/files/74/74-0.txt\"\n";
            return 1;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        training_text = ss.str();
        std::cout << "Loaded " << training_text.size()
                  << " bytes from " << training_file << "\n";
    }

    // ── 2. Analyse text ───────────────────────────────────
    std::cout << "Analysing text (this may take a moment for large files)...\n";
    miniGPT::TextPipeline pipeline;
    auto corpus = pipeline.process_flat(training_text);

    std::cout << "Corpus: " << corpus.size() << " tokens\n";
    std::cout << "Sample: ";
    for (std::size_t i = 0; i < std::min<std::size_t>(8, corpus.size()); ++i)
        std::cout << "[" << corpus[i] << "] ";
    std::cout << "...\n\n";

    // ── 3. Build vocabulary ───────────────────────────────
    // Keep the top 2000 most frequent types.
    // Increase for larger corpora, decrease for faster training.
    std::cout << "Building vocabulary...\n";
    auto vocab = miniGPT::Vocab::build_from_corpus(corpus, /*max_vocab=*/2000);
    vocab.freeze();
    vocab.print_stats();
    std::cout << "\n";

    // ── 4. Configure model ────────────────────────────────
    miniGPT::ModelConfig cfg;
    cfg.vocab_size  = vocab.size();
    cfg.d_model     = 128;
    cfg.n_heads     = 4;
    cfg.n_layers    = 2;
    cfg.d_ff        = 512;
    cfg.max_len     = 64;
    cfg.seed        = 42;
    cfg.tie_weights = true;

    std::cout << "Building model...\n";
    miniGPT::MiniGPT model(cfg);
    model.print_config();
    std::cout << "\n";

    // ── 5. Set up encoder and trainer ─────────────────────
    miniGPT::Encoder encoder(vocab, cfg.max_len);

    miniGPT::TrainerConfig tcfg;
    tcfg.lr            = 3e-4f;
    tcfg.weight_decay  = 0.01f;
    tcfg.max_grad_norm = 1.0f;
    tcfg.n_epochs      = 1;
    tcfg.log_every     = 1000;

    miniGPT::Trainer trainer(model, encoder, tcfg);

    // ── 6. Perplexity before training ─────────────────────
    miniGPT::Inference infer(model, encoder, vocab);
    {
        // Sample the first 200 tokens for a quick perplexity estimate
        std::vector<std::string> sample(
            corpus.begin(),
            corpus.begin() + std::min<std::size_t>(200, corpus.size()));
        float ppl = infer.perplexity(sample);
        std::cout << "Perplexity before training (sample): " << ppl << "\n\n";
    }

    // ── 7. Train ──────────────────────────────────────────
    std::cout << "Training...\n";
    trainer.train(corpus);
    std::cout << "\n";

    // ── 8. Perplexity after training ──────────────────────
    {
        std::vector<std::string> sample(
            corpus.begin(),
            corpus.begin() + std::min<std::size_t>(200, corpus.size()));
        float ppl = infer.perplexity(sample);
        std::cout << "Perplexity after training (sample): " << ppl << "\n\n";
    }

    // ── 9. Generate from prompts ──────────────────────────
    std::cout << "Generating text...\n\n";

    // Edit these prompts to use tokens that appear in your corpus.
    // Run with a small corpus first to see what tokens were learned,
    // then pick prompts from vocab IDs 4-20 shown during vocab build.
    std::vector<std::vector<std::string>> prompts = {
        {"the+DET"},
        {"<WORD:the>"},        // if "the" fell back to WORD form
        {"<WORD:he>"},
        {"<WORD:she>"},
    };

    miniGPT::GenerationConfig gcfg;
    gcfg.max_new_tokens = 12;
    gcfg.top_k          = 1;    // greedy

    for (const auto& prompt : prompts) {
        // Skip prompts whose first token isn't in vocab
        if (!vocab.contains(prompt[0])) continue;

        std::cout << "Prompt: ";
        for (const auto& t : prompt) std::cout << t << " ";
        std::cout << "\nOutput: ";
        for (const auto& t : prompt) std::cout << t << " ";
        auto generated = infer.generate(prompt, gcfg);
        for (const auto& t : generated) std::cout << t << " ";
        std::cout << "\n\n";
    }

    // ── 10. Top-k sampling ────────────────────────────────
    std::cout << "Top-k sampling (k=10, temperature=0.9):\n";
    gcfg.top_k       = 10;
    gcfg.temperature = 0.9f;
    gcfg.seed        = 42;

    // Use the most common non-special token (ID 4) as prompt
    if (vocab.size() > 4) {
        std::string common = vocab.decode(4);
        std::cout << common << " ";
        auto sampled = infer.generate({common}, gcfg);
        for (const auto& t : sampled) std::cout << t << " ";
        std::cout << "\n";
    }

    return 0;
}
