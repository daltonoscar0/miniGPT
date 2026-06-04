# ============================================================
#  miniGPT/Makefile
# ============================================================

CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2
TESTDIR  := tests

SCALPEL_DIR    := ../Scalpel
MORPHOLOGY_DIR := ../morphology

SCALPEL_SRCS := \
    $(SCALPEL_DIR)/tokenizer.cpp \
    $(SCALPEL_DIR)/char_classes.cpp \
    $(SCALPEL_DIR)/sentence_segmenter.cpp

MORPH_SRCS := \
    $(MORPHOLOGY_DIR)/PIPELINE/text_pipeline.cpp \
    $(MORPHOLOGY_DIR)/SYMBOLS/symbol.cpp

UPSTREAM_INC := -I$(SCALPEL_DIR) -I$(MORPHOLOGY_DIR)

TESTS_CORE     := vocab_test encoder_test embed_test attn_test mha_test ffn_test layernorm_test block_test lmhead_test train_test infer_test
TESTS_PIPELINE := pipeline_test

.PHONY: all test test_pipeline run clean

all: $(TESTS_CORE)

test: $(TESTS_CORE)
	@echo ""
	@echo "══════════════════════════════"
	@echo " Running miniGPT core tests"
	@echo "══════════════════════════════"
	@for t in $(TESTS_CORE); do \
		echo ""; \
		echo "▸ $$t"; \
		./$$t || exit 1; \
	done
	@echo ""
	@echo "══ All core test suites passed ══"

test_pipeline: $(TESTS_PIPELINE)
	./pipeline_test

vocab_test: $(TESTDIR)/vocab_test.cpp vocab/Vocab.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/vocab_test.cpp

encoder_test: $(TESTDIR)/encoder_test.cpp encoder/Encoder.hpp vocab/Vocab.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/encoder_test.cpp

embed_test: $(TESTDIR)/embed_test.cpp embed/Embedding.hpp vocab/Vocab.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/embed_test.cpp

attn_test: $(TESTDIR)/attn_test.cpp attention/Attention.hpp embed/Embedding.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/attn_test.cpp

mha_test: $(TESTDIR)/mha_test.cpp attention/MultiHeadAttention.hpp attention/Attention.hpp embed/Embedding.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/mha_test.cpp

ffn_test: $(TESTDIR)/ffn_test.cpp ffn/FeedForward.hpp embed/Embedding.hpp attention/Attention.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/ffn_test.cpp

layernorm_test: $(TESTDIR)/layernorm_test.cpp norm/LayerNorm.hpp embed/Embedding.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/layernorm_test.cpp

block_test: $(TESTDIR)/block_test.cpp block/TransformerBlock.hpp \
            attention/MultiHeadAttention.hpp ffn/FeedForward.hpp norm/LayerNorm.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/block_test.cpp

lmhead_test: $(TESTDIR)/lmhead_test.cpp lm_head/LMHead.hpp embed/Embedding.hpp attention/Attention.hpp vocab/Vocab.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/lmhead_test.cpp

train_test: $(TESTDIR)/train_test.cpp train/AdamW.hpp train/Trainer.hpp model/MiniGPT.hpp \
            block/TransformerBlock.hpp lm_head/LMHead.hpp embed/Embedding.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/train_test.cpp

infer_test: $(TESTDIR)/infer_test.cpp infer/Inference.hpp model/MiniGPT.hpp \
            train/Trainer.hpp train/AdamW.hpp lm_head/LMHead.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/infer_test.cpp

pipeline_test: $(TESTDIR)/pipeline_test.cpp pipeline/TextPipeline.hpp \
               encoder/Encoder.hpp vocab/Vocab.hpp
	$(CXX) $(CXXFLAGS) $(UPSTREAM_INC) -o $@ \
	    $(TESTDIR)/pipeline_test.cpp \
	    $(SCALPEL_SRCS) $(MORPH_SRCS)

miniGPT: main.cpp $(wildcard */*.hpp)
	$(CXX) $(CXXFLAGS) $(UPSTREAM_INC) -o $@ main.cpp $(SCALPEL_SRCS) $(MORPH_SRCS)

run: miniGPT
	./miniGPT

clean:
	rm -f $(TESTS_CORE) $(TESTS_PIPELINE) miniGPT
	rm -f /tmp/test_vocab.tsv /tmp/test_embeddings.bin
