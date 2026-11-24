# miniGPT

[![tests](https://github.com/daltonoscar0/miniGPT/actions/workflows/tests.yml/badge.svg)](https://github.com/daltonoscar0/miniGPT/actions/workflows/tests.yml)

A GPT-style transformer language model built from scratch in C++17, no
PyTorch, no BLAS, no autograd. Every component, including the full backward
pass, is implemented and tested by hand.

What makes this model unusual is its vocabulary: instead of byte-pair-encoded
subword tokens, it uses **morphological analyses** as its atoms. Words are run
through a finite-state tokenizer ([Scalpel](#dependencies)) and a morphological
analyzer ([Morphology](#dependencies)) that turn `cats` into `cat+N+PL` and
`ran` into `run+V+PAST`. The model therefore learns over a linguistically
structured vocabulary rather than raw text fragments.

## Architecture

```
Token IDs
  -> Embedding (token + sinusoidal positional encoding)
  -> Transformer block x 2  (pre-norm: LN -> MHA -> residual -> LN -> FFN -> residual)
  -> Final LayerNorm
  -> LM head (linear + softmax, weights tied to embedding)
  -> probabilities
```

Default configuration is small by design, 2 layers, 4 heads, d_model=128,
~1M parameters, so the whole thing trains on a CPU and stays easy to reason
about.

## Components

Each component lives in its own directory with a matching test file under
`tests/`:

| Directory     | Component                                    |
|---------------|----------------------------------------------|
| `vocab/`      | Vocabulary builder (analysis string -> ID)   |
| `encoder/`    | Sequence encoder (BOS/EOS/padding/masking)   |
| `embed/`      | Token embeddings + positional encodings      |
| `attention/`  | Scaled dot-product + multi-head attention    |
| `ffn/`        | Feedforward sublayer (linear -> ReLU -> linear) |
| `norm/`       | Layer normalization                          |
| `block/`      | Transformer block (attn + FFN + residuals)   |
| `lm_head/`    | LM head (projection + softmax + cross-entropy) |
| `model/`      | Full model assembly                          |
| `train/`      | AdamW optimizer + training loop              |
| `infer/`      | Greedy / top-k decoding + perplexity         |
| `pipeline/`   | Scalpel + Morphology integration layer       |

## Dependencies

This project consumes two sibling libraries that must be cloned next to it:

```
parent_dir/
  Scalpel/      FSM tokenizer and sentence segmenter
  morphology/   Finite-state morphological analyzer
  miniGPT/      this repository
```

Only the pipeline integration (and the demo in `main.cpp`) requires these.
The core model components have no external dependencies and can be built and
tested on their own.

## Building and testing

```sh
make test            # build and run all component unit tests
make test_pipeline   # build and run the Scalpel + Morphology integration test
make run             # build and run the demo in main.cpp
```

The core test suite has no dependencies beyond a C++17 compiler. The pipeline
test and the demo require Scalpel and Morphology as described above.

## Running the demo

The demo in `main.cpp` reads plain text from `training.txt`, builds a
vocabulary, trains the model, and generates sample continuations.

```sh
# Download any plain-text corpus, e.g. from Project Gutenberg:
curl -o training.txt "https://www.gutenberg.org/files/74/74-0.txt"
make run
```

Morphology loads its lexicon from `data/english_lexicon.tsv` relative to the
working directory. Point a symlink at the lexicon shipped with the Morphology
repo:

```sh
ln -s ../morphology/data data
```

## Design notes

- **Tensors** are a single `Matrix2D` type: a row-major `std::vector<float>`
  with shape annotations in comments (`// [seq_len x d_model]`).
- **Manual backprop**, every component implements its own `backward()` and is
  verified with finite-difference gradient checks where applicable.
- **Pre-norm** transformer blocks, following GPT-2, for stable training from
  scratch.
- **Weight tying** between the embedding table and the LM head, reducing
  parameters and improving sample quality.

## License

MIT, see [LICENSE](LICENSE).
