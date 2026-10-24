# llm_from_scratch

A character-level GPT, implemented in pure Rust with **no ML libraries at
all** — not even a tensor/autograd crate. The only external dependency is
`rand`, used for weight initialization and sampling.

This means every piece of "how neural nets actually work" is visible and
editable:

- **`src/tensor.rs`** — a reverse-mode automatic differentiation engine over
  2D `f32` matrices. Every operation (`matmul`, `add`, `softmax`,
  `layer_norm`, `embedding_lookup`, `cross_entropy`, causal masking, column
  slicing/concatenation for multi-head attention, ...) is implemented by
  hand: forward pass computes the result and records an `Op` on a `Tape`;
  `backward()` walks the tape in reverse and applies the calculus for each
  op to propagate gradients into its inputs. No autograd crate, no BLAS —
  just triple-nested loops.
- **`src/nn.rs`** — parameter storage (persists across training steps,
  separate from the tape, which is rebuilt fresh each forward pass, similar
  to how PyTorch's dynamic graph works), a hand-written **Adam optimizer**,
  and the **GPT model itself**: token + positional embeddings, `n_layer`
  pre-LayerNorm transformer blocks (multi-head causal self-attention + MLP
  with residual connections), a final LayerNorm, and an output projection
  that's **weight-tied** to the token embedding table (the same trick
  GPT-2 uses).
- **`src/main.rs`** — a character-level tokenizer, the training loop
  (random `block_size`-length chunks, Adam updates), and autoregressive
  sampling with temperature.

## Architecture (default config)

| | |
|---|---|
| Layers | 3 |
| Attention heads | 4 |
| Embedding dim | 64 |
| Context length (block_size) | 64 |
| Vocab | built from the training text (chars) |

This is intentionally small — hand-rolled `O(n^3)` matmuls in a single
thread mean a bigger model would be very slow to train. The architecture
itself (causal self-attention, LayerNorm, residuals, weight tying) is the
same shape as a real GPT; only the scale is toy-sized.

## Running it

```bash
cargo build --release
./target/release/llm_from_scratch
```

It trains on `data/input.txt` (Tiny Shakespeare is included, truncated to
the first 100k characters for speed) for 2000 steps by default, printing
loss every 50 steps, then generates a sample of text.

Override the step count:
```bash
STEPS=5000 ./target/release/llm_from_scratch
```

Swap in your own text by replacing `data/input.txt`.

## What to expect

With this tiny model/data/step-count, don't expect coherent prose — you'll
see loss drop from ~4.1 (roughly `ln(vocab_size)`, i.e. random guessing) to
around ~2.0–2.5, and the generated text will start showing real structure:
word-length tokens, spaces in the right places, capitalization after
periods, recognizable English word fragments, and Shakespeare-ish
punctuation — without ever having been told what a word is. That's the
transformer learning character-level statistics purely from gradient
descent on next-token prediction.

## Where to go from here

- **Bigger/longer**: raise `n_embd`, `n_layer`, `block_size`, or `steps` in
  `main.rs`. Training time scales roughly with `n_embd^2 * n_layer * seq_len`
  per step since matmuls are unoptimized loops.
- **Batching**: right now each step processes one sequence at a time. Add a
  batch dimension (a 3rd tensor axis, or just average gradients over several
  forward/backward passes before the Adam step) for less noisy training.
- **GELU instead of ReLU**: swap `.relu()` for a GELU op in `tensor.rs`
  (needs its own forward + backward) — a small, contained exercise once you're
  comfortable with the existing ops.
- **BPE tokenizer**: replace the char-level tokenizer with a byte-pair
  encoding one for a smaller vocab-to-context ratio.
- **Speed**: replace the hand-written `matmul` triple loop with a real BLAS
  call (e.g. via the `matrixmultiply` crate) without touching the autograd
  logic at all — the `Op::MatMul` forward/backward only needs the *values*,
  not how they're computed.
