# ColBERT encoder and MaxSim in Rust (candle)

Compiled 2026-11-01 on `dv` (AMD EPYC 7351P, 16 cores / 32 threads)
against xerj `798b2a19`. This is a port of the answerai-colbert-small-v1
reranker that claude-mem-search-eval measured
([results/colbert-rerank.md](https://github.com/rich-murphey/claude-mem-search-eval/blob/main/results/colbert-rerank.md)),
proven against that repo's parity fixture. Wiring it into claude-mem-rs
(the rerank stage and vector storage) is not part of this work.

**Answer.** Parity is exact: ids, masks and kept positions are identical,
the worst token cosine is 1 − 6.6e-12, the MaxSim matrix is within 1.4e-5
of PyLate's, and all 20 query rankings are identical. Speed is not there
yet. On candle 0.9.2 a query takes 48 ms p50 on 8 threads (PyTorch: 25
ms), and documents encode at 406 tokens/s on 15 threads (PyTorch: about
3,100 on 16). Most of the time goes to elementwise ops that candle's CPU
backend runs on one thread, so adding threads barely helps: 1 → 15
threads speeds documents up only 1.6–1.8×. MaxSim is 1.2 ms for 50 × 54
tokens and 23 ms for 50 × 1,000 on one thread, in a scalar SSE loop.

## What was built

`engine/crates/xerj-cmem-embed`, beside the MiniLM embedder:

- `src/colbert.rs`: `Colbert::load` (hub `answerdotai/answerai-colbert-small-v1`
  pinned at `934fa8bb4ce2284f4c2baa232d81aca4d076fa5e`, or a local dir),
  `encode_query(text) → 32 × 96` f32, and
  `encode_documents(texts, Batching) → kept × 96` per text, batched with the
  crate's length-grouped `Batching`. It also has `maxsim(query, doc)`,
  `to_f16` and `from_f16` (`half` 2, already in the tree under candle), and
  `query_input`, `document_input` and `kept` (the ids, mask and output
  positions, for parity).
- BERT loading moved out of `Embedder::load` into `ModelFiles::resolve` and
  `load_bert`, which both models share. The ColBERT weights sit under
  `bert.`, and candle's `BertModel::load` already falls back to
  `{model_type}.`; `linear.weight` [96, 384] comes from the same VarBuilder.
- What the port reproduces, per the eval's "Porting to Rust" section:
  - Query: `[CLS] [unused0] text [SEP]`, the text cut at 31 ids before the
    marker goes in, then padded with `[MASK]` (103) to 32. Mask 0 on the
    pads, but all 32 rows are output.
  - Document: `[CLS] [unused1] text [SEP]`, at most 512 ids. The 32
    `string.punctuation` ids are looked up through the tokenizer, attended
    to, and then dropped from the output.
  - Head: 384 → 96 without bias, then L2 norm per token.
- `src/bin/colbert.rs`: `colbert check <fixture>` and
  `colbert bench <fixture> <rows.jsonl>`, built like the MiniLM `parity`
  binary. The 1 MB fixture is not vendored. Pass its path: it is
  `fixtures/colbert-parity.jsonl` in claude-mem-search-eval (`c82e95e`).

## Parity

`colbert check` (release build; the `quick` profile gives the same result):

| Criterion | Result |
|---|---|
| input ids, attention masks, kept positions (20 queries, 50 docs) | identical, all 70 |
| per-token cosine vs fixture fp32 vectors (1,069 vectors) | min 1 − 6.6e-12 (q15, the empty query, row 31 `[MASK]`); pass is ≥ 0.9999 |
| MaxSim 20 × 50, max abs error | 1.4e-5 (the fixture is rounded to 1e-5) |
| per-query ranking of the 50 docs | identical, 20/20 |

Every criterion passed on the first run. The fixture covers the empty
query and empty doc, a punctuation-only doc, accents, emoji, Chinese, code,
a query over 32 tokens and two docs over 512, so the truncation, marker and
skiplist rules all match PyLate.

## Timings

Release build (fat LTO, `target-cpu=x86-64-v2` per `engine/.cargo/config.toml`),
candle 0.9.2. Thread count is set with `RAYON_NUM_THREADS`, with no taskset;
the host was otherwise idle apart from the load shown. Query latency
is 200 encodes of the fixture's 20 queries, run one at a time. Documents are
the first 200 rows of memsearch en `rows.jsonl` (44,462 input tokens, mean
222 a row). MaxSim uses 50 random unit-vector candidates on one thread, and
"from fp16" includes widening the stored rows.

| Measure | Threads | Rust (candle) | Load avg (1 min) | Python (PyLate / PyTorch) |
|---|---|---|---|---|
| Query encode p50 / p95 | 1 | 103 / 114 ms | 19 | — |
| Query encode p50 / p95 | 8 | 48 / 72 ms | 3.6 | 25 / 28–34 ms (8 threads) |
| Query encode p50 / p95 | 15 | 64 / 83 ms | 5.9 | — |
| Doc encode, tokens/s | 1 | 230 (max_rows 64), 283 (8) | 13 → 3.4 | — |
| Doc encode, tokens/s | 8 | 384 | 4.4 | — |
| Doc encode, tokens/s | 15 | 406 | 4.4 | 3,100–3,200 (16 threads, 2,000-row sample) |
| MaxSim 50 × 54 tokens, p50 | 1 | 1.21 ms f32; 1.77 ms from fp16 | 4–5 | 3.42 ms 1 thread, 2.51 ms 8 threads, incl. widening (MIRACL en depth 50, 4.7k tokens) |
| MaxSim 50 × 1,000 tokens, p50 | 1 | 22.6 ms f32; 32.7 ms from fp16 | 4–5 | ~0.33 µs/token on 1 thread, 0.07 on 8 (GEMM) |

- The 1-thread run overlapped the `ck` code index (load 13–21), so its
  numbers are high. An earlier 15-thread query run under the same `ck`
  load measured 86 / 94 ms.
- 15 threads is slower than 8 for queries: on a 32-token input, synchronising
  more threads costs more than the matmuls gain.
- MaxSim costs 0.45 µs a token in f32 and 0.65 µs from fp16, against numpy's
  0.29 µs (GEMM) + 0.27 µs (widening) on one thread. The kernel is a plain
  8-accumulator dot product. At `x86-64-v2` it compiles to SSE with no
  AVX2/FMA and no F16C. 50 short candidates come to about 1–2 ms, but
  LongMemEval-sized pools (40–100k tokens) would cost 20–65 ms on one thread.

## What dominates

**Document and query encoding: candle's CPU backend, not the model.**
candle 0.9.2 runs unary and binary elementwise ops (bias and residual
adds, scaling, masking, `exp`, `gelu_erf`) and strided copies on the
calling thread. Only the GEMMs (`gemm` crate, runtime AVX2 dispatch) and a
few fused ops (`softmax_last_dim`, `layer_norm`) use rayon. While the
15-thread bench ran, the main thread had used 2:11 of CPU against
0:31–0:35 for each rayon worker. Two probes, each with a temporary
`[patch]` of candle-transformers that was not committed, measure the
serial share on the same 200 rows at 15 threads:

| candle-transformers BERT | Doc tokens/s | Query p50 |
|---|---|---|
| as released (`ops::softmax`: max, sub, exp, sum, div, each serial) | 406 | 64 ms |
| attention softmax → `ops::softmax_last_dim` (parallel, fused) | 538 | 59 ms |
| … plus `gelu_erf` → `relu` (probe only, breaks parity) | 698 | 37 ms |

The softmax swap is free (parity is unchanged: same cosine, same 1.4e-5)
and is a one-line fix to `candle-transformers/src/models/bert.rs`, still
`ops::softmax` on candle main (0.11.1, 2026-10-31). The serial erf GELU
costs about 22 ms of a 59 ms query. The 4.4× that remains is other serial
elementwise work, and probably the GEMM throughput of Zen 1's 128-bit AVX2
units. Without perf (`perf_event_paranoid` = 4 on `dv`) the split between
these two was not measured.

**MaxSim:** the arithmetic. The scalar loop runs at about a third of what
AVX2+FMA would give.

## Left for wiring into claude-mem-rs

- **Encoder speed is the blocker** for ingest. At 406 tokens/s, the live
  store's ~15M kept tokens would take about 10 h on 15 threads, against
  about 1.3 h for PyTorch. The query encode, at 48 ms against 25, is usable
  but doubles the budget. Ways forward, best measured gain first: upstream
  the `softmax_last_dim` change (+32% docs); parallel or fused elementwise
  ops in candle; or the `ort` (onnxruntime) backend that the eval doc names
  as the alternative. ~~Whether `onnx/model.onnx` includes the 384 → 96 head
  has not been checked.~~ Checked in § Profile: it does not, and `ort` is
  the fix to take. Any change must re-pass `colbert check`.
- **MaxSim kernel**: a runtime-dispatched AVX2+FMA variant with F16C widening,
  like xerj's HNSW dot kernel, reading fp16 rows in place. Split the
  candidates across threads for pools over about 10k tokens.
- **Storage and the rerank stage** are not built: where the per-part fp16
  rows live (Postgres `bytea`, a side file or xerj's store), the gather, and
  best-part-per-observation scoring with ties broken by hybrid rank, as
  `src/colbert_rerank.py` does.
- The bench reads its document sample from claude-mem-search-eval's
  `out/<dataset>/rows.jsonl` (`src/export-rows.ts`), which is not in this repo.

## Profile (added 2026-11-01)

Same host. This time `kernel.perf_event_paranoid` was 2 (user-space
sampling only) and the machine was otherwise idle (load 3–8 during runs,
mostly this process). The release build was clean (no probe patch) and
adds `-C force-frame-pointers=yes` and debug info, with `x86-64-v2` stated
explicitly because `RUSTFLAGS` overrides `.cargo/config.toml`. It ran
`perf record -e cycles:u -F 999 --call-graph fp` over `colbert bench` at 15
threads on the same 200 memsearch-en rows. Samples were split into the
query and document phases by the main thread's stack (`encode_query`,
`encode_documents`), and by op from the full inlined stack. The profiled
run measured 389 tokens/s and a 70 ms query p50, so profiling did not
change the picture.

### Where the user-space cycles go

| Op | Thread | Docs (1,264 G cycles) | Query (162 G cycles) |
|---|---|---|---|
| matmul: `gemm_f32` FMA `x2x6` microkernel, `simd::x86::V3` | rayon workers | 64.1% | 63.3% |
| rayon idle spin (`crossbeam_epoch` pin, `sched_yield`) | workers | 9.8% | 25.7% |
| `gelu_erf` (scalar `libm` `erff` + `expf` per element) | main | 8.5% | 4.9% |
| broadcast add/mul/div through `StridedIndex` (attention mask, biases, residuals) | main | 8.1% | 1.9% |
| generic `ops::softmax` (max, sub, glibc `expf`, sum, div) | main | 7.1% | 1.4% |
| matmul output zeroing (`memset`) | main | 0.6% | 0.4% |
| fused `layer_norm` | workers | 0.6% | 1.3% |
| copies / `contiguous` | main | 0.5% | 0.2% |
| tokenization | main | 0.03% | 0.02% |
| main / workers total | | 25.5% / 74.5% | 9.6% / 90.4% |

`to_dtype` does not appear: every tensor is already f32.

**Kernel time.** `cycles:u` cannot see it, but `perf stat` over the same
bench shows a lot of it: 783 s user, 285 s sys, and 21.7M page faults.
candle allocates a fresh output buffer for every op. glibc serves the large
ones (attention scores are `batch × 12 × L × L` f32) with `mmap`, so each
op page-faults its output in and unmaps it on drop. Raising the glibc
thresholds (`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=…:trim_threshold=…`)
cut faults to 0.8M and sys to 133 s, and documents went from 375 to 516
tokens/s (+37%), with the query unchanged (70 ms). Most of the remaining sys
time is probably rayon's `sched_yield` spin. jemalloc via `LD_PRELOAD`
with default settings was worse (35.6M faults, 353 s sys, 371 tokens/s),
because its page purging returns memory too eagerly for this pattern. It
was not tuned further.

### The GEMM kernel, and target-cpu=native

Even at the repo's `x86-64-v2` baseline, the `gemm` crate selects its AVX2+FMA
(`V3`) kernels at runtime, which is the `x2x6` FMA microkernel in the
profile. Per busy core it does about 5 FLOP/cycle (≈ 48 MFLOP a token × 2
passes × 44,462 tokens ≈ 4.2 TFLOP over 811 G worker cycles), about a
third of Zen 1's 16 f32 FLOP/cycle FMA peak. A `-C target-cpu=native` build
(not committed; same flags otherwise, same load), measured against the baseline build:

| | `x86-64-v2` | `native` |
|---|---|---|
| doc tokens/s (max_rows 64 / 8) | 402 / 402 | 425 / 436 (+6–8%) |
| query p50 / p95 | 68 / 95 ms | 58 / 70 ms |
| MaxSim 50 × 54, f32 / from fp16 | 1.21 / 1.86 ms | 1.06 / 1.11 ms |
| MaxSim 50 × 1,000, f32 / from fp16 | 22.5 / 34.2 ms | 19.5 / 20.4 ms |

Parity is unchanged on `native` (worst cosine 1 − 6.7e-12, max score error
1.3e-5, 20/20 rankings). The gain comes from auto-vectorised elementwise
code and from F16C, which makes widening fp16 almost free. The GEMMs
already ran AVX2. A runtime-dispatched MaxSim kernel would get the
MaxSim part of this without changing the baseline.

### The remaining 4.4× after the softmax fix

The GEMM work is not what costs the time. Each document pass is ≈ 2.1
TFLOP. At onnxruntime's measured rate on the same host (below: 6,729
tokens/s × 48 MFLOP ≈ 320 GFLOP/s on 15 threads), that is about 7 s of
the 110 s a pass takes. The rest goes to work that runs on one thread
while 14 rayon workers sleep or spin:

- **Serial elementwise ops on the main thread**: 161 G user cycles a
  pass, about 55 s at 2.9 GHz, roughly half the wall time. In order:
  `gelu_erf` (scalar `libm` erf; replacing it with relu as a probe took
  docs 538 → 698 tokens/s), strided broadcast adds (the `StridedIndex`
  iterator per element, no SIMD), and the generic softmax (fixed by the
  one-line change, 406 → 538).
- **Page faults** on fresh per-op buffers: 27% of all CPU time is kernel,
  and the allocator change alone is +37%.
- **Fork-join overhead**: every GEMM is a rayon join from a non-pool
  thread, so workers wake, compute and spin. That is 10% of document
  cycles and 26% of query cycles, which is also why 15 threads are slower
  than 8 for queries.

Upper bound for candle with all three fixed: the main thread's elementwise
share is about half the wall time, and page faults are a further ~27%, so
even removing all of both leaves candle at roughly 1,000–1,300 tokens/s,
still 5× below onnxruntime.

### onnxruntime on the same inputs

`onnx/model.onnx` at `934fa8bb` outputs `last_hidden_state`
`[batch, seq, 384]`: **it does not include the 384 → 96 head**. Applying
`linear.weight` from `model.safetensors` and L2-normalising reproduces the
fixture (worst token cosine 0.9999999 over all 50 fixture docs).
onnxruntime (Python binding of the C++ runtime that the `ort` crate wraps),
on the same 200 rows with the same length-grouped batches of up to 64 rows
or 4,096 padded tokens:

| Threads | Doc tokens/s | Query p50 / p95 |
|---|---|---|
| 15 | 6,729 | 15.7 / 16.3 ms |
| 8 | 4,265 | 12.4 / 12.8 ms |

That is 17× candle's documents and 4× its query, and 2.2× PyTorch's
3,100 tokens/s.

**Fix to take: `ort`.** xerj-ai already carries `ort = 2.0.0-rc.12` behind
its `onnx-experimental` feature (GNU/Linux only; candle stays for musl). An
`ort` encoder needs the same tokenisation and markers as `colbert.rs`, the
hub's `onnx/model.onnx` at the pinned revision, and the head as a 384 × 96
matmul in Rust. Expect about 6.5k tokens/s and a 12–16 ms query (Python
call overhead is negligible next to batch times of tens of ms). That puts
the live store's ~15M tokens at about 40 min on 15 threads. It must
re-pass `colbert check`. The candle softmax fix is filed upstream
(https://github.com/huggingface/candle/issues/4084) and helps any
candle BERT user, but it does not close the gap for this one.
