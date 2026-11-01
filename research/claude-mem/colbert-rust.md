# ColBERT encoder and MaxSim in Rust (candle)

> 2026-11-01: candle has since been removed from xerj-cmem-embed; both
> models run on ONNX Runtime. See § Candle removed, MiniLM on ort.

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

## ORT backend (added 2026-11-01)

`colbert::onnx::OrtColbert` (`src/colbert/onnx.rs`, commit `ebe3f8ab`),
behind the crate's new `onnx-experimental` feature. It uses the same
`ort = 2.0.0-rc.12` dependency line and features as xerj-ai, so it is
GNU/Linux only, and candle stays the default and the musl path. Both
backends implement `ColbertEncoder`, which owns tokenisation, markers,
`[MASK]` padding, length-grouped batching, the skiplist and the head, so
only BERT's forward pass differs. ORT runs the hub export at `934fa8bb` and
returns `last_hidden_state`. The 384 → 96 `linear.weight` (read from
`model.safetensors`) and the L2 norm are applied in Rust with the same
candle code as the candle path. The `colbert` binary picks ORT when
`COLBERT_ONNX` names the file.

**ONNX files at `934fa8bb`:** `model.onnx` (fp32, 133 MB), `model_fp16`,
`model_int8` (byte-identical to `model_quantized`, same LFS oid),
`model_uint8`, `model_q4`, `model_q4f16` and `model_bnb4`. There are no
O-level optimised exports; ORT optimises at load (Level3).

### Parity (`colbert check`)

| Backend | ids / masks / kept | worst token cosine | MaxSim max abs error | rankings |
|---|---|---|---|---|
| candle | identical | 1 − 6.6e-12 | 1.4e-5 | 20/20 |
| ORT fp32 `model.onnx` | identical | 1 − 6.0e-12 (d18 row 11) | 1.4e-5 | 20/20 |
| ORT int8 `model_int8.onnx` | identical | 0.9636 (q15 row 1); median 0.9981; 1,039 of 1,069 below 0.9999 | 0.384 | 0/20 |

int8 fails parity, and is reported separately. The first place where its
ranking departs from the fixture's is between place 0 and 7, median 3.
Whether that costs rerank accuracy would need the eval's R@5 run, not this
fixture.

### Timings

Same harness, rows and host, release builds. The host carried 4–9 load
from other users' processes (emacs, bun) and was otherwise free; each
measurement's load is in the run log. "Candle fixed" is the softmax
`softmax_last_dim` patch (temporary `[patch]`, not committed) plus
`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=4294967295:glibc.malloc.trim_threshold=4294967295`.
Query is p50 / p95 over 200 single encodes of the fixture's queries;
documents are the 200 memsearch-en rows (44,462 tokens). Thread count is
`RAYON_NUM_THREADS` for candle and ORT intra-op threads for ORT.

| | candle | candle fixed | ORT Rust fp32 | ORT Rust int8 | PyTorch (PyLate) |
|---|---|---|---|---|---|
| docs tokens/s, 8 threads | 384 | 572 | 4,176 | 4,761 | — |
| docs tokens/s, 15 threads | 389–406 | 618 | **5,078** | 6,490 | ~3,100 (16 threads) |
| query, 1 thread | 103 / 114 ms¹ | 82 / 98 ms | 37.0 / 37.9 ms | — | — |
| query, 4 threads | — | 99 / 121 ms | 17.5 / 18.6 ms | — | — |
| query, 8 threads | 48 / 72 ms | 47–84 / 75–102 ms² | **13.1 / 15.3 ms** | 13.2 / 13.5 ms | 25 / 28–34 ms |
| query, 15 threads | 64–86 / 83–95 ms | 71 / 95 ms | 17.2 / 19.2 ms | 16.3 / 18.9 ms | — |
| load avg (1 min) | 4–19 | 5–9 | 4–8 | 7–10 | 6–12 |

¹ under `ck` index load (19). ² two back-to-back runs. Candle's query time
swings by tens of ms run to run, whatever the allocator; ORT's does not.

**ORT session options**, as built: `GraphOptimizationLevel::Level3`
(all graph optimisations), `with_intra_threads(N)`, inter-op left at the
default (sequential executor, no inter-op pool), CPU memory arena and
memory pattern at ORT's defaults (on), and `with_intra_op_spinning`
on by default and off with `COLBERT_ONNX_SPIN=0`. One session sits
behind a mutex, as in xerj-ai.

**Spinning** matters for single queries above a few threads, and helps
documents too:

| intra threads | query p50 / p95, spinning on | spinning off |
|---|---|---|
| 1 | 37.0 / 37.9 ms | 36.9 / 37.4 ms |
| 4 | 17.5 / 18.6 ms | 17.7 / 23.4 ms |
| 8 | 13.1 / 15.3 ms | 20.6 / 23.9 ms |
| 15 | 17.2 / 19.2 ms | 23.8 / 27.7 ms |
| 15, docs | 5,078 tokens/s | 4,416 tokens/s |

xerj-ai's MiniLM backend turns spinning off. For a ColBERT query on a
dedicated pool, leaving it on (the ORT default) saves 7 ms at 8 threads.
The cost is threads burning CPU between runs. 8 intra-op threads is the
query sweet spot here, as it was for PyTorch.

**Against Python onnxruntime.** Rust ORT reaches 5,078 tokens/s against
the 6,729 measured earlier through Python. Most of that gap is the runtime
version: rc.12 links onnxruntime 1.24.2, and Python onnxruntime 1.24.2 on
the same rows and batches gives 5,829 tokens/s, against 6,730 for 1.31.1
(+15%). The remaining 13% is on the Rust side: copying
`last_hidden_state` into a candle tensor, the head on candle's rayon pool
beside ORT's spinning pool, and tokenisation. None of this was profiled.

**Size.** `colbert` is 6.9 MB with candle only and 26.6 MB with
`onnx-experimental` (+19.6 MB). `download-binaries` links onnxruntime
1.24.2 statically: no `.so` is copied and `ldd` shows none. The build
downloads the prebuilt library from pyke's CDN.

### What this means for claude-mem-rs

- **ORT fp32 is the encoder to wire.** It passes parity exactly, encodes
  documents 12.5× faster than candle at 15 threads (8× faster than candle
  with both fixes), and the query takes 13 ms at 8 threads, half
  PyTorch's 25. The live store's ~15M tokens take about 50 min on 15
  threads.
- **Blocks or costs** before it can go in:
  - GNU/Linux only. A musl build of claude-mem-rs gets candle at 400–600
    tokens/s.
  - The build needs network access to pyke's CDN, or `ORT_LIB_LOCATION`
    with a vendored onnxruntime, for offline and reproducible builds.
  - `ort` 2.0.0 is still a release candidate, and rc.12 pins onnxruntime
    1.24.2, 15% slower than 1.31. Moving to a newer rc also moves
    xerj-ai, which shares the dependency line.
  - One run at a time per session (mutex). Concurrent query encodes need
    a session pool at ~133 MB of weights each, or a queue.
  - The ORT and rayon thread pools must be sized together so they don't
    oversubscribe.
- **int8** is 28% faster on documents (6,490 tokens/s) and no faster on
  queries. It fails token parity, so using it needs its own rerank
  accuracy run first.

## T1/T2: ORT version, fused attention, NUMA layout (added 2026-11-01)

The two tests Rich approved. Same host. `pgrep '[t]arget/release/engines'`
found no other benchmark running before any run. Load is noted per row.
At 15–16 threads most of the 1-minute load is the bench's own threads;
the background was 4–5 (emacs, bun).

### T2a: `ort` 2.0.0-rc.15

rc.15 links onnxruntime **1.31.0** (`ort-sys` build script; rc.14 moved to
1.31, per its release notes), against rc.12's 1.24.2. ~~**It cannot land on
its own.**~~ Landed in `6b965768` (2026-11-01, Rich approved the xerj-ai
change). `ort-sys` declares `links = "onnxruntime"`, so the workspace
holds one version, and xerj-ai moves with xerj-cmem-embed. rc.15 changed
`Session::builder()` to `Session::builder(&Environment)`, and allows one
`Environment` per process, ever (`ort::init().build()` fails once one has
existed). Its only process-global accessor is the private
`Environment::current()`. The doc-hidden `ort::test_util::test_env()` is
not an API to build on. So xerj-ai now owns the environment:
`xerj_ai::onnx::environment()` creates it once, behind a mutex, and every
session uses it. That covers the MiniLM pool, and the ColBERT backend
through a new optional `xerj-ai` dependency under xerj-cmem-embed's
`onnx-experimental` feature. MiniLM's session options are unchanged
(spinning off, `Level3`, thread count).

Verification at `6b965768`:
- **Checks:** `cargo check --workspace --all-targets` is clean, and so are
  xerj-ai, xerj-cmem-embed, xerj-engine and xerj-server with
  `onnx-experimental`.
- **xerj-ai's onnx tests:** 19/19, including the ignored
  `independent_pool_sessions_are_bit_identical`, run against
  all-MiniLM-L6-v2's `onnx/model.onnx` at `1110a243`. It builds three
  sessions in one process on the shared environment.
- **`colbert check`:** 20/20 on the hub model (1 − 5.9e-12, 1.39e-5) and on
  the fused head-in-graph model (1 − 5.7e-12, 1.77e-5).
- **Bench (load 4.7–11):** 1 × 16 pinned to cores 0–15 gives 8,298
  tokens/s (T1: 8,604); the 1 × 8 pinned query is 10.5 / 11.15 ms
  (T1: 10.7 / 10.95).
- **Binary size:** the `colbert` binary with the feature is 31.2 MB
  (rc.12: 26.6 MB).

Hub `onnx/model.onnx`, same session options as before (`Level3`): parity
1 − 5.9e-12, 1.39e-5, 20/20. Docs at 15 threads: 5,534–5,901 tokens/s
(rc.12: 5,078), so **+9–16%** from the runtime version alone. Query at 8
threads: 13.4–14.1 ms p50 (rc.12: 13.1), unchanged within noise.

### T2b: eager re-export, fused attention, head in the graph

`research/claude-mem/colbert_export.py` (uv script):
- **Versions:** torch 2.11.0 (+cu130 wheel, run on CPU), transformers
  5.3.0, onnx 1.23.2, onnxruntime 1.31.1.
- **Export:** `BertModel.from_pretrained(..., revision=934fa8bb,
  attn_implementation="eager")`, TorchScript exporter, opset 17, dynamic
  batch and sequence.
- **Optimizer:** `onnxruntime.transformers.optimizer.optimize_model(model_type="bert", num_heads=12, hidden_size=384)`
  (config: 12 heads, hidden 384).

With `--head` the graph also carries `linear.weight` (from `model.safetensors`)
and `F.normalize`, with output `token_vectors` [batch, seq, 96]. The models
are stored outside the repo, in `~/.cache/xerj-colbert/` on dv:

| File | sha256 | Ops after optimisation |
|---|---|---|
| `colbert-eager-opt.onnx` (body, Rust head) | `c8d83894a0b3993a23843e46f5a2b52ae1bdfc58d12ec9703f29fb545b70294a` | Attention 12, EmbedLayerNormalization 1, SkipLayerNormalization 24, BiasGelu 12, MatMul 36, Cast 3 |
| `colbert-eager-head-opt.onnx` (head in graph) | `45a436b89aec98f5d364d7e4980bb958912a37593e20015de9cb840557b4d743` | the same + MatMul 1, ReduceL2, Clip, Div, Shape, Expand, 2 Constant |

The hub export fused no Attention (0 of 12). The eager export fuses all
12, plus the embedding layer norm, and leaves no other unfused ops.

**Parity** (`colbert check`; the ranking test now lets documents the
fixture scores exactly equal swap places, see below):

| Model | `ort` | worst token cosine | max score error | rankings |
|---|---|---|---|---|
| hub `model.onnx` | rc.15 | 1 − 5.9e-12 | 1.39e-5 | 20/20 |
| eager, Rust head | rc.15 | 1 − 5.8e-12 | 1.35e-5 | 20/20 |
| eager, head in graph | rc.15 | 1 − 5.7e-12 | 1.77e-5 | 20/20 |
| hub `model.onnx` | rc.12 | 1 − 6.0e-12 | 1.40e-5 | 20/20 |
| eager, head in graph | rc.12 | 1 − 5.5e-12 | 1.40e-5 | 20/20 |
| eager, Rust head | rc.12 | 1 − 5.8e-12 | 1.40e-5 | 20/20 (19/20 under the strict test) |
| candle | — | 1 − 6.6e-12 | 1.40e-5 | 20/20 |

The one strict-test miss is q4 d42/d45, which the fixture scores 29.76151
both, an exact tie at its 1e-5 rounding. The fixture breaks ties by
index, so a 1e-6 difference flips them. A strict test on exact ties is
moot, and `colbert check` now compares the sequence of fixture scores in
our order, so only exactly tied documents may swap.

### T2c: version against fusion (200 rows, docs at 15 threads, query at 8)

| Model | `ort` | level | docs tokens/s (load) | query p50 / p95 (load) |
|---|---|---|---|---|
| hub `model.onnx` | rc.12 | Level3 | 5,078 (7.4) | 13.1 / 15.3 ms (6.0) |
| hub `model.onnx` | rc.15 | Level3 | 5,534–5,901 (4.4–7.5) | 13.4–14.1 / 15.3–15.5 ms (7–8) |
| hub `model.onnx` | rc.15 | All | 5,787 (9.0) | 16.7 / 17.4 ms (8.0) |
| eager, Rust head | rc.15 | Level3 | 6,419 (9.5) | 10.8 / 11.7 ms (9.2) |
| eager, Rust head | rc.15 | All | 6,579 (8.8) | 10.2 / 11.4 ms (9.0) |
| **eager, head in graph** | rc.15 | All | **7,079** (9.6) | **9.6 / 10.6 ms** (9.1) |

- **Version:** +9–16% on docs.
- **Fused attention:** docs +16% (Level3) and +14% (All), against the hub
  model in the same matrix run; query p50 −19% (Level3) and −39% (All).
- **Head in the graph:** another +8% on docs and −6% on queries. It removes
  the hidden-state copy and candle's second (rayon) thread pool.
- **`All` against `Level3`** (`ORT_ENABLE_LAYOUT` in `ort`): within noise.
- **Net,** against the rc.12 hub baseline: **+39% docs, −27% query.**

### T1: sessions, threads and pinning

Run with `colbert-eager-head-opt.onnx` on rc.15 at `All`, with intra-op
spinning on, over all 955 memsearch-en rows (221,869 input tokens).
Multi-session layouts are separate processes under `numactl
--physcpubind=<that node's physical cores> --membind=<node>`, each given a
round-robin share of the rows. Their aggregate is total tokens over the
slowest process's wall time; the sum of per-process rates is in brackets.

| Layout | docs tokens/s | load (start → end) |
|---|---|---|
| (i) 1 × 15 threads, unpinned | 7,210 | 7.1 → 11.5 |
| (ii) 1 × 16, `taskset -c 0-15` (physical cores) | 8,604 (+19%) | 11.2 → 14.4 |
| (iii) 4 × 4, one per NUMA node, node-local memory | 8,804 [9,235] (+22%) | 14.0 → 15.6 |
| (iv) 2 × 8, nodes 0+1 and 2+3 | 8,946 [9,037] (+24%) | 14.3 → 16.2 |
| (ii) on rc.12 (the committed state) | 8,347 | 5.3 → 10.5 |

Single-query encode, 8-row docs file:

| Layout | p50 / p95 / max | load |
|---|---|---|
| 1 × 4, node 0 cores 0–3, node-0 memory | 13.4 / 14.5 / 14.9 ms | 13.4 |
| 1 × 8, unpinned | 10.4 / 11.6 / 44.8 ms | 11.6 |
| 1 × 8, cores 0–7, nodes 0–1 memory | 10.7 / 10.95 / 11.7 ms | 9.6 |
| 1 × 8, cores 0–7, rc.12 | 10.5 / 10.8 / 11.1 ms | 10.9 |

**Reading:**
- **Keeping threads on physical cores is most of the gain.** That means
  no SMT siblings and no migration onto the cores the background load
  uses.
- **NUMA sharding adds only 2–4% over one pinned 16-thread session,** at
  4× (or 2×) the sessions and weight copies. The 1.22× scaling from 8 to
  15 threads seen earlier was mostly unpinned threads, not cross-die
  traffic.
- **Queries:** pinning cuts the tail (max 44.8 → 11.7 ms) rather than the
  median. Four threads on one node cost 3 ms more than eight on two
  nodes.

### Recommended production configuration (dv-class host)

- **Model:** `colbert-eager-head-opt.onnx`, regenerated with
  `colbert_export.py` and checked by sha256, run with ORT at `All`,
  intra-op spinning on. It passes the fixture 20/20.
- **Bulk ingest:** one session, 16 intra-op threads, pinned to physical
  cores 0–15. That gives 8.3k tokens/s on rc.12 and 8.6k on rc.15, so the
  live store's ~16.6M input tokens (16.3M MiniLM tokens plus a marker per
  part) take **about 32–33 min**. Use 2 × 8 NUMA sessions only if the
  last 4% matters.
- **Live queries:** one session, 8 threads pinned to cores 0–7 (nodes
  0–1), 10.5 ms p50, 11 ms p95. If ingest and queries share the host,
  give queries 4 threads on one node (13.4 / 14.5 ms) and ingest the
  other 12 cores. That ingest rate was not measured; at three quarters of
  (ii) it would be about 6.3–6.5k tokens/s.
- **Runtime:** rc.15, landed in `6b965768`: +9–16% unpinned and +3%
  pinned, plus the memory-safety fixes in its notes.

## Candle removed, MiniLM on ort (added 2026-11-01)

Rich approved, 2026-11-01: one engine, ONNX Runtime, with parity fixtures
as the cross-check
([claude-mem-rs#7](https://github.com/rich-murphey/claude-mem-rs/issues/7)).
Landed in `5bb12a59`.

- **ColBERT:** the candle backend and the `ColbertEncoder` trait are gone;
  `colbert::Colbert` is the ort encoder. The hub body's 384 → 96 head is
  plain Rust over `linear.weight`, read with `safetensors`; a graph with a
  `token_vectors` output (`colbert-eager-head-opt.onnx`) skips it.
  `colbert check`: 20/20 on hub `onnx/model.onnx` (1 − 5.8e-12, 1.45e-5)
  and on `colbert-eager-head-opt.onnx` (1 − 5.7e-12, 1.77e-5).
- **MiniLM:** runs the file live claude-mem runs: `Xenova/all-MiniLM-L6-v2`
  `onnx/model.onnx` at `751bff37` (sha256 `759c3cd2…c46e`, byte-identical
  to the `@huggingface/transformers` 4.3.1 cache in both live containers),
  with that revision's `tokenizer.json`. The stored model tag is
  `Xenova/all-MiniLM-L6-v2/fp32/mean/l2/256` on all 312,114 rows. Mean
  pooling and L2 norm in Rust, 256-token cap as before.
- **Build:** `ort` and `xerj-ai` (for the shared environment) are no longer
  optional, and `onnx-experimental` is gone from this crate. pyke ships
  no musl library (ort-sys `dist.tsv`), so there is no musl build of the
  crate, and none is needed: release.yml builds `-p xerj-server` only, and
  nothing depends on this crate. `--workspace` builds now unify
  `xerj-ai/onnx-experimental` on; `cargo check --workspace --all-targets`
  is clean, and `xerj-server` alone has no `ort` in its tree.

**MiniLM parity**, 500 live `observation_embeddings` rows (450 random, 50
narratives over 1,600 chars; 51 at the 256 cap), and 50 queries (the 22
search-vs-live queries plus 28 live titles):

| Comparison | min cosine | below 0.9999 |
|---|---|---|
| ort vs stored vectors | 1 − 7.3e-13 | 0 / 500 |
| ort vs candle (production build at `da26465`), rows | 1 − 9.4e-13 | 0 / 500 |
| ort vs candle, queries | 1 − 7.0e-13 | 0 / 50 |
| candle vs stored | 1 − 9.0e-13 | 0 / 500 |

Top-10 over the 500 rows is identical, order included, for 50/50 queries
in each pairing: ort query on stored vectors against candle query on
stored vectors, and ort query on ort re-embedded rows against stored. The
stored vectors stay; nothing needs re-embedding.

**Timings.** `parity bench` over the 450 random rows (47.0 tokens mean),
pinned with `taskset` to physical cores, intra-op spinning off as in
xerj-ai. Each run started only with no `rustc`/`cc1`/`cargo build`/
`target/release` process, 1-minute load under 4, and the dv-bench lock
held exclusively; load at start was 3.68–3.97. Docs tokens/s is the best
batch setting; query is p50 / p99 over 200 encodes of an 18-token query.

| | docs tokens/s, 16 thr (0–15) | 8 thr (0–7) | query 16 thr | query 8 thr |
|---|---|---|---|---|
| candle (`da26465`) | 1,415 | 1,307 | 17.6 / 24.6 ms | 19.0 / 29.9 ms |
| ort, hub `model.onnx` | 7,182 (5.1×) | 7,149 (5.5×) | 9.5 / 11.4 ms | 8.4 / 10.6 ms |
| ort, fused (not wired) | 11,407 (8.1×) | 9,033 (6.9×) | 6.4 / 7.6 ms | 5.5 / 6.6 ms |

"Fused" is the hub file through `onnxruntime.transformers.optimizer`
(`model_type="bert"`, 12 heads, hidden 384, `opt_level=0`; onnxruntime
1.31.1): Attention 6, EmbedLayerNormalization 1, SkipLayerNormalization
12, BiasGelu 6, nothing left unfused. It passes parity against the stored
vectors (min 1 − 1.07e-12, 0 / 500 below 0.9999) and is another 1.3–1.6×
on docs and 1.5× on queries. It is not wired: production keeps the file
claude-mem runs. Taking it means pinning the optimiser output by sha256,
as for `colbert-eager-head-opt.onnx`. The file is outside the repo, at
`~/.cache/xerj-minilm/minilm-opt.onnx` on dv (sha256 `bedfe603…8186`).

Spinning, from an earlier sweep at load 7–17 (indicative only): it did
not help single queries at 1 or 8 threads, and helped docs at 4 threads
(68 → 104 passages/s). MiniLM keeps it off.

**claude-mem-rs** (rev bump in `b02783e`, described in `1a5535d`):
`cargo tree -i candle-core` is empty. The contract test gives 680/0.
Search vs live on the 22 queries is the same table under candle and ort
(top-1 15/22, overlap@10 85%, live top-3 in ours 66/66), and `/v1/search`
top-10 is identical between the candle and ort builds on the same SQLite
copy for 50/50 queries. One fix came out of it: ort's INFO tracing went
to stdout and corrupted the `embed` subcommand's vectors, so the default
log filter is now `info,ort=warn`.
