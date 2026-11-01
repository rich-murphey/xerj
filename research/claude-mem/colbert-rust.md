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
  as the alternative. Whether `onnx/model.onnx` includes the 384 → 96 head
  has not been checked. Any change must re-pass `colbert check`.
- **MaxSim kernel**: a runtime-dispatched AVX2+FMA variant with F16C widening,
  like xerj's HNSW dot kernel, reading fp16 rows in place. Split the
  candidates across threads for pools over about 10k tokens.
- **Storage and the rerank stage** are not built: where the per-part fp16
  rows live (Postgres `bytea`, a side file or xerj's store), the gather, and
  best-part-per-observation scoring with ties broken by hybrid rank, as
  `src/colbert_rerank.py` does.
- The bench reads its document sample from claude-mem-search-eval's
  `out/<dataset>/rows.jsonl` (`src/export-rows.ts`), which is not in this repo.
