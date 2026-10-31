# Phase 0: claude-mem retrieval on xerj vs pgvector

Run 2026-10-31 on `dv` (32-core EPYC 7351P, 125 GiB). xerj at c7abfced
(`--profile profiling`: opt 3, thin LTO), `--embed-mode lexical`, bound to
127.0.0.1, data dirs under `~/.cache/xerj-phase0/`. xerj embedded nothing:
every vector is a stored MiniLM vector.

**Verdict: no-go as xerj stands.** Accuracy matches or beats pgvector on
the four datasets xerj could finish, but xerj's kNN does not finish at
claude-mem's scale. Every claude-mem search is project-scoped, and xerj
serves a filtered kNN only by brute force. That scan takes 15 s at
81k parts and passes the 30 s search timeout at 179k–300k, where it
returns 0 hits. The live store is one 300k-part project.

## What was compared

- **Baseline:** claude-mem's own `PostgresObservationRepository.search`,
  from fork main at a02f23770 (the commit the live server runs). It runs on
  `postgres-pgvector-textsearch:17` (pgvector 0.8, pg_textsearch bm25),
  through claude-mem-search-eval's `src/server-search.ts --mode scoped` with
  minilm-l6 release vectors. claude-mem-search-eval e6f6d7e added
  `--rankings-out` to save per-query rankings and milliseconds. Its scores
  reproduce the published `results/*-server-scoped-bm25.md` within 0.003 R@5.
- **xerj:** the same release vectors and texts, loaded by
  [load-eval.ts](load-eval.ts) into two indexes per dataset.
  `cm-<d>-parts` holds one doc per (observation, part), with keyword fields
  `observation_id` and `project_id`, text `text`, and `vec` as
  `dense_vector` 384 cosine. `cm-<d>-obs` holds one doc per observation,
  with its parts joined. Each query gets the scope `server-search.ts` gives
  it: its project, or LongMemEval's haystack as a `terms` filter on
  `observation_id`.
- **Labels and metrics:** claude-mem-search-eval's own loader
  ([qrels.ts](qrels.ts)). hit@5 is the share of queries with at least one
  relevant document in the top 5. MRR@10 and R@5 are as in its
  `metrics.ts`. Δ is paired against the pgvector counterpart, with a 95%
  bootstrap interval (2000 resamples, seed 1).

### Fusion found in claude-mem

`searchHybrid` in `src/storage/postgres/observations.ts` fuses by
**reciprocal rank fusion, k = 60** (sum of 1/(60 + rank) over the legs an
observation appears in; ties go to `updated_at DESC`). Each leg holds 100
observations:

- **Keyword leg:** bm25 (pg_textsearch) over the whole observation's
  `content` english tsvector. It matches any query term.
- **Vector leg:** the 400 nearest parts from HNSW
  (`hnsw.iterative_scan = strict_order`, `ef_search` = parts). Each
  observation ranks by its nearest part. The leg reads ×4 more parts while
  it holds fewer than 100 observations and the cutoff was hit, up to
  100 × 66 parts.

[lib.ts](lib.ts) reproduces this client-side: `knnLeg`, `bm25ObsLeg` and
`rrf`. `bm25PartLeg` (BM25 over parts, collapsed by the best part) is an
extra arm.

### Collapse

Collapse is done **client-side**, by first-seen `observation_id` over
part hits. claude-mem's leg needs "400 parts → ≤100 observations, widen if
short", which a single `collapse` request cannot express. xerj accepts
`collapse` beside `knn` (es_compat folds kNN into the normal search path),
but this was not relied on. Every dataset that finished has one part per
document, so collapse never changed a ranking there.

## Accuracy

`—`: xerj did not finish. Filtered kNN timed out at 30 s, returning 0 hits
(below). pg rows are claude-mem's search.

| Dataset (queries) | pg keyword hit@5 / MRR | pg vector | pg hybrid | xerj BM25 (obs) | xerj kNN | xerj hybrid | Δ hybrid hit@5 (95%) | Δ hybrid MRR (95%) |
|---|---|---|---|---|---|---|---|---|
| LongMemEval-S (470) | 0.955 / 0.909 | 0.979 / 0.912 | 0.981 / 0.932 | — | — | — | — | — |
| memsearch zh (2172) | 0.434 / 0.320 | 0.300 / 0.200 | 0.459 / 0.315 | 0.767 / 0.631 | 0.300 / 0.200 | 0.587 / 0.428 | +0.128 (+0.111–+0.145) | +0.112 (+0.100–+0.125) |
| memsearch en (2172) | 0.871 / 0.724 | 0.723 / 0.560 | 0.840 / 0.694 | 0.888 / 0.734 | 0.723 / 0.560 | 0.838 / 0.695 | −0.001 (−0.009–+0.006) | +0.001 (−0.006–+0.007) |
| MIRACL en (799) | 0.493 / 0.345 | 0.579 / 0.396 | 0.677 / 0.480 | — | — | — | — | — |
| MIRACL zh (393) | 0.003 / 0.001 | 0.005 / 0.002 | 0.005 / 0.003 | 0.379 / 0.230 | 0.005 / 0.002 | 0.219 / 0.115 | +0.214 (+0.176–+0.252) | +0.112 (+0.090–+0.135) |
| MIRACL ko (213) | 0.394 / 0.328 | 0.019 / 0.013 | 0.310 / 0.155 | 0.394 / 0.332 | 0.028 / 0.020 | 0.296 / 0.158 | −0.014 (−0.038–+0.009) | +0.003 (−0.011–+0.016) |
| MIRACL ja (860) | 0.073 / 0.056 | 0.020 / 0.014 | 0.064 / 0.040 | — | — | — | — | — |

Full table with R@5, BM25-over-parts and hybrid-over-parts arms:
`node research/claude-mem/score.ts` (output kept outside the repo).

- **The kNN arm equals pgvector exactly** (Δ 0.000 on memsearch and MIRACL zh).
  On MIRACL ko xerj is slightly better: its filtered kNN is exact, and
  pgvector's HNSW loses recall there, as claude-mem-search-eval's README
  already notes.
- **BM25 ties or wins.** English: +0.017 hit@5 (+0.008–+0.026) on memsearch
  en. CJK: +0.33 on memsearch zh and +0.38 on MIRACL zh. xerj's Unicode
  tokenizer splits ideographs, while Postgres's english parser keeps a CJK
  run as one token that no document contains.
- **Hybrid** ties on English and Korean and gains +0.13 and +0.21 hit@5 on
  the two Chinese sets.
- **xerj quirk:** a per-field mapping `"analyzer": "english"` is ignored.
  The field indexes with `standard`: no stop words, no stemming. With it,
  xerj BM25 lost 0.047 hit@5 to pg on memsearch en. English analysis has to
  be set as the index default (`settings.analysis.analyzer.default`),
  which [lib.ts](lib.ts) does. Measured with `_analyze` and `_count`
  against two scratch indexes.

## Latency

Single client, one search at a time, pg and xerj interleaved query by
query ([latency.ts](latency.ts)). Each query runs one arm only, so xerj's
query-result cache is never hit. Wall time at the client includes HTTP and
JSON for xerj, and the pg driver for claude-mem.

| Dataset (parts) | Arm | pg p50 / p95 ms | xerj p50 / p95 ms |
|---|---|---|---|
| memsearch en (955) | keyword | 2.2 / 3.5 | 5.8 / 13.5 |
| | vector | 7.1 / 10.0 | 9.2 / 32.5 |
| | hybrid | 7.5 / 10.5 | 9.0 / 34.5 |
| memsearch zh (955) | hybrid | 7.4 / 14.3 | 11.4 / 40.1 |
| MIRACL ko (43k) | keyword | 1.8 / 3.2 | 2.9 / 4.1 |
| | vector | 17.0 / 20.0 | 10.0 / 15.8 |
| | hybrid | 17.2 / 19.7 | 11.1 / 18.4 |
| MIRACL zh (81k) | keyword | 3.4 / 21.1 | 2.6 / 4.2 |
| | vector | 35.8 / 44.4 | **15,118 / 15,419** |
| | hybrid | 55.0 / 58.8 | **14,987 / 15,192** |
| MIRACL en (179k), ja (185k) | vector (pg) | 16.6–20.1 / 22.6–23.7 | timeout (30 s), 0 hits |
| LongMemEval (200k) | vector (pg) | 308 / 324 | timeout (30 s), 0 hits, 277 s with leg widening |
| live store (300k, 1 project) | vector | not measured (read-only) | timeout (31.9 s), 0 hits, 3 of 3 queries |

Why, from `xerj-engine/src/index.rs` (`run_knn_hnsw` and its caller):
HNSW serves only a **plain** top-level kNN with no filter, boost,
similarity cutoff or aggs. Every filtered kNN is an exact brute-force scan
that hydrates documents. The slow-query log shows about 445 MB of read
budget per query at 81k parts (`dec=18437ms scan=11361ms` at 179k).
Unfiltered kNN also returned 0 hits after the timeout at 81k and 179k,
though `_cat/ann` reports the graph present and fully covered. A
`_forcemerge` to 1 segment took 18 min at 81k and changed nothing. At
43k, unfiltered kNN answers in 110–270 ms.

## Resources

| | xerj | Postgres (live) |
|---|---|---|
| Live store loaded | 300,306 parts / 43,400 observations in 270 s (exporter RSS 392 MiB) | 300,523 embedding rows at query time; it grows while the service runs |
| Disk | 1.2 GiB data dir (`cm-live-parts` 1.1 GB, `cm-live-obs` 50 MB) | `observation_embeddings` 1,675 MB + `observations` 289 MB |
| RAM | xerj RSS 9.7 GiB after load (peak 9.9 GiB) | shared with the live service; not isolated |
| Eval instance (7 datasets, ~690k parts) | 4.4 GiB on disk, peak RSS 15.3 GiB | — |

## Embedding compatibility

[bench-live.ts](bench-live.ts) re-embeds stored live part texts with
claude-mem-search-eval's encoder (MiniLM fp32, masked mean, L2, 256 tokens
with `[SEP]` restored). Result: 139 of 139 parts, cosine ≥ 0.9999 (min
1.000000). The part texts come from the same `observationEmbeddingParts`
rules ([export-live.ts](export-live.ts)): 0 of 300,306 parts lacked
derivable text.

## Go / no-go

**No-go** for xerj as the claude-mem store, at c7abfced. Retrieval quality
is there: kNN is exact, BM25 ties English and wins CJK, and hybrid ties or
wins on all 4 datasets that finished. But project-scoped kNN, which is
every claude-mem search, is a brute-force scan. It costs 15 s at 81k parts
and times out with 0 hits on the 300k-part live store. Phase 1 needs
filtered HNSW, or a vector-column brute path that does not hydrate
`_source`. Then the 3 unfinished datasets get rerun.

## Reproduce

Data and indexes stay outside the repo, in `~/.cache/xerj-phase0/`.

```sh
# release vectors + rows/scope (claude-mem-search-eval: scripts/fetch.sh v1 <dataset>; node src/export-rows.ts)
node research/claude-mem/qrels.ts --eval ~/wt/claude-mem-search-eval/<tree> --dest $C/qrels
bun  src/server-search.ts --claude-mem ~/wt/claude-mem/<tree at a02f23770> --db $PG --dataset $d \
     --models minilm-l6 --mode scoped --rankings-out $C/baseline        # in claude-mem-search-eval
XERJ_URL=http://127.0.0.1:9300 node research/claude-mem/load-eval.ts   --out $C/eval-out --dataset $d
XERJ_URL=http://127.0.0.1:9300 node research/claude-mem/search-eval.ts --out $C/eval-out --dataset $d --dest $C/xerj-runs
node research/claude-mem/score.ts --qrels $C/qrels --baseline $C/baseline --xerj $C/xerj-runs
XERJ_URL=http://127.0.0.1:9300 bun research/claude-mem/latency.ts --claude-mem … --db $PG --out $C/eval-out --datasets …
node research/claude-mem/export-live.ts                                # live store → :9200, read-only SELECT via psql
node research/claude-mem/bench-live.ts --eval ~/wt/claude-mem-search-eval/<tree> --sample 20
```
