# Making xerj's exact kNN path answer at claude-mem scale

Compiled 2026-10-31 on `dv` against xerj `be2468ca` (fork main). Research
and a standalone bench only; no engine code changed. Follows
[phase0-results.md](phase0-results.md).

**What "more accurate" means here.** An exact scan is exact against the
stored vectors by construction. Its inaccuracy today is operational: from
179k parts it passes the 30 s deadline and the caller gets **0 hits**, and
unfiltered kNN also returned 0 hits at 81k and 179k. So the goal is (1) the
true top-k for the filter every time, never a silent short or empty answer,
(2) fast enough that no deadline cuts it, and (3) deliberate choices inside
the scan: f32 vs quantized scoring, quantized prefilter plus exact rescore,
part-to-observation aggregation, deterministic ties, and normalisation.
Retrieval-quality techniques (whitening, fusion, part aggregation as a
ranking signal, PRF, MMR) belong to a separate run in claude-mem-search-eval
and are not repeated here.

**Answer.** The arithmetic is not the problem. On the real MiniLM vectors,
xerj's own sequential-f64 dot product over 300k × 384 takes 178 ms on one
core and 28 ms on 16. The 15–30 s goes to **decoding every stored document
to `serde_json::Value` and running the filter on JSON, every query**,
because a filter disables the column-only scan. Evaluate the filter from
doc values into a per-segment bitmap and scan the existing vector column,
and a project-scoped kNN at 300k drops to ~20 ms (f32 SIMD, 16 threads) or
~8 ms (int8 plus exact rescore, recall@400 = 1.000). That is pgvector's
range, and exact.

## 1. The exact path today, cost by cost

All paths are `engine/crates/xerj-engine/src/index.rs` unless named.

| Step | Where | Cost at 300k parts |
|---|---|---|
| Filtered kNN never tries HNSW | `plain = filter_opt.is_none() && …` 24103–24106 | Every claude-mem query is project-scoped, so every query is exact |
| Filter disables the column-only scan | `columns_only && filter.is_none() && !use_sq8` 19365 | The cached f32 column (#1091, [vector_column.rs](../../engine/crates/xerj-engine/src/vector_column.rs)) is bypassed |
| Whole segment decoded to JSON per query | `stored_values_for_async` 19533 → 34805 | Log at 300k: `dec=20167ms scan=9677ms walked=184180` (eval2.log, 16:05:58); 445 MB read per query at 81k against 124 MB of raw f32. Parsed `Value`s for 300k docs (text plus 384 numbers each) do not fit the hydration budget, so each query decodes again |
| Filter run on JSON, per candidate | `KnnFilterPlan::matches` 46435 | A new `serde_json::Map` plus `doc_matches_query_typed` per document, after the decode |
| Scoring | `QueryScorer::score` 45320 → `dot_f64` [vector_column.rs:424](../../engine/crates/xerj-engine/src/vector_column.rs) | Strictly sequential f64, which no compiler vectorizes. Measured 178 ms per query on 1 core over 300k. Small next to the decode |
| Ranking | `rank_knn_pool` 46165 | Full `sort_by` of every scored candidate. O(n log n) with no bounded heap. Tie order = candidate order (memtable, then segments in snapshot order, then position) |
| Parallelism | the whole scan is one async task with `yield_now` every 128 docs (18370) | One core per query |
| Deadline | `exact_scan_checkpoint` 18370, `timeout_at` on segment loads 19480 and 19533 | Default `timeout_ms.unwrap_or(30_000)` 18464 (also 20262, 20957, 21323, 23273, 23649) |

### Partial results are flagged, but the flagged answer is never delivered

- The scan does the right thing: it sets `result.timed_out = true` and
  `relation = Gte` (19981–19984) on the candidates it reached.
- But `search_at_generation` wraps the same work in
  `tokio::time::timeout_at(request_deadline, search_fut)` (23713–23723), at
  the **same instant** as the scan's own deadline. The scan reaches its
  checkpoint at the deadline and still has to score, sort and hydrate, so
  the outer timeout fires first. The dropped future is replaced by
  `hits: [], total: 0 (gte), timed_out: true` (23784–23817).
- The ES-compat layer does surface it: `timed_out` goes from the result
  (es_compat.rs 16231) to the response body (19373–19395). The empty answer
  is silent only to the client. [lib.ts](lib.ts) `knnLeg` never reads
  `timed_out`, and its widening loop takes `hits.length < parts` to mean
  "corpus exhausted".

## 2. Why unfiltered kNN also returned 0 hits

This is inferred from the code and agrees with every observation. It was
not reproduced here.

1. `run_knn_hnsw` (18151) gets past its gates when `_cat/ann` says the graph
   is present and covered. The same `id_map.len() == vector_doc_count && !stale`
   test is used in both places (18213 and 18084).
2. It then hydrates the hits **one id at a time** through
   `get_document_uncounted` (18269–18278). claude-mem's leg asks for k = 400
   parts (`NEAREST_PARTS`, lib.ts:12), and `ef` is at least 800
   (`HNSW_EF_FLOOR`, 45350).
3. Each hydration needs the segment's `StoredSlices`. When the hydration
   budget has no room for a segment, `stored_slices_for_with` (35973)
   builds them **transiently**: it decompresses the whole stored section and
   drops it after that one id. If the slices or the id map are missing, the
   legacy loop decodes the whole section per id (the comment at 22184 says
   "~470 ms/id at 100 k docs"). Cost is O(k × segment).
4. The deadline passes, `run_knn_hnsw` returns `None` (18272), and the
   request falls to the exact scan with a spent deadline. The outer
   `timeout_at` then returns 0 hits.

Consistent with this:

- 43k parts answers in 110–270 ms.
- A merge to one big segment made it worse, not better.
- Unfiltered kNN "returns 0 hits after a background merge until restart"
  (claude-mem observation, 2026-10-31): a new merged segment has nothing
  cached.
- With `XERJ_SEGMENT_HYDRATION_CACHE_MB=8192 XERJ_READ_TRANSIENT_MB=6144`,
  the Phase 2–3 server answered unfiltered kNN on all 300k in 0.1–0.6 s.

The fix is the same as for the exact path: hydrate the top k **by segment
position in one row-selective pass** (`hydrate_segment_rows`, 18832, which
the column scan already uses), not by id through the slice cache.

## 3. How others do exact and filtered kNN

Each mechanism was checked at the pinned source on 2026-10-31.

- **Lucene 10.6** `AbstractKnnVectorQuery`, per segment:
  - If `acceptDocs.cost() <= perLeafTopK` it runs `exactSearch`.
  - Otherwise it runs an approximate search with `visitedLimit = cost+1`. It
    falls back to exact if the graph search was cut short or found fewer
    than k.
  - [L278–297](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/search/AbstractKnnVectorQuery.java#L278-L297).
  - `exactSearch` scores in bulk batches and skips a batch whose `maxScore`
    is below the current k-th. On timeout it sets `GREATER_THAN_OR_EQUAL_TO`,
    so the answer is flagged, not silent
    ([L356–410](https://github.com/apache/lucene/blob/releases/lucene/10.6.0/lucene/core/src/java/org/apache/lucene/search/AbstractKnnVectorQuery.java#L356-L410)).
  - Filtered kNN arrived in 9.1 (LUCENE-10382), timeouts in exact and graph
    search in 9.11 (GITHUB#13202), bulk scorers in 10.4 (GITHUB#15171).
- **Elasticsearch:**
  - A filter inside `knn` is a pre-filter. A `bool.filter` beside it is a
    post-filter and "can return fewer than k".
  - Indices created in 9.1 or later default to the `acorn` filter heuristic
    ([filtered kNN](https://www.elastic.co/docs/solutions/search/vector/knn/filtered-knn-search.md),
    [exact kNN](https://www.elastic.co/docs/solutions/search/vector/knn/exact-knn.md)).
  - `timed_out` means "results may be partial or empty".
    `allow_partial_search_results=false` turns that into an error
    ([search API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-search)).
- **OpenSearch k-NN 3.1** `isFilteredExactSearchPreferred` picks exact search
  in this order:
  1. When `filterCount <= k`.
  2. Otherwise, when `index.knn.advanced.filtered_exact_search_threshold`
     (default unset) is at least `filterCount`.
  3. Otherwise, when `filterCount × dim <= 2,048,000`.

  After the ANN pass it re-runs exact search if ANN returned fewer than k
  hits although more than k documents passed the filter
  ([KNNWeight.java L567–599](https://github.com/opensearch-project/k-NN/blob/3.1.0.0/src/main/java/org/opensearch/knn/index/query/KNNWeight.java#L567-L599)).
  For 384 dims that rule means exact search below 5,333 matches.
- **Qdrant 1.20:**
  - `full_scan_threshold_kb` (default 10000) becomes a point count.
  - With a filter, `estimate_cardinality` gives `{min, exp, max}`. It scans
    plainly when `max` is below the threshold, uses HNSW when `min` is above
    it, and samples in between.
  - ACORN-1 applies when `exp/available <= 0.4` (added in v1.16)
    ([dispatch.rs](https://github.com/qdrant/qdrant/blob/v1.20.0/lib/segment/src/index/hnsw_index/hnsw/read_view/dispatch.rs#L60-L190),
    [hnsw.rs](https://github.com/qdrant/qdrant/blob/v1.20.0/lib/segment/src/index/hnsw_index/hnsw.rs#L194-L203)).
- **pgvector 0.8.0** (2024-10-30) iterative index scans: `hnsw.iterative_scan`,
  `hnsw.max_scan_tuples` (default 20000). At the cap it returns fewer rows
  with no signal ([README](https://github.com/pgvector/pgvector/blob/v0.8.0/README.md#iterative-index-scans)).
- **Faiss 1.15.1:**
  - `IndexFlat::search` honours an `IDSelector` (`params->sel`), but a
    selector forces the sequential path instead of BLAS
    ([distances.cpp L571–609](https://github.com/facebookresearch/faiss/blob/v1.15.1/faiss/utils/distances.cpp#L571-L609)).
    A fast filtered flat scan needs its own kernel, as below.
  - `IndexRefineFlat` takes `k × k_factor` candidates from a quantized index
    and rescores them exactly
    ([IndexRefine.h](https://github.com/facebookresearch/faiss/blob/v1.15.1/faiss/IndexRefine.h#L14-L36)).
- **LanceDB 0.41:**
  - Pre-filter is the default.
  - `refine_factor(n)` reranks n·k candidates with full vectors.
  - `bypass_vector_index()` gives a flat scan, which it recommends as ground
    truth.
  - `fast_search()` silently skips unindexed rows
    ([query.py](https://github.com/lancedb/lancedb/blob/v0.41.0/python/python/lancedb/query.py#L1395-L1422)).
- **usearch 2.26** `search(..., exact=True)` is an exhaustive SIMD scan through
  NumKong ([README](https://github.com/unum-cloud/usearch/blob/v2.26.6/README.md#exact-vs-approximate-search)).
  NumKong was called SimSIMD until v7.0.0 (2026-03-17). Its AVX2
  `nk_dot_f32_haswell` accumulates in f64
  ([haswell.h](https://github.com/ashvardanian/numkong/blob/v7.8.6/include/numkong/dot/haswell.h#L149-L158)).
- **In-graph filtering**, the alternative to brute force:
  - ACORN: Patel, Kraft, Guestrin, Zaharia, SIGMOD 2024,
    [arXiv:2403.04871](https://arxiv.org/abs/2403.04871).
  - Filtered-DiskANN: Gollapudi et al., WWW 2023,
    [doi:10.1145/3543507.3583552](https://doi.org/10.1145/3543507.3583552).
- **Quantization:**
  - RaBitQ: Gao & Long, SIGMOD 2024,
    [arXiv:2405.12497](https://arxiv.org/abs/2405.12497).
  - Hugging Face's embedding-quantization post measures NDCG@10 retained
    with 4× rescoring, not recall. For 384-d all-MiniLM-L6-v2, binary plus
    rescore keeps 93.8%, int8 without rescore keeps 90.8%, and int8 with
    rescore reaches about 99%
    ([post](https://github.com/huggingface/blog/blob/main/embedding-quantization.md)).

**What to copy from them.** Lucene's cutover (exact search when the filter
is small, graph search otherwise, and fall back to exact whenever the graph
search comes up short) plus its flagged partial answer. pgvector, LanceDB
`fast_search`, OpenSearch post-filtering and Qdrant without ACORN all
return short answers silently, which is the failure seen here.

## 4. Bench: packed column, bitmap filter, bounded heap

[bench-bruteforce/](bench-bruteforce/src/main.rs), raw output in
[results-dv.tsv](bench-bruteforce/results-dv.tsv).

**Setup:**

- Data: 300,306 real MiniLM-L6 rows, the live store's count. These are
  LongMemEval-S (199,509) plus the first 100,797 MIRACL-en rows, from the
  release files in `~/.cache/xerj-phase0/eval-out`, renormalised as
  `loadVectors` does.
- Queries: 200 (100 from each dataset).
- Layout: 16 contiguous segments, each with its own roaring bitmap.
  Filters pass 100%, 10% or 1%; the 10% and 1% filters are uniform random,
  the worst case for locality.
- Top-k: one heap per segment, ordered by (score desc, row asc), merged
  after the segments run in parallel on rayon.
- Ground truth: xerj's sequential-f64 arithmetic.
- Machine: EPYC 7351P (Zen 1, AVX2/FMA/F16C, no AVX-512). The box was
  shared at load average 39–44 on 32 threads (ck indexing, eval runs), so
  the 16-thread numbers are pessimistic.
- Memory per encoding: f32 440 MiB, f16 220 MiB, i8 110 MiB, 1-bit
  13.7 MiB. Peak RSS was 804 MiB with all four held at once.

p50 / p95 ms per query. "rec" is recall@k against xerj's f64 ranking.
"=" counts queries whose ranked list is identical to it, out of 200.

**k = 400** (claude-mem's part leg):

| Method | 100% 1 thr | 100% 16 thr | 10% 16 thr | 1% 1 thr | rec | = (100%) |
|---|---|---|---|---|---|---|
| f64 sequential (xerj's arithmetic) | 181 / 226 | 28 / 38 | 7.5 / 11.5 | 2.9 / 3.3 | 1 | 200 |
| f32 sequential | 151 / 182 | 26 / 36 | 6.0 / 11.0 | 2.4 / 2.9 | 1.0000 | 184 |
| f32 AVX2 FMA | 42 / 60 | **20 / 26** | 4.9 / 6.9 | 1.2 / 1.6 | 1.0000 | 195 |
| f16 AVX2 (F16C) | 42 / 50 | 13 / 21 | 4.1 / 6.1 | 0.8 / 1.2 | 0.9999 | 2 |
| i8 → top 400 → f32 rescore | 20 / 25 | 8.3 / 12 | 2.9 / 5.2 | 0.7 / 0.9 | 0.9911 | 2 |
| **i8 → top 800 → f32 rescore** | 24 / 30 | **7.7 / 10** | 3.7 / 5.8 | 0.9 / 1.2 | **1.0000** | 195 |
| i8 → top 4000 → f32 rescore | 38 / 46 | 13 / 19 | 6.5 / 9.6 | 1.8 / 2.2 | 1.0000 | 195 |
| 1-bit → top 4000 → f32 rescore | 22 / 24 | 8.8 / 11 | 5.6 / 8.7 | 1.5 / 1.7 | 0.9740 | 34 |
| 1-bit → top 16000 → f32 rescore | 38 / 45 | 26 / 31 | 9.4 / 14 | 1.4 / 2.4 | 0.9984 | 148 |

**k = 10:**

| Method | 100% 1 thr | 100% 16 thr | 10% 16 thr | 1% 1 thr | rec (min) |
|---|---|---|---|---|---|
| f64 sequential | 178 / 256 | 30 / 41 | 6.1 / 9.2 | 2.5 / 3.1 | 1 |
| f32 AVX2 | 41 / 54 | 23 / 27 | 4.1 / 5.9 | 1.4 / 1.7 | 1.0000 (1.0) |
| f16 AVX2 | 40 / 52 | 12 / 16 | 3.0 / 4.8 | 0.9 / 1.2 | 1.0000 at 100%, 0.9995 (0.9) at 10% |
| i8 → top 10 → rescore | 17 / 23 | 6.3 / 8.7 | 1.9 / 3.7 | 0.3 / 0.4 | 0.9890 (0.9) |
| i8 → top 20 → rescore | 18 / 20 | 6.0 / 9.5 | 2.2 / 4.3 | 0.3 / 0.4 | **1.0000 (1.0)** |
| 1-bit → top 400 → rescore | 7.8 / 10 | 2.6 / 4.9 | 1.9 / 3.9 | 0.5 / 1.3 | 0.9945 (0.8) |

**Readings:**

- **Rich's hypothesis holds.** Over a packed column with a roaring filter,
  a bounded heap and parallel segments, 300k × 384 is answered exactly in
  20–28 ms at 16 threads, even on a loaded box, and in 1–3 ms at 1% on one
  thread. That is 1000× faster than xerj today on the same data, and
  pgvector's range (17–36 ms p50 in Phase 0).
- **Scan speed is memory-bound beyond about 4 threads.** f32 at 16 threads
  reads ~22 GB/s. f16 and i8 win by moving fewer bytes, not by doing less
  arithmetic.
- **Exactness of the arithmetic.**
  - SIMD f32 reassociation gives the same top-400 set every time but
    reorders near-ties in 5 of 200 queries.
  - To keep xerj's "bit-identical to the f64 oracle" contract
    (vector_column.rs header), rescore the final k' candidates in
    sequential f64. That costs 400 × 384 MACs, which is negligible.
  - With that rescore, f32 SIMD, i8 and f16 prefilters all return exactly
    the oracle's list whenever the prefilter recall is 1.
- **Rescore depth.**
  - int8 with symmetric global scale (127/max|x|) and a 2k rescore depth
    reached recall 1.000, min 1.0, on all 200 queries, at every
    selectivity, for k = 10 and k = 400.
  - Depth 1k loses about 1%.
  - 1-bit sign codes need 40k depth to pass 0.998 and still miss on some
    queries (minimum recall 0.96 at k = 400). That is too lossy for a
    384-d model. The Hugging Face numbers above agree.
  - int8 + 2k rescore is the best measured trade: 2.6× faster than f32 at
    16 threads, a quarter of the memory, and no measured recall loss. It
    is still not provably exact. Keep f32 (or f16) as the default exact
    scan, and offer int8 + rescore as the `scalar8` field's exact-ish mode.
- **Thread scaling at 1% is nil.** About 3,000 rows is under a millisecond
  on one core, and rayon dispatch costs more. Pick parallelism from the
  bitmap's cardinality: one thread below about 50k rows.

## 5. Proposed change to xerj

Ordered by payoff. Effort is engineer-days, tests included.

1. **Filter from doc values into a bitmap, then scan the column** (about 3 d).
   - In `run_knn_exact_scan`, compile `filter` with the leaf evaluators
     `scored_columnar` already uses. These are `term` and `terms` on
     `keyword` ordinals (`FilterEval::Kw` and its builder, 33992–34066),
     `exists` from the presence bitmap, and numeric equality. The output is
     one `RoaringBitmap` per segment (roaring is already a dependency of
     xerj-storage and xerj-fts).
   - When every leaf compiles, run the existing `columns_only` scan with the
     bitmap: remove `filter.is_none()` at 19365, and skip
     `stored_values_for_async`.
   - Keep `KnnFilterPlan` as the fallback for filter shapes that do not
     compile.
   - claude-mem's scopes (`term project_id`, `terms observation_id`) all
     compile. This one change removes the 20 s decode.
2. **Bounded heap, SIMD kernel, parallel segments** (about 3 d).
   - Replace `rank_knn_pool`'s full sort with a k-heap per segment, ordered
     by (score desc, segment ordinal, position). That equals today's tie
     order (candidate order), so results stay deterministic.
   - Score with an AVX2/FMA f32 kernel and keep a scalar fallback, since
     releases must run without AVX2 (#2253). Then rescore the merged top k
     with `dot_f64`, so scores stay bit-identical to today's.
   - Run segments on the engine's background pool, not on the tokio
     worker.
   - `numkong` 7.8 has the kernels, but its Haswell f32 dot accumulates in
     f64. Measure it before adopting it, or write the about 40 lines in the
     bench.
3. **Make the vector column a segment section** (about 5 d).
   - Today the column is built lazily and cached under a byte budget
     (`publish_current`, 19599), so a cold start or a merge pays a decode,
     and over budget it is rebuilt every query. That is the same failure as
     §2.
   - Write packed f32 (or f16) at segment build and mmap it. It costs
     440 MB on disk at 300k for f32, or 220 MB for f16.
4. **Never return a silent partial** (about 1 d).
   - Give the inner scan a deadline earlier than the outer `timeout_at`
     (23713–23723), for example the outer deadline minus a hydration
     margin, so the flagged partial answer actually reaches the caller
     instead of `hits: []`.
   - Honour `allow_partial_search_results=false` as Elasticsearch does: a
     timed-out kNN becomes an error, not a 200.
   - In [lib.ts](lib.ts), read `timed_out` and treat it as a failure. Never
     treat a short answer as "exhausted".
   - At items 1–3 speed the deadline is a backstop. Up to a few million
     vectors the exact scan should simply finish.
5. **Hydrate winners by position, not by id** (about 1 d).
   - Route `run_knn_hnsw`'s hydration (18269) through `hydrate_segment_rows`
     (18832), grouped by segment, as the column scan does.
   - This fixes §2's unfiltered 0-hit case without the 8 GiB cache
     override.
6. **Selectivity cutover to filtered HNSW** (about 5 d, plus a recall eval).
   - With the bitmap's cardinality known, choose exact search when
     `card <= max(k, threshold)`, as OpenSearch and Lucene do. Otherwise use
     `HnswIndex::search_filtered` (xerj-vector/src/hnsw.rs:1249, which has
     no engine caller) with the bitmap as its predicate.
   - Fall back to exact search whenever the graph search visits more than
     `card` nodes or returns fewer than k, as Lucene does.
   - This is upstream [xerj-org/xerj#147](https://github.com/xerj-org/xerj/issues/147),
     closed `not_planned` on 2026-08-03 and "consolidated into internal
     tracking". The issue's own bench shows the graph losing at 1%
     selectivity (22 ms vs 0.5 ms).
   - For claude-mem this step is optional: at 300k, items 1–3 make the
     exact scan as fast as pgvector's HNSW, and exact.
7. **Collapse parts to observations in the engine** (about 4 d).
   - Score parts, keep the best part per `observation_id` (keyword ordinal
     column) in each segment, merge across segments by max, then cut to the
     top 100 observations.
   - That is claude-mem's "nearest part" rule, done exactly in one pass.
     The client no longer has to widen ×4 and re-query, which is what made
     LongMemEval cost 277 s.
   - Collapse after scoring and before the cut. Collapsing before the cut
     (top 400 parts, then group) is what forces the widening loop.

**Normalisation.**

- claude-mem stores unit vectors. Keep the mapping `cosine`. The column
  already stores each document's norm (`VectorEntry.norm`), so cosine costs
  the same as a dot product.
- f16 or i8 storage shifts norms by about 1e-3. Normalise before
  quantizing, as the `scalar8` path does (19645), and rescore against the
  f32 copy.

**Total for items 1–5**, which get claude-mem's scoped kNN to an exact
~10–30 ms with no silent partials: about 2½ weeks. Items 6–7 add about
2 weeks.

## Upstream record (xerj-org/xerj)

No open issue or PR covers the filtered exact path, a selectivity cutover
or kNN partial flags. Related:

- Filtered HNSW and the selectivity cutover:
  [#147](https://github.com/xerj-org/xerj/issues/147) (closed not_planned).
- Exact scan ranks addresses, not clones:
  [#939](https://github.com/xerj-org/xerj/issues/939) and
  [PR #979](https://github.com/xerj-org/xerj/pull/979).
- Exact scan ranks from vector columns:
  [PR #1217](https://github.com/xerj-org/xerj/pull/1217) (#1091 in the
  code).
- Capture bracket: [#1220](https://github.com/xerj-org/xerj/issues/1220)
  and [#2473](https://github.com/xerj-org/xerj/issues/2473).
- Partial pages: [#1212](https://github.com/xerj-org/xerj/issues/1212) and
  [PR #1637](https://github.com/xerj-org/xerj/pull/1637).
- scalar8 codes: [#392](https://github.com/xerj-org/xerj/issues/392) and
  [PR #999](https://github.com/xerj-org/xerj/pull/999).
- No-AVX2 releases: [#2161](https://github.com/xerj-org/xerj/issues/2161).

## Reproduce

```sh
cd research/claude-mem/bench-bruteforce
E=~/.cache/xerj-phase0/eval-out
CARGO_TARGET_DIR=~/.cache/xerj-bench-target cargo run --release -- \
  --rows $E/longmemeval-s/minilm-l6/rows.f16.bin,$E/miracl-en/minilm-l6/rows.f16.bin \
  --queries $E/longmemeval-s/minilm-l6/queries.f16.bin,$E/miracl-en/minilm-l6/queries.f16.bin \
  --nq 200 --threads 16 > results.tsv      # ~6.5 min; --only f32 to run one method
```
