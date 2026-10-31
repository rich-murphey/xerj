// Shared helpers for the Phase 0 claude-mem retrieval comparison: a minimal
// xerj ES-compat client, the release vector format, and claude-mem's fusion.
import { readFileSync } from 'node:fs';

export const XERJ = process.env.XERJ_URL ?? 'http://127.0.0.1:9200';

// claude-mem fork, src/storage/postgres/observations.ts (a02f23770):
// RRF_K = 60, HYBRID_LEG_CANDIDATES = 100, HYBRID_NEAREST_PARTS = 400,
// MAX_PARTS_PER_OBSERVATION = MAX_FACT_PARTS (64) + 2.
export const RRF_K = 60;
export const LEG = 100;
export const NEAREST_PARTS = 400;
export const MAX_PARTS = 66;
export const DIMS = 384;

export async function xerj(method: string, path: string, body?: unknown, ndjson = false): Promise<any> {
  const res = await fetch(`${XERJ}${path}`, {
    method,
    headers: { 'content-type': ndjson ? 'application/x-ndjson' : 'application/json' },
    body: body === undefined ? undefined : ndjson ? (body as string) : JSON.stringify(body),
  });
  const text = await res.text();
  const json = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(`${method} ${path}: ${res.status} ${text.slice(0, 500)}`);
  return json;
}

/**
 * One index per corpus: a part index (one doc per observation part) and an
 * observation index for BM25. English analysis (stop words, Snowball) is set
 * as the index default: this xerj build ignores a per-field mapping
 * `analyzer` and indexes such a field with `standard`.
 */
export async function createIndexes(name: string): Promise<void> {
  for (const index of [`${name}-parts`, `${name}-obs`]) {
    await fetch(`${XERJ}/${index}`, { method: 'DELETE' });
  }
  const meta = {
    observation_id: { type: 'keyword' },
    project_id: { type: 'keyword' },
    kind: { type: 'keyword' },
    title: { type: 'text' },
    created_at: { type: 'date' },
  };
  const settings = { analysis: { analyzer: { default: { type: 'english' } } } };
  await xerj('PUT', `/${name}-parts`, {
    settings,
    mappings: {
      properties: {
        ...meta,
        part: { type: 'keyword' },
        text: { type: 'text' },
        vec: { type: 'dense_vector', dims: DIMS, similarity: 'cosine', index: true },
      },
    },
  });
  await xerj('PUT', `/${name}-obs`, {
    settings,
    mappings: { properties: { ...meta, content: { type: 'text' } } },
  });
}

/** Buffers bulk index actions and flushes them in batches. */
export class Bulk {
  private lines: string[] = [];
  private count = 0;
  private batch: number;
  constructor(batch = 2000) { this.batch = batch; }
  async add(index: string, id: string, doc: object): Promise<void> {
    this.lines.push(JSON.stringify({ index: { _index: index, _id: id } }), JSON.stringify(doc));
    if (++this.count >= this.batch) await this.flush();
  }
  async flush(): Promise<void> {
    if (this.count === 0) return;
    const res = await xerj('POST', '/_bulk', this.lines.join('\n') + '\n', true);
    if (res.errors) {
      const bad = res.items.find((i: any) => Object.values(i).some((v: any) => v.error));
      throw new Error(`bulk errors: ${JSON.stringify(bad).slice(0, 500)}`);
    }
    this.lines = [];
    this.count = 0;
  }
}

/** fp16 rows as fp32, renormalised, as claude-mem-search-eval's loadVectors reads the release files. */
export function loadVectors(path: string, dims = DIMS): Float32Array {
  const buf = readFileSync(path);
  const v = Float32Array.from(new Float16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2));
  for (let r = 0; r < v.length; r += dims) {
    let norm = 0;
    for (let d = 0; d < dims; d++) norm += v[r + d] * v[r + d];
    norm = Math.sqrt(norm);
    for (let d = 0; d < dims; d++) v[r + d] /= norm;
  }
  return v;
}

/** Ids in first-seen order: an observation ranked by its best part. */
export function collapse(hits: { _source: { observation_id: string } }[]): string[] {
  const seen = new Set<string>();
  for (const h of hits) seen.add(h._source.observation_id);
  return [...seen];
}

/** claude-mem's fusion: sum of 1/(k + rank) over the legs an observation appears in. */
export function rrf(legs: string[][], limit: number): string[] {
  const score = new Map<string, number>();
  const firstRank = new Map<string, number>();
  for (const leg of legs) {
    leg.forEach((id, i) => {
      score.set(id, (score.get(id) ?? 0) + 1 / (RRF_K + i + 1));
      firstRank.set(id, Math.min(firstRank.get(id) ?? Infinity, i));
    });
  }
  return [...score.keys()]
    .sort((a, b) => score.get(b)! - score.get(a)! || firstRank.get(a)! - firstRank.get(b)! || (a < b ? -1 : 1))
    .slice(0, limit);
}

export function percentile(sorted: number[], p: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
}

export type Scope = object; // an ES filter clause

/**
 * kNN leg as claude-mem's vector leg runs it: the nearest parts within the
 * scope, collapsed client-side to observations by their nearest part, read
 * again with four times the parts while fewer than LEG observations come back
 * and the part cutoff was reached.
 */
export async function knnLeg(name: string, scope: Scope, vector: ArrayLike<number>): Promise<string[]> {
  for (let parts = NEAREST_PARTS; ; parts = Math.min(LEG * MAX_PARTS, parts * 4)) {
    const res = await xerj('POST', `/${name}-parts/_search`, {
      size: parts,
      _source: ['observation_id'],
      knn: { field: 'vec', query_vector: Array.from(vector), k: parts, num_candidates: Math.max(parts, 800), filter: scope },
    });
    const ids = collapse(res.hits.hits);
    if (ids.length >= LEG || res.hits.hits.length < parts || parts >= LEG * MAX_PARTS) return ids.slice(0, LEG);
  }
}

/** BM25 leg over whole observations (claude-mem's keyword leg ranks observation content). */
export async function bm25ObsLeg(name: string, scope: Scope, query: string): Promise<string[]> {
  const res = await xerj('POST', `/${name}-obs/_search`, {
    size: LEG,
    _source: ['observation_id'],
    query: { bool: { filter: [scope], must: [{ match: { content: query } }] } },
  });
  return res.hits.hits.map((h: any) => h._source.observation_id);
}

/** BM25 over parts, collapsed to observations by their best part. */
export async function bm25PartLeg(name: string, scope: Scope, query: string): Promise<string[]> {
  const res = await xerj('POST', `/${name}-parts/_search`, {
    size: NEAREST_PARTS,
    _source: ['observation_id'],
    query: { bool: { filter: [scope], must: [{ match: { text: query } }] } },
  });
  return collapse(res.hits.hits).slice(0, LEG);
}

export interface ArmResult { ids: string[]; ms: number }

/** Every arm for one query; each arm timed on its own, the hybrid with its legs run concurrently. */
export async function runArms(name: string, scope: Scope, text: string, vector: ArrayLike<number>, limit = 10) {
  const timed = async (f: () => Promise<string[]>): Promise<ArmResult> => {
    const t0 = performance.now();
    const ids = await f();
    return { ids, ms: performance.now() - t0 };
  };
  const out: Record<string, ArmResult> = {};
  out.knn = await timed(async () => (await knnLeg(name, scope, vector)).slice(0, limit));
  out['bm25-obs'] = await timed(async () => (await bm25ObsLeg(name, scope, text)).slice(0, limit));
  out['bm25-part'] = await timed(async () => (await bm25PartLeg(name, scope, text)).slice(0, limit));
  out.hybrid = await timed(async () => {
    const [k, b] = await Promise.all([knnLeg(name, scope, vector), bm25ObsLeg(name, scope, text)]);
    return rrf([b, k], limit);
  });
  out['hybrid-part'] = await timed(async () => {
    const [k, b] = await Promise.all([knnLeg(name, scope, vector), bm25PartLeg(name, scope, text)]);
    return rrf([b, k], limit);
  });
  return out;
}

/**
 * Lines of a stream split on '\n' only. node:readline also breaks at U+2028
 * and U+2029, which JSON leaves unescaped inside strings.
 */
export async function* lines(input: AsyncIterable<Buffer | string>): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let rest = '';
  for await (const chunk of input) {
    rest += typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true });
    let nl: number;
    while ((nl = rest.indexOf('\n')) >= 0) {
      yield rest.slice(0, nl);
      rest = rest.slice(nl + 1);
    }
  }
  rest += decoder.decode();
  if (rest.length > 0) yield rest;
}
