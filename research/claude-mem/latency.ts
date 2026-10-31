// Quiet-host latency pass, pgvector and xerj interleaved query by query, one
// search at a time. Query i runs one arm only (keyword, vector, hybrid in
// rotation), on both engines, so no xerj request repeats one its query-result
// cache already holds; an arm stops at --per-arm samples or 120 s of xerj time.
// Searches: claude-mem's own PostgresObservationRepository.search
// against the schemas claude-mem-search-eval's server-search.ts --mode scoped
// left loaded (cmeval_<dataset>_scoped_minilm_l6), and the xerj arms over
// the same release vectors. Runs under bun, which claude-mem's source needs.
//
//   bun research/claude-mem/latency.ts --claude-mem ~/wt/claude-mem/<tree> \
//     --db postgres://postgres:t@127.0.0.1:55480/postgres --out <eval out dir> [--per-arm 100] [--datasets a,b]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { bm25ObsLeg, DIMS, knnLeg, loadVectors, percentile, rrf } from './lib.ts';

const { values } = parseArgs({
  options: {
    'claude-mem': { type: 'string' },
    db: { type: 'string' },
    out: { type: 'string' },
    'per-arm': { type: 'string', default: '100' },
    datasets: { type: 'string', default: 'longmemeval-s,memsearch-zh,memsearch-en,miracl-en,miracl-zh,miracl-ko,miracl-ja' },
  },
});
const claudeMem = resolve(values['claude-mem']!);
const { PostgresObservationRepository } = await import(`${claudeMem}/src/storage/postgres/index.ts`);
const { LocalEmbedder } = await import(`${claudeMem}/src/services/vector/LocalEmbedder.ts`);
const { EMBEDDING_MODELS } = await import(`${claudeMem}/src/services/vector/models.ts`);
const pg = (await import(`${claudeMem}/node_modules/pg/lib/index.js`)).default;
const modelId = new LocalEmbedder(EMBEDDING_MODELS['all-MiniLM-L6-v2']).modelId;
const DATASETS = values.datasets!.split(',');
const PER_ARM = Number(values['per-arm']);
const WARMUP = 6;
const ARMS = ['keyword', 'vector', 'hybrid'] as const;

let md = `| Dataset | Arm | n | pg p50 ms | pg p95 ms | xerj p50 ms | xerj p95 ms | xerj empty |\n|---|---|---|---|---|---|---|---|\n`;
for (const d of DATASETS) {
  const schema = `cmeval_${d}_scoped_minilm_l6`.replace(/-/g, '_');
  const pool = new pg.Pool({ connectionString: values.db, options: `-c search_path=${schema},public`, max: 2 });
  const repo = new PostgresObservationRepository(pool);
  const projects = new Map<string, string>((await pool.query('SELECT name, id FROM projects')).rows.map((r: any) => [r.name, r.id]));
  const teamId = (await pool.query('SELECT id FROM teams LIMIT 1')).rows[0].id;
  const dir = `${values.out}/${d}`;
  const queries = readFileSync(`${dir}/scope.jsonl`, 'utf8').trim().split('\n').map(l => JSON.parse(l));
  const vectors = loadVectors(`${dir}/minilm-l6/queries.f16.bin`);
  const ms: Record<string, { pg: number[]; xerj: number[]; spent: number; empty: number }> = {};
  for (const arm of ARMS) ms[arm] = { pg: [], xerj: [], spent: 0, empty: 0 };
  for (let i = 0; i < queries.length; i++) {
    const arm = ARMS[i % 3];
    const m = ms[arm];
    if (m.pg.length >= PER_ARM || m.spent > 120_000) {
      if (ARMS.every(a => ms[a].pg.length >= PER_ARM || ms[a].spent > 120_000)) break;
      continue;
    }
    const q = queries[i];
    const vector = vectors.subarray(i * DIMS, (i + 1) * DIMS);
    const projectId = projects.get(q.candidates ? q.id : q.project)!;
    const embedding = { vector, model: modelId };
    const input = arm === 'keyword' ? { query: q.text }
      : arm === 'vector' ? { query: 'the', queryEmbedding: embedding }
      : { query: q.text, queryEmbedding: embedding };
    let t0 = performance.now();
    await repo.search({ projectId, teamId, limit: 10, ...input });
    const pgMs = performance.now() - t0;
    const scope = q.candidates ? { terms: { observation_id: q.candidates } } : { term: { project_id: q.project } };
    t0 = performance.now();
    let ids: string[] = [];
    try {
      if (arm === 'keyword') ids = await bm25ObsLeg(`cm-${d}`, scope, q.text);
      else if (arm === 'vector') ids = await knnLeg(`cm-${d}`, scope, vector);
      else ids = rrf(await Promise.all([bm25ObsLeg(`cm-${d}`, scope, q.text), knnLeg(`cm-${d}`, scope, vector)]), 10);
    } catch (e) {
      console.error(`${d} ${q.id} ${arm}: ${String(e).slice(0, 200)}`);
    }
    const xMs = performance.now() - t0;
    if (i < WARMUP * 3) continue;
    if (ids.length === 0) m.empty++;
    m.pg.push(pgMs); m.xerj.push(xMs); m.spent += xMs;
  }
  await pool.end();
  for (const [arm, { pg: p, xerj: x, empty }] of Object.entries(ms)) {
    if (p.length === 0) continue;
    p.sort((a, b) => a - b); x.sort((a, b) => a - b);
    md += `| ${d} | ${arm} | ${p.length} | ${percentile(p, 50).toFixed(1)} | ${percentile(p, 95).toFixed(1)} | ` +
      `${percentile(x, 50).toFixed(1)} | ${percentile(x, 95).toFixed(1)} | ${empty} |\n`;
  }
  console.error(`${d} done`);
}
console.log(md);
