// Scores the pgvector baseline and the xerj arms on every dataset: hit@5,
// MRR@10 and R@5 over the eval's labels, p50/p95 search latency, and the
// paired difference of each xerj arm from its pgvector counterpart with a
// 95% bootstrap interval (2000 resamples over queries, seeded as the eval's).
//
//   node research/claude-mem/score.ts --qrels <dir> --baseline <dir> --xerj <dir>
import { existsSync, readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { percentile } from './lib.ts';

const { values } = parseArgs({ options: { qrels: { type: 'string' }, baseline: { type: 'string' }, xerj: { type: 'string' } } });
const DATASETS = ['longmemeval-s', 'memsearch-zh', 'memsearch-en', 'miracl-en', 'miracl-zh', 'miracl-ko', 'miracl-ja'];
// [label, file stem, pgvector counterpart label]
const RUNS: [string, (d: string) => string, string | null][] = [
  ['pg keyword', d => `${values.baseline}/${d}-scoped-keyword-only`, null],
  ['pg vector', d => `${values.baseline}/${d}-scoped-minilm-l6-vector`, null],
  ['pg hybrid', d => `${values.baseline}/${d}-scoped-minilm-l6-hybrid`, null],
  ['xerj bm25-obs', d => `${values.xerj}/${d}-xerj-bm25-obs`, 'pg keyword'],
  ['xerj bm25-part', d => `${values.xerj}/${d}-xerj-bm25-part`, 'pg keyword'],
  ['xerj knn', d => `${values.xerj}/${d}-xerj-knn`, 'pg vector'],
  ['xerj hybrid', d => `${values.xerj}/${d}-xerj-hybrid`, 'pg hybrid'],
  ['xerj hybrid-part', d => `${values.xerj}/${d}-xerj-hybrid-part`, 'pg hybrid'],
];

const readTsv = (path: string) => new Map(readFileSync(path, 'utf8').trim().split('\n').map(l => {
  const [id, ...rest] = l.split('\t');
  return [id, rest] as [string, string[]];
}));

function rng(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
function interval(v: number[]): [number, number] {
  const next = rng(1), means: number[] = [];
  for (let b = 0; b < 2000; b++) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[Math.floor(next() * v.length)];
    means.push(s / v.length);
  }
  means.sort((a, b) => a - b);
  return [means[50], means[1949]];
}
const f3 = (x: number) => x.toFixed(3);
const signed = (x: number) => `${x >= 0 ? '+' : ''}${f3(x)}`;

let table = '| Dataset | Run | hit@5 | MRR@10 | R@5 | p50 ms | p95 ms | Δhit@5 vs pg (95%) | ΔMRR@10 vs pg (95%) |\n|---|---|---|---|---|---|---|---|---|\n';
for (const d of DATASETS) {
  const qrels = readFileSync(`${values.qrels}/${d}.qrels.jsonl`, 'utf8').trim().split('\n')
    .map(l => JSON.parse(l) as { id: string; relevant: string[] });
  const per = new Map<string, { hit: number[]; mrr: number[]; r5: number[] }>();
  for (const [label, stem, base] of RUNS) {
    if (!existsSync(`${stem(d)}.tsv`)) continue;
    const ranked = readTsv(`${stem(d)}.tsv`);
    const ms = [...readTsv(`${stem(d)}.ms`).values()].map(v => Number(v[0])).sort((a, b) => a - b);
    const m = { hit: [] as number[], mrr: [] as number[], r5: [] as number[] };
    for (const q of qrels) {
      const docs = ranked.get(q.id);
      if (!docs) throw new Error(`${stem(d)}.tsv: no ranking for ${q.id}`);
      const rel = new Set(q.relevant);
      const top5 = docs.slice(0, 5).filter(x => rel.has(x)).length;
      const first = docs.slice(0, 10).findIndex(x => rel.has(x));
      m.hit.push(top5 > 0 ? 1 : 0);
      m.mrr.push(first < 0 ? 0 : 1 / (first + 1));
      m.r5.push(top5 / rel.size);
    }
    per.set(label, m);
    let diff = ' | ';
    if (base && per.has(base)) {
      const b = per.get(base)!;
      const dh = m.hit.map((x, i) => x - b.hit[i]), dm = m.mrr.map((x, i) => x - b.mrr[i]);
      const [hl, hh] = interval(dh), [ml, mh] = interval(dm);
      diff = `${signed(mean(dh))} (${signed(hl)}–${signed(hh)}) | ${signed(mean(dm))} (${signed(ml)}–${signed(mh)})`;
    }
    table += `| ${d} (${qrels.length}) | ${label} | ${f3(mean(m.hit))} | ${f3(mean(m.mrr))} | ${f3(mean(m.r5))} | ` +
      `${percentile(ms, 50).toFixed(1)} | ${percentile(ms, 95).toFixed(1)} | ${diff} |\n`;
  }
}
console.log(table);
