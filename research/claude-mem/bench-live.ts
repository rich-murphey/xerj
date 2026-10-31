// Two checks on the live store, read-only against Postgres:
//
// 1. Query embeddings are byte-compatible with the stored ones: re-embed a
//    sample of stored part texts with claude-mem-search-eval's encoder
//    (MiniLM fp32, masked mean, L2, 256 tokens with [SEP] restored) and
//    compare each with its stored vector by cosine.
// 2. Latency on the full live index in xerj: each sampled observation's title
//    is a query, embedded the same way, searched within its project, every
//    arm timed; hit@5 of the observation the title came from is reported as a
//    known-item sanity check, not as an accuracy measure.
//
//   node research/claude-mem/bench-live.ts --eval ~/wt/claude-mem-search-eval/<tree> [--sample 500] [--dest dir]
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { partTexts, streamObservations } from './export-live.ts';
import { percentile, runArms } from './lib.ts';

const { values } = parseArgs({
  options: {
    eval: { type: 'string' },
    sample: { type: 'string', default: '500' },
    container: { type: 'string', default: 'claude-mem-server-postgres-1' },
    name: { type: 'string', default: 'cm-live' },
    model: { type: 'string', default: 'Xenova/all-MiniLM-L6-v2/fp32/mean/l2/256' },
    dest: { type: 'string' },
  },
});
const evalDir = resolve(values.eval!);
const { loadEncoder } = await import(`${evalDir}/src/encoder.ts`);
const { modelByKey } = await import(`${evalDir}/src/models.ts`);
const encode = await loadEncoder(modelByKey('minilm-l6'));

// A deterministic sample: the observations whose md5(id) sorts first.
const n = Number(values.sample);
const sample: any[] = [];
for await (const line of streamObservations(values.container!, values.model!,
  `o.id IN (SELECT id FROM observations ORDER BY md5(id) LIMIT ${n})`)) sample.push(JSON.parse(line));

// 1. Stored-vector check over every part of the first 60 sampled observations.
const checks: { text: string; vec: number[]; part: string }[] = [];
for (const o of sample.slice(0, 60)) {
  const texts = partTexts(o.kind, o.metadata ?? {}, o.content);
  for (const p of o.parts) if (texts.has(p.part)) checks.push({ text: texts.get(p.part)!, vec: p.vec, part: p.part });
}
const reembedded = await encode(checks.map(c => c.text));
const cos = checks.map((c, i) => {
  let dot = 0, n2 = 0;
  for (let d = 0; d < c.vec.length; d++) { dot += c.vec[d] * reembedded[i][d]; n2 += c.vec[d] * c.vec[d]; }
  return dot / Math.sqrt(n2);
}).sort((a, b) => a - b);
console.log(`embedding check: ${cos.length} parts, cosine min ${cos[0].toFixed(6)}, p50 ${percentile(cos, 50).toFixed(6)}, ` +
  `${cos.filter(c => c >= 0.9999).length} of ${cos.length} >= 0.9999`);

// 2. Title queries over the live index.
const queries = sample.filter(o => typeof o.metadata?.title === 'string' && o.metadata.title.trim().length > 0);
const qv = await encode(queries.map(o => o.metadata.title));
const ms: Record<string, number[]> = {}, hit: Record<string, number> = {};
for (const [i, o] of queries.entries()) {
  const arms = await runArms(values.name!, { term: { project_id: o.project_id } }, o.metadata.title, qv[i]);
  for (const [arm, r] of Object.entries(arms)) {
    (ms[arm] ??= []).push(r.ms);
    hit[arm] = (hit[arm] ?? 0) + (r.ids.slice(0, 5).includes(o.id) ? 1 : 0);
  }
}
let md = `Live store, ${queries.length} title queries, each within its project.\n\n| Arm | known-item hit@5 | p50 ms | p95 ms |\n|---|---|---|---|\n`;
for (const arm of Object.keys(ms)) {
  const s = ms[arm].sort((a, b) => a - b);
  md += `| ${arm} | ${(hit[arm] / queries.length).toFixed(3)} | ${percentile(s, 50).toFixed(1)} | ${percentile(s, 95).toFixed(1)} |\n`;
}
console.log(md);
if (values.dest) writeFileSync(`${values.dest}/live-bench.md`, md);
