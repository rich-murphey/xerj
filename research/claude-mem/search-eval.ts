// Runs every eval query of one dataset against xerj and writes each arm's top
// 10 per query, in the form claude-mem-search-eval's eval.ts --rankings reads,
// with each search's milliseconds beside it.
//
//   node research/claude-mem/search-eval.ts --out <eval out dir> --dataset memsearch-en --dest <dir>
//
// Scope as server-search.ts --mode scoped gives it: LongMemEval's own
// haystack, otherwise the query's project.
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { DIMS, loadVectors, runArms } from './lib.ts';

const { values } = parseArgs({ options: { out: { type: 'string' }, dataset: { type: 'string' }, dest: { type: 'string' } } });
const dir = `${values.out}/${values.dataset}`;
const name = `cm-${values.dataset}`;
const queries = readFileSync(`${dir}/scope.jsonl`, 'utf8').trim().split('\n').map(l => JSON.parse(l)) as
  { id: string; project: string | null; candidates: string[] | null; text: string }[];
const vectors = loadVectors(`${dir}/minilm-l6/queries.f16.bin`);
if (vectors.length !== queries.length * DIMS) throw new Error('queries and vectors differ in count');

const ranked: Record<string, string[]> = {};
const ms: Record<string, string[]> = {};
const started = performance.now();
for (const [i, q] of queries.entries()) {
  const scope = q.candidates ? { terms: { observation_id: q.candidates } } : { term: { project_id: q.project } };
  const arms = await runArms(name, scope, q.text, vectors.subarray(i * DIMS, (i + 1) * DIMS));
  for (const [arm, r] of Object.entries(arms)) {
    (ranked[arm] ??= []).push([q.id, ...r.ids].join('\t'));
    (ms[arm] ??= []).push(`${q.id}\t${r.ms.toFixed(3)}`);
  }
}
for (const arm of Object.keys(ranked)) {
  writeFileSync(`${values.dest}/${values.dataset}-xerj-${arm}.tsv`, ranked[arm].join('\n') + '\n');
  writeFileSync(`${values.dest}/${values.dataset}-xerj-${arm}.ms`, ms[arm].join('\n') + '\n');
}
console.log(`${values.dataset}: ${queries.length} queries, ${Object.keys(ranked).length} arms, ${((performance.now() - started) / 1000).toFixed(0)}s`);
