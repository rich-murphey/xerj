// Loads one claude-mem-search-eval dataset into xerj from the release files:
// the minilm-l6 vectors claude-mem's server search was scored with, and the
// rows and query scopes export-rows.ts writes. No text is embedded here.
//
//   node research/claude-mem/load-eval.ts --out <eval out dir> --dataset memsearch-en
//
// Index cm-<dataset>-parts: one doc per (document, part) with its vector.
// Index cm-<dataset>-obs: one doc per document, its parts joined by blank
// lines, as server-search.ts stores an observation's content for BM25.
import { createReadStream } from 'node:fs';
import { parseArgs } from 'node:util';
import { Bulk, createIndexes, DIMS, lines, loadVectors, xerj } from './lib.ts';

const { values } = parseArgs({ options: { out: { type: 'string' }, dataset: { type: 'string' } } });
const dir = `${values.out}/${values.dataset}`;
const name = `cm-${values.dataset}`;
const vectors = loadVectors(`${dir}/minilm-l6/rows.f16.bin`);

const started = performance.now();
await createIndexes(name);
const bulk = new Bulk(1000);
let row = 0;
let doc: { id: string; project: string; parts: string[] } | null = null;
const flushDoc = async () => {
  if (!doc) return;
  await bulk.add(`${name}-obs`, doc.id, { observation_id: doc.id, project_id: doc.project, content: doc.parts.join('\n\n') });
};
// rows.jsonl lists each document's parts consecutively, in part order.
for await (const line of lines(createReadStream(`${dir}/rows.jsonl`))) {
  const r = JSON.parse(line) as { doc: string; part: number; project: string; text: string };
  if (doc?.id !== r.doc) {
    await flushDoc();
    doc = { id: r.doc, project: r.project, parts: [] };
  }
  doc.parts.push(r.text);
  await bulk.add(`${name}-parts`, `${r.doc}#${r.part}`, {
    observation_id: r.doc,
    project_id: r.project,
    part: `part_${r.part}`,
    text: r.text,
    vec: Array.from(vectors.subarray(row * DIMS, (row + 1) * DIMS)),
  });
  row++;
}
await flushDoc();
await bulk.flush();
if (row * DIMS !== vectors.length) throw new Error(`${row} rows, ${vectors.length / DIMS} vectors`);
for (const index of [`${name}-parts`, `${name}-obs`]) await xerj('POST', `/${index}/_refresh`);
const count = (await xerj('GET', `/${name}-parts/_count`)).count;
console.log(`${values.dataset}: ${row} parts (${count} searchable) in ${((performance.now() - started) / 1000).toFixed(0)}s`);
