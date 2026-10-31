// Writes each eval dataset's queries and relevant documents through
// claude-mem-search-eval's own loader (datasets.ts), so scoring here uses
// exactly the labels its results were scored with.
//
//   node research/claude-mem/qrels.ts --eval ~/wt/claude-mem-search-eval/<tree> --dest <dir>
//
// <dest>/<dataset>.qrels.jsonl: {id, relevant[]} per evaluated query.
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: { eval: { type: 'string' }, dest: { type: 'string' } } });
const evalDir = resolve(values.eval!);
const dest = resolve(values.dest!);
process.chdir(evalDir); // datasets.ts reads cache/ relative to its repo
const { DATASETS, loadDataset } = await import(`${evalDir}/src/datasets.ts`);
for (const key of DATASETS as string[]) {
  const ds = await loadDataset(key);
  const lines = ds.queries.map((q: any) => JSON.stringify({ id: q.id, relevant: [...q.relevant] }));
  writeFileSync(`${dest}/${key}.qrels.jsonl`, lines.join('\n') + '\n');
  console.log(`${key}: ${lines.length} queries`);
}
