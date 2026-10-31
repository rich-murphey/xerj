// Streams claude-mem's live server store into xerj, read-only: one SELECT,
// through psql in the Postgres container, of every observation that has
// embeddings, with its parts' stored vectors. No text is embedded.
//
//   node research/claude-mem/export-live.ts [--container claude-mem-server-postgres-1] [--name cm-live]
//
// Index <name>-parts: one doc per (observation_id, part): observation_id,
// project_id, kind, title, created_at, part, the part's text (BM25) and its
// stored 384-dim vector. Index <name>-obs: one doc per observation with its
// whole content, which is what claude-mem's keyword leg ranks.
//
// Part texts are derived as claude-mem derives them before embedding
// (observationEmbeddingParts, src/storage/postgres/observation-embeddings.ts).
import { spawn } from 'node:child_process';
import { parseArgs } from 'node:util';
import { Bulk, createIndexes, DIMS, lines, xerj } from './lib.ts';

const SUMMARY_FIELDS = ['request', 'investigated', 'learned', 'completed', 'next_steps', 'notes'];
const MAX_FACT_PARTS = 64;

/** claude-mem's observationEmbeddingParts, as a part name → text map. */
export function partTexts(kind: string, md: any, content: string): Map<string, string> {
  const parts = new Map<string, string>();
  const add = (part: string, v: unknown) => {
    if (typeof v === 'string' && v.trim().length > 0) parts.set(part, v);
  };
  if (kind === 'summary') {
    for (const f of SUMMARY_FIELDS) add(f, md[f]);
  } else {
    add('narrative', md.narrative);
    add('text', md.legacy && typeof md.legacy === 'object' ? md.legacy.text : undefined);
    const facts = (Array.isArray(md.facts) ? md.facts : [])
      .filter((f: unknown): f is string => typeof f === 'string' && f.trim().length > 0)
      .slice(0, MAX_FACT_PARTS);
    facts.forEach((f: string, i: number) => add(`fact_${i}`, f));
    if (parts.size === 0) {
      const title = typeof md.title === 'string' && md.title.trim() !== 'Untitled' ? md.title.trim() : '';
      const subtitle = typeof md.subtitle === 'string' ? md.subtitle.trim() : '';
      const heading = [title, subtitle].filter(t => t.length > 0).join('\n');
      add('title', heading);
      if (content.trim() !== heading) add('content', content);
    }
  }
  if (parts.size === 0) add('content', content);
  return parts;
}

/** One JSON object per observation, streamed by psql (FETCH_COUNT keeps it from buffering the result). */
export function streamObservations(container: string, model: string, where = 'TRUE') {
  const sql = `
    SELECT json_build_object(
      'id', o.id, 'project_id', o.project_id, 'kind', o.kind, 'created_at', o.created_at,
      'metadata', o.metadata, 'content', o.content,
      'parts', (SELECT json_agg(json_build_object('part', e.part, 'vec', e.embedding::text::json))
                FROM observation_embeddings e WHERE e.observation_id = o.id AND e.model = ${quote(model)}))
    FROM observations o
    WHERE EXISTS (SELECT 1 FROM observation_embeddings e WHERE e.observation_id = o.id AND e.model = ${quote(model)})
      AND ${where};`;
  const psql = spawn('docker', ['exec', '-i', container, 'sh', '-c',
    'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -v ON_ERROR_STOP=1 -v FETCH_COUNT=500'], { stdio: ['pipe', 'pipe', 'inherit'] });
  psql.stdin.end(sql);
  return lines(psql.stdout);
}
const quote = (s: string) => `'${s.replace(/'/g, "''")}'`;

if (import.meta.main) {
  const { values } = parseArgs({
    options: {
      container: { type: 'string', default: 'claude-mem-server-postgres-1' },
      name: { type: 'string', default: 'cm-live' },
      model: { type: 'string', default: 'Xenova/all-MiniLM-L6-v2/fp32/mean/l2/256' },
    },
  });
  const started = performance.now();
  await createIndexes(values.name!);
  const bulk = new Bulk(1000);
  let observations = 0, parts = 0, missingText = 0;
  for await (const line of streamObservations(values.container!, values.model!)) {
    const o = JSON.parse(line);
    const md = o.metadata ?? {};
    const texts = partTexts(o.kind, md, o.content);
    const meta = {
      observation_id: o.id,
      project_id: o.project_id,
      kind: o.kind,
      title: typeof md.title === 'string' ? md.title : null,
      created_at: o.created_at,
    };
    await bulk.add(`${values.name}-obs`, o.id, { ...meta, content: o.content });
    for (const p of o.parts) {
      if (p.vec.length !== DIMS) throw new Error(`${o.id} ${p.part}: ${p.vec.length} dims`);
      const text = texts.get(p.part);
      if (text === undefined) missingText++;
      await bulk.add(`${values.name}-parts`, `${o.id}#${p.part}`, { ...meta, part: p.part, text: text ?? '', vec: p.vec });
      parts++;
    }
    if (++observations % 5000 === 0) console.log(`${observations} observations, ${parts} parts`);
  }
  await bulk.flush();
  for (const index of [`${values.name}-parts`, `${values.name}-obs`]) await xerj('POST', `/${index}/_refresh`);
  const count = (await xerj('GET', `/${values.name}-parts/_count`)).count;
  console.log(`${observations} observations, ${parts} parts (${count} in xerj), ${missingText} parts with no derivable text, ` +
    `${((performance.now() - started) / 1000).toFixed(0)}s`);
}
