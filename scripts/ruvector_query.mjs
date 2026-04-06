#!/usr/bin/env node
/**
 * RuVector Query Helper — BIZRA-DATA-LAKE
 * Accepts JSON on stdin: { "vector": [float...], "k": 5 }
 * Returns JSON on stdout: [{ "id", "score", "text" }, ...]
 * Usage: echo '{"vector":[...],"k":5}' | NODE_PATH=/usr/lib/node_modules node scripts/ruvector_query.mjs
 */
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const { VectorDb } = require('@ruvector/core');

const DB_PATH = process.env.RUVECTOR_DB || '04_GOLD/ruvector_bizra';
const DIM = 384;

async function main() {
  // Read JSON from stdin
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));

  const db = new VectorDb({ path: DB_PATH, collection: 'bizra_chunks', dimensions: DIM });
  const queryVec = new Float32Array(input.vector);
  const k = input.k || 5;

  const results = await db.search({ vector: queryVec, k });

  const output = [];
  if (Array.isArray(results)) {
    for (const r of results) {
      const meta = r.metadata ? JSON.parse(r.metadata) : {};
      output.push({
        id: r.id || '',
        score: r.score ?? 0,
        text: meta.text || '',
      });
    }
  }

  process.stdout.write(JSON.stringify(output));
}

main().catch(err => {
  process.stderr.write(`ruvector_query error: ${err.message}\n`);
  process.exit(1);
});
