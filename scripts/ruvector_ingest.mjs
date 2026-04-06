#!/usr/bin/env node
/**
 * RuVector Ingestion Pipeline — BIZRA-DATA-LAKE
 * Loads 84K+ chunk embeddings from JSONL into RuVector native HNSW.
 * Usage: NODE_PATH=/usr/lib/node_modules node scripts/ruvector_ingest.mjs
 */
import { createRequire } from 'module';
import { createReadStream } from 'fs';
import { createInterface } from 'readline';

const require = createRequire(import.meta.url);
const { VectorDb, CollectionManager } = require('@ruvector/core');

const DB_PATH = process.env.RUVECTOR_DB || '04_GOLD/ruvector_bizra';
const INPUT = process.env.RUVECTOR_INPUT || '04_GOLD/ruvector_ingest.jsonl';
const DIM = 384;
const BATCH_SIZE = 500;

async function main() {
  const t0 = Date.now();
  console.log(`RuVector Ingestion Pipeline`);
  console.log(`  DB:    ${DB_PATH}`);
  console.log(`  Input: ${INPUT}`);
  console.log(`  Dim:   ${DIM} (cosine)\n`);

  // Ensure collection exists (createCollection is async)
  const cm = new CollectionManager(DB_PATH);
  try {
    await cm.createCollection('bizra_chunks', { dimensions: DIM, metric: 'cosine' });
    console.log('  Created collection: bizra_chunks');
  } catch (e) {
    if (e.message && e.message.includes('already exists')) {
      console.log('  Collection exists: bizra_chunks');
    } else {
      throw e;
    }
  }

  // Open database
  const db = new VectorDb({ path: DB_PATH, collection: 'bizra_chunks', dimensions: DIM });
  const existingCount = await db.len();
  console.log(`  Existing vectors: ${existingCount}`);

  if (existingCount > 50000) {
    console.log('  Already ingested. Search test only.\n');
    await searchTest(db);
    return;
  }

  // Stream JSONL
  const rl = createInterface({ input: createReadStream(INPUT), crlfDelay: Infinity });
  let count = 0;
  let batch = [];

  for await (const line of rl) {
    if (!line.trim()) continue;
    const rec = JSON.parse(line);
    batch.push({
      id: rec.id,
      vector: new Float32Array(rec.vector),
      metadata: JSON.stringify({ text: rec.text })
    });
    count++;

    if (batch.length >= BATCH_SIZE) {
      await db.insertBatch(batch);
      batch = [];
      if (count % 10000 === 0) {
        const s = ((Date.now() - t0) / 1000).toFixed(1);
        process.stdout.write(`  ${count} vectors (${Math.round(count / (Date.now() - t0) * 1000)}/s, ${s}s)\r`);
      }
    }
  }
  if (batch.length > 0) await db.insertBatch(batch);

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const total = await db.len();
  console.log(`\n  DONE: ${count} vectors in ${elapsed}s (${Math.round(count / (Date.now() - t0) * 1000)}/s)`);
  console.log(`  Total in DB: ${total}\n`);

  await searchTest(db);
}

async function searchTest(db) {
  // Read only first line — full file exceeds V8 string limit
  const rl2 = createInterface({ input: createReadStream(INPUT), crlfDelay: Infinity });
  let firstLine = '';
  for await (const line of rl2) { firstLine = line; break; }
  const queryVec = new Float32Array(JSON.parse(firstLine).vector);

  const t0 = Date.now();
  const results = await db.search({ vector: queryVec, k: 5 });
  const ms = Date.now() - t0;

  console.log(`  Search test (${ms}ms, top-5):`);
  if (Array.isArray(results)) {
    for (const r of results) {
      const meta = r.metadata ? JSON.parse(r.metadata) : {};
      console.log(`    ${(r.score ?? 0).toFixed(4)} | ${(r.id || '?').slice(0, 24).padEnd(24)} | ${(meta.text || '').slice(0, 50)}`);
    }
  } else {
    console.log('  Results:', JSON.stringify(results).slice(0, 200));
  }
}

main().catch(err => { console.error('FATAL:', err.message); process.exit(1); });
