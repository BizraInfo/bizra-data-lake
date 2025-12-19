# Neo4j Runbook (House of Wisdom)

Purpose: Run Neo4j locally (Node0), ingest the BIZRA knowledge ledger, and operate the graph with secure defaults and auditable procedures.

Truth labels: VERIFIED | MEASURED | TARGET | DERIVED  
Ihsān rule: Mark VERIFIED only with evidence (`path:line`) or captured command output.

---

## 1) Secure Defaults (Amānah)

- Bind Neo4j to localhost unless you have a documented reason to expose it.
- Do not store the password in tracked files; prefer an environment variable (example: `GRAPH_PASSWORD`).

---

## 2) Start Neo4j (Docker)

### Option A (recommended): standalone Neo4j container (local-only)

PowerShell (example):
```powershell
$env:GRAPH_PASSWORD = "CHANGE_ME"

docker run -d --name bizra-node0-graph `
  -p 127.0.0.1:7474:7474 `
  -p 127.0.0.1:7687:7687 `
  -e NEO4J_AUTH="neo4j/$env:GRAPH_PASSWORD" `
  neo4j:5.15-community
```

Or use the helper scripts (recommended for repeatability):
```powershell
$env:GRAPH_PASSWORD = "CHANGE_ME"
.\scripts\start_neo4j_local.ps1
```

If PowerShell script execution is restricted on your machine:
```powershell
$env:GRAPH_PASSWORD = "CHANGE_ME"
powershell -ExecutionPolicy Bypass -File .\scripts\start_neo4j_local.ps1
```

Verify the port is open:
```powershell
Test-NetConnection -ComputerName localhost -Port 7687
```

Open Neo4j Browser:
- http://localhost:7474

### Option B: integrate into a compose stack (TARGET)

If/when Node0 compose includes Neo4j, ensure:
- `NEO4J_AUTH` comes from env var (not committed),
- ports are bound to `127.0.0.1`,
- volumes are persisted for durability.

---

## 3) Ingest the Knowledge Ledger (Synaptic Loader)

Preconditions:
- `BIZRA_KNOWLEDGE_MANIFEST.json` exists (local output).
- `BIZRA_KNOWLEDGE_LEDGER.jsonl` exists (local output).

Dry-run (verifies chain; no DB writes):
```powershell
python .\bizra_synaptic_loader.py --dry-run
```

Ingest:
```powershell
$env:GRAPH_PASSWORD = "CHANGE_ME"
python .\bizra_synaptic_loader.py
```

Notes:
- The loader verifies ledger-chain integrity by default before writing.
- The loader creates uniqueness constraints unless `--no-constraints` is passed.
- `--batch-size 1000` is the default; tune if needed.
- The loader writes an evidence receipt by default under `docs/evidence/receipts/knowledge_graph_ingest_*/receipt.json`.
  - Disable: `--no-receipt`
  - Override location: `--receipt-out <path>` or `--receipt-dir <dir>`
  - Privacy warning: `--receipt-include-paths` writes raw paths + scan_root into the receipt.

---

## 4) Minimal Query Playbook (SNR-first)

Top artifacts by impact:
```cypher
MATCH (a:Artifact)
RETURN a.filename, a.path, a.impact_value
ORDER BY a.impact_value DESC
LIMIT 25;
```

Filter by extension:
```cypher
MATCH (a:Artifact)-[:HAS_EXTENSION]->(e:FileExtension {ext: ".rs"})
RETURN a.filename, a.path, a.impact_value
ORDER BY a.impact_value DESC
LIMIT 50;
```

Filter by class:
```cypher
MATCH (a:Artifact)-[:CLASSIFIED_AS]->(c:AssetClass {name: "docs"})
RETURN a.filename, a.path, a.impact_value
ORDER BY a.impact_value DESC
LIMIT 50;
```

Verify anchoring to Genesis:
```cypher
MATCH (m:KnowledgeManifest)-[:ANCHORED_TO]->(g:GenesisBlock)
RETURN m.ledger_chain_sha256, g.hash, m.total_files, m.total_value_bzr_g
LIMIT 10;
```

---

## 5) Backup / Restore (TARGET → operationalize in Phase 3)

Backup (inside container):
```powershell
docker exec bizra-node0-graph neo4j-admin database dump neo4j --to-path=/data
```

Restore (example, destructive; run only when intended):
```powershell
docker exec bizra-node0-graph neo4j-admin database load neo4j --from-path=/data --overwrite-destination=true
```

---

## 6) Operational Notes

- Treat the ledger + manifest as sensitive (paths can reveal personal structure); keep them local and git-ignored.
- Evidence receipts for ingestion should include: manifest hash, ledger chain, node count, duration, and constraint creation status.
