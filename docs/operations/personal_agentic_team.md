# Personal Agentic Team (Local Ops Loop)

Purpose: run a repeatable, evidence-first cycle that (1) inventories the codebase, (2) ingests high-signal local datasets, (3) captures receipts, and (4) (optionally) delegates synthesis to a local LLM under Ihsān constraints.

Hard rules:
- No deletions (index + copy small artifacts only).
- No tracked secrets. Anything sensitive stays in env vars or external vaults.
- “VERIFIED” claims require evidence (receipt + `path:line` or command output).

## One-command run

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_master.ps1 `
  -IngestCrashReports `
  -RunLLMTeam `
  -ModelTarget qwen2.5:7b
```

## Outputs (where to look)

**Activation receipt (MEASURED)**
- Location: `docs/evidence/receipts/activation_<timestamp>/`
- Key files:
  - `35_codebase_inventory.json` and `35_codebase_context.txt`
  - `41_chat_ingest_receipt.json` (Data Lake chat index pointer + hashes)
  - `46_windows_crash_ingest_receipt.json` (Data Lake crash ingest pointer + hashes)
  - `51_llm_team_receipt.json` (LLM team output directory pointer)
  - `ZZ_RUN_MANIFEST.json` (hashes for everything in the run dir)

**LLM team receipts (DERIVED)**
- Location: `docs/evidence/receipts/llm_team_<timestamp>/`
- Each role produces:
  - `prompt_<ROLE>.txt` (what we asked)
  - `response_<ROLE>.json` (raw Ollama response)
  - `result_<ROLE>.json` (parsed + schema-validated JSON)

**Data Lake (indexed artifacts)**
- Chat history index: `C:\BIZRA-DATA-LAKE\03_INDEXED\chat_history\<run_id>\`
- Crash ingest: `C:\BIZRA-DATA-LAKE\03_INDEXED\windows_crash\<run_id>\`

## Truth labels

- `MEASURED`: direct command output, hashes, file stats, or indexed inventories.
- `DERIVED`: computations from measured sources (aggregation, heuristics).
- `PLANNED`: future work or recommendations.
- `ASSUMED/UNKNOWN`: explicitly not verified.

## Safety notes

- Crash ingest copies only “small” files by default (`tools/ingest_crash_reports.py` defaults to `--copy-max-mb 50`).
- LLM role outputs are advisory; treat them as `DERIVED` until verified against the repo and receipts.

