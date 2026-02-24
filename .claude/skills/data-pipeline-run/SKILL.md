---
name: data-pipeline-run
description: Run the BIZRA 4-stage data pipeline with validation between stages. Stages: corpus_manager -> vector_engine -> langextract_engine -> arte_engine.
disable-model-invocation: true
---

# BIZRA Data Pipeline Runner

Run the 4-stage data pipeline with inter-stage validation.

## Arguments

$ARGUMENTS

If no arguments provided, run all 4 stages in order. Accepted arguments:
- `all` — Run all stages (default)
- `corpus` / `stage1` — Only corpus_manager.py
- `vectors` / `stage2` — Only vector_engine.py
- `extract` / `stage4` — Only langextract_engine.py
- `arte` / `snr` — Only arte_engine.py
- `from:<stage>` — Run from a given stage onward (e.g., `from:vectors`)
- `--dry-run` — Show what would be executed without running

## Pipeline Stages

| Stage | Script | Input | Output | Validation |
|-------|--------|-------|--------|------------|
| 1 | `corpus_manager.py` | `00_INTAKE/` | `04_GOLD/documents.parquet` | File exists + row count > 0 |
| 2 | `vector_engine.py` | `04_GOLD/documents.parquet` | `04_GOLD/chunks.parquet` | File exists + embeddings shape |
| 4 | `langextract_engine.py` | chunks | `assertions.jsonl` | File exists + valid JSON lines |
| ARTE | `arte_engine.py` | assertions | SNR validation report | SNR scores >= 0.85 |

## Execution Protocol

For each stage:

1. **Pre-check**: Verify input files exist
2. **Execute**: Run the script from repo root with `python3 <script>.py`
3. **Validate**: Check output files were created/updated
4. **Report**: Show row counts, file sizes, timing

```bash
# Example execution for each stage:
cd "$REPO_ROOT"
source .venv/bin/activate 2>/dev/null || true

# Stage 1
python3 corpus_manager.py
# Validate: ls -la 04_GOLD/documents.parquet

# Stage 2
python3 vector_engine.py
# Validate: ls -la 04_GOLD/chunks.parquet

# Stage 4
python3 langextract_engine.py
# Validate: wc -l assertions.jsonl 2>/dev/null || echo "assertions.jsonl not found"

# ARTE
python3 arte_engine.py
```

## Error Handling

- If a stage fails, STOP the pipeline and report the error
- Do NOT continue to the next stage on failure
- Show the last 20 lines of output for diagnosis
- Suggest common fixes (missing .env keys, LM Studio not running, etc.)

## Environment Requirements

- Python virtualenv activated (`.venv/bin/activate`)
- `.env` file configured (LM_STUDIO_API_KEY for stage 4)
- For stage 2: sentence-transformers installed (`pip install -e ".[full]"`)
- For stage 4: Gemini API key or LM Studio running

## Output Format

```
=== BIZRA Data Pipeline ===

[Stage 1/4] corpus_manager.py
  Input:  00_INTAKE/ (N files)
  Status: SUCCESS (12.3s)
  Output: 04_GOLD/documents.parquet (1,234 rows, 45.2 MB)

[Stage 2/4] vector_engine.py
  Input:  04_GOLD/documents.parquet (1,234 rows)
  Status: SUCCESS (45.1s)
  Output: 04_GOLD/chunks.parquet (5,678 rows, 123.4 MB)

[Stage 4/4] langextract_engine.py
  Input:  04_GOLD/chunks.parquet (5,678 rows)
  Status: SUCCESS (120.5s)
  Output: assertions.jsonl (890 lines)

[ARTE] arte_engine.py
  Status: SUCCESS (8.2s)
  SNR scores: min=0.87, mean=0.93, max=0.99

=== Pipeline Complete (186.1s total) ===
```
