# BIZRA Data Lake — Local Development Makefile
# Single entry point for all dev workflows on Node0
#
# Usage:
#   make test          — run core proof_engine tests (fast)
#   make test-all      — run full test suite
#   make gate          — run promotion gate pipeline
#   make fate          — run FATE gate on a sample input (live Ollama)
#   make lint          — run pre-commit hooks
#   make check         — cargo check Rust workspace
#   make build         — cargo build Rust workspace (release)
#   make health        — Node0 system health check
#   make clean         — remove caches

SHELL := /bin/bash
VENV := .venv/bin/activate
OMEGA := bizra-omega
PYTEST_FLAGS := -q --timeout=60
LOG_DIR := /data/bizra/logs

.PHONY: test test-all test-proof gate fate lint check build health clean

# ── Python Tests ──────────────────────────────────────────────

test:
	@source $(VENV) && python -m pytest tests/core/proof_engine/ $(PYTEST_FLAGS)

test-mvda:
	@source $(VENV) && python -m pytest tests/mvda/ $(PYTEST_FLAGS) --timeout=300

test-all:
	@source $(VENV) && python -m pytest tests/ $(PYTEST_FLAGS) --timeout=120 \
		-m "not slow and not requires_ollama and not requires_gpu and not requires_network" \
		2>&1 | tee $(LOG_DIR)/test-all-$$(date +%Y%m%d-%H%M%S).log

test-proof:
	@source $(VENV) && python -m pytest tests/core/proof_engine/ -v --timeout=60 \
		2>&1 | tee $(LOG_DIR)/test-proof-$$(date +%Y%m%d-%H%M%S).log

# ── Promotion Gate ────────────────────────────────────────────

gate:
	@bash scripts/promotion_gate.sh all

# ── FATE Gate (live Ollama) ───────────────────────────────────

fate:
	@source $(VENV) && python -c "\
	from core.proof_engine.fate_gate import validate_with_evidence; \
	from core.proof_engine.sat_validator import SimplePatOutput; \
	pat = SimplePatOutput( \
	    answer='The Spearpoint seal is commit b08f2208, sealing Substrate Transition v1.', \
	    evidence_refs=['git-show:b08f2208', 'file:core/zpk/kernel.py'], \
	); \
	r = validate_with_evidence(pat); \
	print(f'Verdict: {r.verdict.verdict}'); \
	print(f'Ihsan: {r.verdict.ihsan_score}'); \
	print(f'Evidence valid: {r.evidence_audit.all_refs_valid}'); \
	print(f'Short-circuited: {r.short_circuited}'); \
	print(f'Passed: {r.passed}'); \
	"

# ── Lint ──────────────────────────────────────────────────────

lint:
	@source $(VENV) && pre-commit run --all-files 2>&1 | tail -20

lint-check:
	@source $(VENV) && ruff check core/ --select E,W,F --quiet

# ── Rust ──────────────────────────────────────────────────────

check:
	@cd $(OMEGA) && cargo check --workspace 2>&1 | tail -5

build:
	@cd $(OMEGA) && cargo build --workspace --release 2>&1 | tail -5

# ── Loop Proof ────────────────────────────────────────────────

loop-proof:
	@source $(VENV) && python -m core.proof_engine.loop_proof_cli

# ── Cockpit ───────────────────────────────────────────────────

cockpit:
	@source $(VENV) && python -m core.cockpit.server

# ── Health ────────────────────────────────────────────────────

health:
	@echo "=== Node0 Health ===" && \
	echo "GPU:" && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null && \
	echo "Ollama:" && curl -s http://127.0.0.1:11434/api/version | python3 -c "import json,sys;print(json.load(sys.stdin)['version'])" 2>/dev/null && \
	echo "Disk:" && df -h /data | tail -1 && \
	echo "Tests:" && source $(VENV) && python -m pytest tests/core/proof_engine/ --co -q 2>&1 | tail -1

# ── Ancestry ──────────────────────────────────────────────────

spearpoint:
	@git merge-base --is-ancestor b08f2208 HEAD && echo "Spearpoint b08f2208: REACHABLE" || echo "BROKEN"

# ── Clean ─────────────────────────────────────────────────────

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache 2>/dev/null; \
	echo "Cleaned."
