#!/usr/bin/env bash
set -euo pipefail

# Phase 56 security hardening gate.
# Autonomous Graph-of-Thought execution with weighted SNR scoring.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

THRESHOLD="${BIZRA_PHASE56_SNR_THRESHOLD:-0.95}"
ALLOW_OFFLINE="${BIZRA_CARGO_FALLBACK_OFFLINE:-1}"

ENGINE_ARGS=(
  "scripts/phase56_autonomous_engine.py"
  "--root" "${ROOT_DIR}"
  "--threshold" "${THRESHOLD}"
  "--report-json" "artifacts/phase56/phase56_engine_report.json"
  "--report-md" "artifacts/phase56/phase56_engine_report.md"
)

if [[ "${ALLOW_OFFLINE}" == "1" ]]; then
  ENGINE_ARGS+=("--allow-offline-cargo")
fi

echo "[phase56] Running autonomous hardening engine..."
PYTHONPATH="${ROOT_DIR}" "${PYTHON:-python3}" "${ENGINE_ARGS[@]}"

echo "[phase56] Security hardening gate passed."
