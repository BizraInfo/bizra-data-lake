#!/usr/bin/env bash
set -euo pipefail
ROOT=""
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done
if [[ -z "$ROOT" || -z "$OUT" ]]; then
  echo "Usage: repo_health.sh --root <path> --out <outdir>"
  exit 1
fi
mkdir -p "$OUT"
OUTFILE="$OUT/repo_health.txt"
echo "Repo health run at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "$OUTFILE"

pushd "$ROOT" >/dev/null

run() {
  echo "---- $*" >> "$OUTFILE"
  ( "$@" >> "$OUTFILE" 2>&1 ) || true
}

if [[ -f "Cargo.toml" ]]; then
  run cargo --version
  run cargo test --all --locked
  command -v cargo-audit >/dev/null 2>&1 && run cargo audit || true
fi

if [[ -f "package.json" ]]; then
  run node --version
  run npm --version
  run npm ci
  run npm test
  run npm audit --audit-level=high || true
fi

if [[ -f "pyproject.toml" || -f "requirements.txt" ]]; then
  run python3 --version || run python --version
  command -v pip-audit >/dev/null 2>&1 && run pip-audit || true
  [[ -d "tests" ]] && run python3 -m pytest -q || true
fi

popd >/dev/null
echo "Wrote: $OUTFILE"
