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
  echo "Usage: secrets_scan.sh --root <path> --out <outdir>"
  exit 1
fi
mkdir -p "$OUT"
OUTFILE="$OUT/secrets_scan.txt"

if command -v trufflehog >/dev/null 2>&1; then
  echo "Running trufflehog filesystem scan..." > "$OUTFILE"
  trufflehog filesystem --no-update "$ROOT" >> "$OUTFILE" 2>&1 || true
  echo "Wrote: $OUTFILE"
  exit 0
fi

echo "trufflehog not found. Running lightweight regex scan (best-effort)..." > "$OUTFILE"
rg -n --hidden --no-ignore -S \
  -e 'AKIA[0-9A-Z]{16}' \
  -e '-----BEGIN(.*)PRIVATE KEY-----' \
  -e 'sk-[A-Za-z0-9]{20,}' \
  -e 'AIza[0-9A-Za-z\-_]{35}' \
  -e 'xox[baprs]-[0-9A-Za-z\-]{10,}' \
  "$ROOT" >> "$OUTFILE" 2>/dev/null || true

echo "Wrote: $OUTFILE"
