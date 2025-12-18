#!/usr/bin/env bash
set -euo pipefail

ROOT=""
OUT=""
INCLUDE_HASH="false"
HASH_MAX_MB=10
MAX_FILES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --include-hash) INCLUDE_HASH="true"; shift 1;;
    --hash-max-mb) HASH_MAX_MB="$2"; shift 2;;
    --max-files) MAX_FILES="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$ROOT" || -z "$OUT" ]]; then
  echo "Usage: inventory.sh --root <path> --out <outdir> [--include-hash] [--hash-max-mb N] [--max-files N]"
  exit 1
fi

mkdir -p "$OUT"
MANIFEST="$OUT/manifest.jsonl"
rm -f "$MANIFEST"

count=0
hash_max_bytes=$((HASH_MAX_MB*1024*1024))

# Portable stat: try GNU then BSD
stat_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }
stat_mtime() { stat -c%Y "$1" 2>/dev/null || stat -f%m "$1"; }

while IFS= read -r -d '' f; do
  if [[ "$MAX_FILES" -gt 0 && "$count" -ge "$MAX_FILES" ]]; then
    break
  fi

  size=$(stat_size "$f")
  mtime_epoch=$(stat_mtime "$f")
  # ISO8601 UTC
  mtime=$(date -u -d "@$mtime_epoch" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u -r "$mtime_epoch" +"%Y-%m-%dT%H:%M:%SZ")
  ext="${f##*.}"
  if [[ "$f" == "$ext" ]]; then ext=""; else ext=".$ext"; fi
  ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

  sha=""
  if [[ "$INCLUDE_HASH" == "true" && "$size" -le "$hash_max_bytes" ]]; then
    sha=$(sha256sum "$f" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$f" | awk '{print $1}')
  fi

  if [[ -z "$sha" ]]; then
    printf '{"path":%q,"size_bytes":%s,"mtime_utc":%q,"ext":%q}\n' "$f" "$size" "$mtime" "$ext" >> "$MANIFEST"
  else
    printf '{"path":%q,"size_bytes":%s,"mtime_utc":%q,"ext":%q,"sha256":%q}\n' "$f" "$size" "$mtime" "$ext" "$sha" >> "$MANIFEST"
  fi

  count=$((count+1))
done < <(find "$ROOT" -type f -print0)

echo "Wrote manifest: $MANIFEST"
echo "Files indexed: $count"
