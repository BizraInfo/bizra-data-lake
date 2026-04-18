#!/usr/bin/env bash
# sync-bindings.sh — pull the latest Rust→TS contracts into dema-console
#
# بسم الله الرحمن الرحيم
#
# Source of truth: bizra-omega/bizra-cognition-gateway/bindings/*.ts
# Generator: ts-rs via `cargo test -p bizra-cognition-gateway`
#
# This script is idempotent. Run whenever the canonical bindings
# regenerate. CI drift between Rust DTOs and committed .ts files is
# caught in bizra-cognition-gateway's own drift gate; this script just
# mirrors the result into the console.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC="$REPO_ROOT/bizra-omega/bizra-cognition-gateway/bindings"
DST="$REPO_ROOT/dema-console/src/bindings"

if [ ! -d "$SRC" ]; then
    echo "error: canonical bindings not found at $SRC" >&2
    echo "       run 'cargo test -p bizra-cognition-gateway' first" >&2
    exit 1
fi

mkdir -p "$DST"
# Copy only .ts files; leave dema-console's own README (if any) alone.
find "$SRC" -maxdepth 1 -name '*.ts' -type f -exec cp -v {} "$DST/" \;

echo ""
echo "✓ synced $(ls "$SRC"/*.ts 2>/dev/null | wc -l) .ts contracts into dema-console/src/bindings/"
