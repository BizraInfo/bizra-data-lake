#!/usr/bin/env bun
# ═══════════════════════════════════════════════════════════════
# DEMA — Quality Gate Pipeline
# Lint → Typecheck → DB Push → Build → Health Check
# Exit code 0 = all gates passed. Non-zero = failure.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

GATE=0
TOTAL=4
PASS=0
FAIL=0

gate() {
  local name="$1"
  GATE=$((GATE + 1))
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  GATE $GATE/$TOTAL: $name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

pass() {
  PASS=$((PASS + 1))
  echo "  ✅ PASSED"
}

fail() {
  FAIL=$((FAIL + 1))
  echo "  ❌ FAILED: $1"
}

# ─── Gate 1: ESLint ─────────────────────────────────────────
gate "ESLint — Static Analysis"
if bun run lint 2>&1; then
  pass
else
  fail "Lint errors detected"
fi

# ─── Gate 2: Database Schema Push ────────────────────────────
gate "Prisma — Schema Validation & Push"
if bun run db:push 2>&1; then
  pass
else
  fail "Database schema push failed"
fi

# ─── Gate 3: Prisma Generate ─────────────────────────────────
gate "Prisma — Client Generation"
if bun run db:generate 2>&1; then
  pass
else
  fail "Client generation failed"
fi

# ─── Gate 4: Next.js Build ───────────────────────────────────
gate "Next.js — Production Build"
if bun run build 2>&1; then
  pass
else
  fail "Build failed"
fi

# ─── Summary ──────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  PIPELINE SUMMARY"
echo "  Passed: $PASS/$TOTAL"
echo "  Failed: $FAIL/$TOTAL"
echo "════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
  echo "  ⛔ QUALITY GATE BLOCKED"
  exit 1
else
  echo "  🟢 ALL GATES PASSED — READY TO SHIP"
  exit 0
fi
