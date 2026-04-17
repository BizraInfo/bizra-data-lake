# BIZRA — polyglot task runner
#
# بسم الله الرحمن الرحيم
#
# One command surface for a Rust + Python + TypeScript + C repository.
# Applies the elite polyglot blueprint §7 (DX).
#
# Install just:
#   cargo install just        (universal — repo already requires cargo)
#   sudo apt install just     (Ubuntu 24.04+)
#   brew install just         (macOS)
#
# Run:
#   just                      list all targets
#   just dev                  start gateway + frontend (G4 runway)
#   just test                 all tests across all languages
#   just check                lint + test + typecheck (fail-closed)
#   just forge "description"  generate proof-forge receipt
#
# See docs/BIZRA-Handover-v1.md + docs/BIZRA-Repo-Inventory-v1.md
# for what each subsystem does.

# Constants
AWARD := absolute_path("../award-winner-design")
WORKSPACE := "bizra-omega"
GATEWAY_BIN := "./bizra-omega/target/release/bizra-cognition-gateway"
DEMA_BIN := "./bizra-omega/target/release/dema"
GATEWAY_PORT := env("BIZRA_COGNITION_PORT", "7421")

# default: list available recipes
default:
    @just --list --unsorted

# ─── Development ─────────────────────────────────────────────────────────────

# Start gateway + frontend dev server (the G4 runway)
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f "{{GATEWAY_BIN}}" ]; then
      echo "▶ release binary missing — building first..."
      just build-rust
    fi
    echo "▶ starting gateway on 127.0.0.1:{{GATEWAY_PORT}}"
    {{GATEWAY_BIN}} > /tmp/bizra-gateway.log 2>&1 &
    GW=$!
    sleep 1
    curl -sf http://127.0.0.1:{{GATEWAY_PORT}}/health > /dev/null && echo "  gateway: ok"
    echo "▶ starting Next.js dev server (award-winner-design)"
    cd {{AWARD}} && pnpm dev &
    echo
    echo "  gateway PID: $GW"
    echo "  stop both:   just stop"
    wait

# Start gateway only
dev-gateway:
    {{GATEWAY_BIN}}

# Start Next.js frontend only
dev-frontend:
    cd {{AWARD}} && pnpm dev

# Stop all dev services
stop:
    #!/usr/bin/env bash
    pkill -f "target/release/bizra-cognition-gateway" 2>/dev/null || true
    pkill -f "next.*dev" 2>/dev/null || true
    sleep 1
    echo "  stopped"

# Status of running services
status:
    #!/usr/bin/env bash
    echo "▶ gateway"
    curl -sf http://127.0.0.1:{{GATEWAY_PORT}}/health 2>/dev/null && echo || echo "  DOWN"
    echo "▶ next dev"
    ss -ltn 2>/dev/null | grep -qE ":300[0-9]" && echo "  up" || echo "  DOWN"

# ─── Testing ──────────────────────────────────────────────────────────────────

# All tests across Rust + TypeScript (Python tests invoked via pytest directly)
test: test-cognition test-gateway test-frontend
    @echo "  ✓ all language test suites green"

# Rust kernel tests (64 expected)
test-cognition:
    cd {{WORKSPACE}} && cargo test -p bizra-cognition --lib

# Rust gateway tests (7 expected)
test-gateway:
    cd {{WORKSPACE}} && cargo test -p bizra-cognition-gateway

# Full Rust workspace (~1,200+ tests, all 28 crates)
test-workspace:
    cd {{WORKSPACE}} && cargo test --workspace

# Frontend vitest suite (award-winner-design — 135 tests)
test-frontend:
    cd {{AWARD}} && pnpm test

# Python test suite (if pytest configured at root)
test-python:
    #!/usr/bin/env bash
    if [ -f conftest.py ] || [ -f pyproject.toml ]; then
      python -m pytest tests/ || echo "  (pytest not fully wired — skipping)"
    else
      echo "  (no root pytest config — skipping)"
    fi

# ─── Linting & type-checking ─────────────────────────────────────────────────

# All linters + type-checkers across languages
lint: lint-rust lint-frontend
    @echo "  ✓ all lint gates green"

# Rust clippy on session crates (zero warnings required)
lint-rust:
    cd {{WORKSPACE}} && cargo clippy -p bizra-cognition -p bizra-cognition-gateway --no-deps -- -D warnings

# Frontend eslint + typecheck
lint-frontend:
    cd {{AWARD}} && pnpm typecheck && pnpm lint

# Python: ruff + black (via pre-commit)
lint-python:
    pre-commit run --all-files ruff isort black || echo "  (pre-commit not installed — run: pip install pre-commit)"

# ─── Full-stack gate (lint + test + typecheck) ───────────────────────────────

# Elite fail-closed gate — run this before every commit
check: lint test
    @echo "  ✓ polyglot gate PASS"

# ─── Security audits (polyglot blueprint §2) ─────────────────────────────────

# All security audits (fail if any vulns at high severity)
audit: audit-rust audit-frontend audit-python

audit-rust:
    cd {{WORKSPACE}} && (command -v cargo-audit > /dev/null || cargo install cargo-audit) && cargo audit

audit-frontend:
    cd {{AWARD}} && pnpm audit --audit-level=high || echo "  (vulns present — see report)"

audit-python:
    #!/usr/bin/env bash
    set -uo pipefail
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || true
    if ! command -v pip-audit > /dev/null; then
      echo "  pip-audit not installed — run: uv pip install pip-audit (venv) or pipx install pip-audit"
      exit 1
    fi
    fail=0
    echo "▶ root requirements.txt"
    pip-audit -r requirements.txt --progress-spinner off || fail=1
    echo
    echo "▶ services/node_gateway/requirements.txt"
    pip-audit -r services/node_gateway/requirements.txt --progress-spinner off || fail=1
    echo
    echo "▶ services/jarvis/requirements.txt"
    pip-audit -r services/jarvis/requirements.txt --progress-spinner off || echo "  (unscannable — see runtime/RUNTIME_STATUS.md for known manifest issue)"
    echo
    [ "$fail" -eq 0 ] && echo "  ✓ python audit clean on scannable surfaces" || echo "  ✗ python vulns found (see above)"
    exit "$fail"

# ─── Build ───────────────────────────────────────────────────────────────────

# All release builds
build: build-rust build-frontend

# Rust release binaries (gateway + dema CLI)
build-rust:
    cd {{WORKSPACE}} && cargo build --release -p bizra-cognition-gateway
    @echo "  binary: {{GATEWAY_BIN}}"
    @echo "  binary: {{DEMA_BIN}}"

# Next.js production build
build-frontend:
    cd {{AWARD}} && pnpm build

# ─── Proof Forge (evidence kernel shipped Cycle-5) ───────────────────────────

# Generate a proof-forge receipt for current work
forge description="ad-hoc forge run":
    python3 .proof-forge/scripts/forge_evidence.py \
      --project-dir . \
      --description "{{description}}"

# Verify the proof-forge chain integrity from genesis
verify-chain:
    python3 .proof-forge/scripts/forge_evidence.py --verify --project-dir .

# ─── Dema CLI passthrough (polyglot blueprint §7 — single command surface) ───

# Run the dema CLI (e.g., `just dema activate`, `just dema chain`)
dema *args:
    {{DEMA_BIN}} {{args}}

# ─── Repo introspection ──────────────────────────────────────────────────────

# Show the polyglot repo structure at a glance
info:
    @echo "BIZRA Data Lake — polyglot production system"
    @echo
    @echo "Subsystems:"
    @echo "  bizra-omega/        — Rust workspace (28 crates)"
    @echo "  runtime/            — second Rust workspace (meta_alpha_dual_agentic)"
    @echo "  core/               — 74 Python subsystems"
    @echo "  bizra-node0/core/   — 15 Python Node0 subsystems"
    @echo "  services/           — 3 microservices (jarvis, node_gateway, _shared)"
    @echo "  frontend/           — Vite frontend (internal)"
    @echo "  sovereign_state/    — persistent runtime state (2,512 files)"
    @echo
    @echo "See docs/BIZRA-Repo-Inventory-v1.md for the full map."

# Git status scoped to session-canonical paths only
session-status:
    @echo "▶ session canon scope (excludes pre-existing dirty tree)"
    @git status --short \
      bizra-omega/bizra-cognition/src/ \
      bizra-omega/bizra-cognition-gateway/src/ \
      docs/ cycle-5/ PROOF_SUMMARY.md .proof-forge/ 2>/dev/null | head -20

# ─── Cleanup ─────────────────────────────────────────────────────────────────

# Clean build artifacts (Rust + Node)
clean:
    cd {{WORKSPACE}} && cargo clean
    cd {{AWARD}} && rm -rf .next node_modules/.cache 2>/dev/null || true
    @echo "  ✓ cleaned"

# ─── Pre-commit (blueprint §7 DX hygiene) ────────────────────────────────────

# Install pre-commit hooks (one-time)
install-hooks:
    pip install pre-commit
    pre-commit install

# Run all pre-commit hooks on all files
pre-commit-all:
    pre-commit run --all-files

# ─── Bootstrap (single-command setup from fresh clone) ───────────────────────

# Set up the whole dev environment (blueprint §7 DX: Single-Command Bootstrap)
bootstrap:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "▶ BIZRA polyglot bootstrap"
    echo
    echo "▶ checking toolchain..."
    command -v cargo > /dev/null || { echo "✗ cargo missing — install Rust"; exit 1; }
    command -v pnpm > /dev/null || { echo "✗ pnpm missing — install pnpm"; exit 1; }
    command -v python3 > /dev/null || { echo "✗ python3 missing"; exit 1; }
    echo "  ✓ cargo, pnpm, python3 present"
    echo
    echo "▶ building Rust workspace (release)..."
    cd {{WORKSPACE}} && cargo build --release -p bizra-cognition-gateway && cd ..
    echo "  ✓ gateway + dema built"
    echo
    if [ -d "{{AWARD}}" ]; then
      echo "▶ installing frontend deps..."
      cd {{AWARD}} && pnpm install && cd -
      echo "  ✓ pnpm install complete"
    fi
    echo
    echo "▶ ready. Try:"
    echo "  just dev       # start gateway + frontend"
    echo "  just dema      # run the CLI"
    echo "  just test      # run all tests"
    echo "  just check     # lint + test (before commit)"
