# Copilot instructions for BIZRA-DATA-LAKE

BIZRA-DATA-LAKE is a proof-native constitutional intelligence monorepo. Treat it as three coupled systems: Python sovereignty infrastructure in `core/`, the Rust workspace in `bizra-omega/`, and operator/web surfaces in `frontend/` and `dema-console/`. More specific `AGENTS.md` files override these instructions, especially `bizra-omega/AGENTS.md` for the hunter crate and `docs/internal/AGENTS.md` for data-lake tiers.

## Build, test, and lint commands

Activate the repo virtualenv before Python commands. The Makefile and MCP configs expect `.venv`; WSL-oriented docs may use `.venv-linux`, so use the venv that exists in the checkout.

```bash
source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/                                                     # default suite; skips slow and e2e_http via pyproject
pytest tests/core/pci/                                            # one module
pytest tests/test_snr_engine.py::test_function_name               # one test
pytest tests/ -m "not slow and not integration and not e2e_http and not requires_ollama and not requires_gpu and not requires_real_data"
pytest tests/ --cov=core --cov-report=term-missing

ruff check core/
black --check core/
isort --check-only core/
mypy core/ --ignore-missing-imports
pre-commit run --all-files
```

Useful Makefile shortcuts:

```bash
make test       # fast proof-engine gate: tests/core/proof_engine/
make test-all   # filtered Python suite, logs under /data/bizra/logs
make lint       # pre-commit checks
make check      # cargo check in bizra-omega
make build      # release cargo build in bizra-omega
```

Rust lives under `bizra-omega/`; install Z3 (`libz3-dev`) before running workspace gates.

```bash
cd bizra-omega
cargo build --workspace --release
cargo test --workspace --release
cargo test -p bizra-core canonical_receipt --release        # one crate/test filter example
cargo test --doc --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

cd bizra-python
maturin develop --release                                   # PyO3 bindings
```

Frontend operator surface (`frontend/`, Vite + React 18):

```bash
cd frontend
npm ci
npm run typecheck
npm run lint
npm run test
npm run test -- tests/api.test.ts                            # one Vitest file
npm run build
npm run ci                                                   # typecheck + lint + test + build
```

Dema Console (`dema-console/`, Next.js + React 19 + Bun) binds to the Rust cognition gateway:

```bash
cd bizra-omega
BIZRA_DEMA_CACHE_ROOT=/tmp/dema-console-dev \
  BIZRA_IDENTITY_ANCHOR=/tmp/dema-console-dev/identity/credentials.json \
  cargo run --release -p bizra-cognition-gateway

cd ../dema-console
bun install
bun run dev
bun test src/lib/activation-state.test.ts                    # one node:test suite
bun run lint
bun run build
```

## High-level architecture

The governed stack is layered: Rust constitutional core and crates in `bizra-omega/` enforce fail-closed primitives; Python `core/` hosts sovereign cognition, proof engines, governance, memory, orchestration, and runtime APIs; `frontend/` and `dema-console/` are operator faces; receipts, manifests, and benchmark artifacts form the proof surface.

The constitutional spine is frozen around canonical root objects in `bizra-omega/bizra-core/src/`: `CanonicalReceipt`, `MissionState`, `TopologyCanon`, `GenesisSeal`, and `ReceiptStateMachine`. Python mirrors canonical receipt behavior in `core/proof_engine/canonical_receipt_adapter.py`; `services/node_gateway/app/routers.py` exposes mission submission that emits canonical receipts.

Python and Rust intentionally mirror major domains. Key pairs include `core/pci/` with `bizra-core/`, `core/proof_engine/` with `bizra-proofspace/`, `core/sovereign/` with `bizra-cli/`, Python orchestration/governance/reasoning modules with Rust action, memory, hooks, agent, node, mission, and protocol crates. When decomposed Python modules exist (`governance/`, `reasoning/`, `orchestration/`, `treasury/`), prefer them over adding more logic to monolithic `core/sovereign/`.

Constitutional thresholds are centralized in `core/integration/constants.py`; do not define local copies. Frontend mirrors live in `frontend/src/tokens.ts`, and cross-language drift is guarded by CI/pre-commit. The core thresholds are Ihsan >= 0.95, SNR >= 0.85, and ADL Gini <= 0.35.

The data pipeline is `00_INTAKE/` -> `01_RAW/` -> `02_PROCESSED/` -> `03_INDEXED/` -> `04_GOLD/`, with duplicates/corruption in `99_QUARANTINE/`. Treat `01_RAW/` as immutable provenance and avoid editing generated evidence/artifact bundles unless the task explicitly targets them.

Runtime and API landmarks: `bizra_config.py` centralizes paths and model backend settings; the installed `bizra` command points to `core.sovereign.__main__:main`; the Python kernel exposes health, knowledge, mission, and briefing endpoints from `core/sovereign/`; the Rust cognition gateway exposes the local HTTP surface consumed by `dema-console/`.

## Key conventions

- Truth labels are part of the code contract for public claims and architectural boundaries: `[ENFORCEMENT: PROVEN]`, `[ENFORCEMENT: WIRED]`, `[OPTIMIZATION: PARTIAL]`, `[OPTIMIZATION: PLANNED]`.
- Security and quality gates are fail-closed. Do not add `|| true` to security-critical checks, broad `except:` handlers, hardcoded secrets, or success-shaped fallbacks that hide failure.
- Use BLAKE3 for new proof/receipt integrity work; legacy SHA-256 use should already be explicitly justified.
- Import constitutional constants from `core.integration.constants`; use `ADL_GINI_THRESHOLD`, not older/local names.
- Known import paths: `TierPolicy` is in `core.spearpoint.config`; `EvidenceLedger` is in `core.proof_engine.evidence_ledger`; SNR adapters are in `core.iaas.snr_v2_adapter` or `core.apex.snr_apex_engine`; PCI gates use `core.pci.gates.PCIGateKeeper`.
- Pytest uses `asyncio_mode = "auto"` and a 60s timeout. Use `asyncio.run()` for sync-to-async bridges; avoid `get_event_loop().run_until_complete()`. Mark runtime-heavy tests with `@pytest.mark.xdist_group("runtime_heavy")`; use `pytest.importorskip()` for optional dependencies.
- Keep new modules and all imports of those modules in the same change so CI never sees an import of an untracked file.
- `frontend/src/tokens.ts` mirrors constitutional thresholds; keep UI threshold changes synchronized with Python constants.
- In `dema-console/`, `src/bindings/*.ts` are generated from `bizra-omega/bizra-cognition-gateway/bindings/`; never edit them by hand. Run `cargo test -p bizra-cognition-gateway --bin bizra-cognition-gateway` after Rust DTO changes and commit regenerated bindings with the same change.
- The repo already contains MCP configuration in `.mcp.json` and `.codex/config.toml` for Context7, GitHub, memory, SQLite/living memory, BIZRA sovereign/ecosystem tools, and Neo4j.

## References

- `README.md` for product framing and the governed stack.
- `CLAUDE.md` for the most detailed agent/operator context.
- `ARCHITECTURE.md` and `docs/ARCHITECTURE_BLUEPRINT_v2.3.0.md` for system architecture.
- `CONTRIBUTING.md` and `AGENTS.md` for repository guidelines.
- `.github/workflows/ci.yml` for CI gates and exact release checks.
