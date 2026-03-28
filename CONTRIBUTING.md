# Contributing to BIZRA

BIZRA is a proof-native constitutional intelligence system. We welcome contributions that strengthen the integrity and clarity of the system.

## Prerequisites

- Rust nightly (latest stable also supported for non-core work)
- Python 3.11 or later
- Node 20 or later
- Git

## Building and Testing

### Rust Workspace

```bash
cargo test --workspace
```

All 1,016 tests must pass. Clippy and `cargo fmt` must pass with no warnings.

```bash
cargo fmt --check
cargo clippy --workspace -- -D warnings
```

### Python Suite

```bash
pytest
```

All tests must pass. Code must conform to ruff and black standards.

```bash
ruff check .
black --check .
isort --check .
```

## Code Standards

Every claim in the codebase must carry a truth label. These are non-negotiable:

- `[ENFORCEMENT: PROVEN]` — Claim is verified by test, proof, or artifact. No exceptions.
- `[ENFORCEMENT: WIRED]` — Claim is enforced by invariant, type system, or guard. Cannot fail without code change.
- `[OPTIMIZATION: PARTIAL]` — Claim is heuristic, not guaranteed. Side effects possible.
- `[OPTIMIZATION: PLANNED]` — Claim is aspirational. Not yet implemented.

Every function, module, and architectural boundary must declare its truth status in its docstring or comment.

Example:
```rust
/// Returns claim confidence.
/// [ENFORCEMENT: PROVEN] — always returns value in [0.0, 1.0]
fn confidence(&self) -> f64 { ... }
```

## Constitutional Constraints

All code must respect these invariants:

- **Ihsān >= 0.95** — Ethical floor. No claim can stand if it violates ethical bounds.
- **SNR >= 0.85** — Signal-to-noise ratio. Code must be verifiable and clear.
- **ADL Gini <= 0.35** — Inequality constraint. System design must not concentrate power or knowledge asymmetrically.

Violations are non-negotiable rejections.

## Pull Request Requirements

1. **Tests must pass.** All Rust and Python tests pass locally before pushing.
2. **No new broad except handlers.** Exceptions must be specific. Never catch all.
3. **AI-assisted code must be labeled.** If you used Claude, ChatGPT, or similar, mark it:
   ```
   // AI-assisted: Claude 3.5 Sonnet (2025-03-27)
   ```
4. **Documentation must pass the Daughter Test.** Can a non-technical family member understand the intent? Use plain language.
5. **No emojis in code or docs.** Professional tone throughout.

## The Daughter Test

Before submitting a PR with docs or public APIs, ask: "Would my teenage daughter understand what this does and why it matters?" If not, rewrite it.

This is not about simplification. It is about clarity and honesty.

## Code of Conduct

BIZRA is guided by Ihsān principles:

- Act with integrity. Your reputation is your only currency.
- Honor intellectual lineage. Cite the Giants.
- Respect those who came before. Every line of code sits on centuries of thought.
- Disagreement is welcome. Dishonesty is not.

## Giants Registry

When you use an idea from the Giants Registry, cite it inline. This is not bureaucracy. This is reverence.

Format:
```
// Ibn Khaldun (1377): wealth concentration dynamics
// See: GIANTS.md — Khaldunian Curve, ADL Gini constraint
```

## Getting Help

- Read GIANTS.md for intellectual context.
- Read METRICS_CANONICAL.md to understand current system state.
- Ask questions in issues before starting large work.

## Release Gate

Releases require:

- CR (Coverage Ratio) = 1.0000
- SR (Stability Ratio) = 1.0000
- CV (Code Validation) = 1.0000
- G (Giants Lineage) = 1.0000

No exceptions.

---

Thank you for strengthening the proof.
