# Repository Guidelines

## Mission & Operating Model
`bizra-hunter` is the security-focused hunting engine inside the `bizra-omega` workspace. Treat this repo area like a professional multi-agent think tank: maximize signal, keep outputs reproducible, and never trade safety for speed. Every change must preserve the three cascade gates in `src/cascade.rs` (`Ethics`, `Legal`, `Technical`) and the non-weaponized proof standard enforced by `src/poc.rs`.

## Project Structure & Module Organization
Hunter code lives in `bizra-hunter/`. Core runtime and orchestration are in `src/hunter.rs` and `src/pipeline.rs`. Domain modules are split by responsibility: `entropy.rs` for SNR scoring, `evm.rs` for bytecode decoding, `invariant.rs` for deduplication, `submission.rs` for bonded validation, and `poc.rs` for safe demonstration output. Use `tests/integration.rs` for end-to-end pipeline behavior and `benches/snr_pipeline.rs` for throughput work. Design notes live in `README.md`, `CLI_GIANTS_TUI.md`, and `PERFORMANCE_REPORT.md`.

## Build, Test, and Development Commands
- `cargo build -p bizra-hunter` builds the Hunter crate.
- `cargo run -p bizra-hunter --bin bizra-hunter-snr -- health` checks gate and runtime health.
- `cargo run -p bizra-hunter --bin bizra-hunter-snr -- decode --bytecode 0x...` inspects one bytecode sample.
- `cargo test -p bizra-hunter` runs crate tests.
- `cargo bench -p bizra-hunter` runs the SNR pipeline benchmark.
- `cargo clippy -p bizra-hunter --all-targets -- -D warnings` enforces the lint bar.
- `cargo fmt --all -- --check` verifies formatting before review.

## Coding Style & Naming Conventions
Use Rust 2021 with 4-space indentation. Follow existing naming: `snake_case` for modules/functions, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for constants. Prefer predictable memory behavior, crate-local abstractions, and small focused modules. Do not bypass `CriticalCascade`, `SafePoC`, or `BondedSubmission` to “speed things up”; those are core safety controls, not optional layers.

## Testing Guidelines
Add tests with every heuristic, gate, or vulnerability-rule change. Mirror existing patterns such as high-entropy positive cases, low-entropy filter cases, deduplication checks, and health/status assertions. Name new files or tests clearly by behavior, for example `reentrancy_filter_tests.rs` or `test_scan_filters_low_entropy`.

## Commit, PR, and Security Rules
Use Conventional Commits like `feat(hunter):`, `fix(clippy):`, or `test(hunter):`. PRs should state the threat class touched, list commands run, and explain operator impact. Include benchmark deltas for performance work and sample CLI output when behavior changes. Never add exploit code, live target secrets, or instructions that enable value extraction; Hunter proofs must stay detection-only and professionally reportable.
