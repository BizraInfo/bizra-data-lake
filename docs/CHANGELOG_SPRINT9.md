# Sprint 9 Changelog — CI Stabilization

## fix(lint): Clippy 1.91 + isort + black compliance [ddd3139e]

### Rust (Clippy 1.91 new lints resolved)
- `assign_op_pattern`: `islamic_finance.rs` — use compound assignment
- `manual_is_multiple_of`: factorization loops — use `.is_multiple_of()` or `#![allow]`
- `new_without_default`: add `Default` impl for `AutopoieticState`
- `approx_constant`: use `std::f64::consts::E` instead of literal `2.7183`
- `too_many_arguments`: `#[allow]` on `execute_full_flow` (protocol boundary function)
- `items_after_test_module`: `#![allow]` on `proof_pyramid_e2e.rs` (smoke-test `main()`)
- `useless_format`: replace `format!(literal)` with `.to_string()`
- `double_ended_iterator_last`: use `.next_back()` instead of `.last()`
- `identity_op`: remove redundant `& 0xFF` on `u8`
- `manual_clamp`: simplify `min_f64` fold chain
- `dead_code`: `#[allow]` on bench stubs and PLANNED `event_bridge` field
- `unused_import`: remove `bb`, `IHSAN_THRESHOLD`, `GatewayError`

### Python (isort + black)
- `isort --profile black`: 170+ files import-sorted
- `black`: 276 files reformatted

### Verification
- `cargo clippy --workspace --all-targets -- -D warnings` → **0 errors, 0 warnings**
- `isort --check --profile black .` → **clean**
- `black --check .` → **clean**
