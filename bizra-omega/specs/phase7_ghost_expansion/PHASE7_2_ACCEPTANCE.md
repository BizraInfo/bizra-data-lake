# Phase 7.2 — Mission View + Receipt Deep View | Acceptance Criteria

**Sprint:** 7.2
**Scope:** Receipt detail widget, scrollable receipt list, mission input in TUI.
**Constraint:** No new crates. No shadow state. Same truth path. Borrow-only widgets.

---

## Success Criteria

1. **Receipt detail view exists** — new widget shows full receipt: Ihsan score, SNR, model, signature status, chain link, state history
2. **Scrollable receipt list** — Up/Down or j/k navigates all receipts (not capped at 10), Enter opens detail
3. **Mission input in TUI** — 'm' key activates mission input mode, Enter submits through mission_bridge, result appears in Ghost feed events
4. **Receipt card after mission** — after mission completes, receipt summary rendered inline with Ihsan + SEED + hash
5. **No regression** — existing 30 CLI tests still pass
6. **Clippy + fmt clean** — zero warnings, zero formatting drift
7. **New tests** — receipt detail rendering test, receipt list navigation test, mission input flow test
8. **No shadow state** — all data from existing backends (ledger, mission_bridge, trust)

## Out of Scope

- Wallet view
- Memory/skills view
- Gate confirmation dialogs (defer to 7.3)
- Export functionality
- Service health probes

## Definition of Done

```
cargo fmt --all -- --check     # PASS
cargo clippy --workspace -- -D warnings  # PASS (or pre-existing only)
cargo test -p bizra-cli        # ALL PASS (30 existing + new)
cargo build -p bizra-cli --release  # CLEAN
```

Receipt detail shows Ihsan score when selecting a receipt in the TUI.
