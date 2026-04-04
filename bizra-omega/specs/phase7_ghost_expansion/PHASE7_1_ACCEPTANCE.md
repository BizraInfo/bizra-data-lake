# Phase 7.1 — EventBus + Live Ghost | Acceptance Criteria

**Sprint:** 7.1
**Scope:** Wire event bus into TUI cockpit. Ghost feed reacts to live events.
**Constraint:** No new crates. No shadow state. Same truth path.

---

## Success Criteria

1. **Event channel exists** — tokio mpsc channel wired into TUI event loop alongside crossterm events
2. **3 event classes flow** — receipt.created, trust.changed, mission.completed propagate to Ghost feed
3. **Ghost feed shows events** — recent events rendered as timestamped entries in the Ghost feed widget
4. **No regression** — existing 19 CLI tests + 725 workspace tests still pass
5. **Clippy + fmt clean** — zero warnings, zero formatting drift
6. **One headless test** — proves event injection into DashboardData and rendering in Ghost feed
7. **Manual + periodic refresh preserved** — `r` key and 5-second auto-refresh still work unchanged
8. **No shadow state** — events derive from existing backends (receipts.jsonl, trust checks, mission output)

## Out of Scope

- Wallet view
- Memory/skills view
- Mission input in TUI
- Receipt detail view
- Service health probes
- Export functionality

## Definition of Done

```
cargo fmt --all -- --check     # PASS
cargo clippy --workspace -- -D warnings  # PASS (or pre-existing only)
cargo test -p bizra-cli        # ALL PASS (19 existing + new)
cargo build -p bizra-cli --release  # CLEAN
```

Ghost feed shows at least one live event entry after a `bizra mission` execution.
