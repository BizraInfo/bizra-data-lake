# DEPRECATED — Moved to Unified Workspace

The 4 crates from `native/` have been merged into `bizra-omega/`:

| Crate | Old Location | New Location |
|-------|-------------|--------------|
| bizra-hooks | `native/bizra-hooks/` | `bizra-omega/bizra-hooks/` |
| bizra-memory | `native/bizra-memory/` | `bizra-omega/bizra-memory/` |
| fate-binding | `native/fate-binding/` | `bizra-omega/fate-binding/` |
| iceoryx-bridge | `native/iceoryx-bridge/` | `bizra-omega/iceoryx-bridge/` |

## Build from unified workspace

```bash
cd bizra-omega
cargo build --workspace
cargo test --workspace
```

The unified workspace is `bizra-omega/` with 18 crates, 610+ tests.

This `native/` directory will be removed in a future cleanup.
