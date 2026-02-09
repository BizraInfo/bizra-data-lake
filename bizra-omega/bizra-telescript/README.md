# bizra-telescript

**TELESCRIPT-BIZRA BRIDGE** — Mobile agent framework with sovereign ethics.

## Overview

A fusion of General Magic's Telescript (1990s mobile agent technology) with BIZRA's sovereign ethics framework.

### The 9 Primitive Types

1. `Authority` — Who granted the permit (chain of delegation)
2. `Permit` — What capabilities are allowed
3. `Place` — Where agents can go (hosts, services)
4. `Agent` — The mobile code unit itself
5. `AgentState` — Lifecycle: Created, Traveling, Meeting, Frozen, Terminated
6. `Ticket` — Travel authorization with destination and expiry
7. `Meeting` — Agent-to-agent interaction protocol
8. `Region` — Geographic/logical partitioning
9. `AuthorityChain` — Delegated permission hierarchy

## Benchmarks

```bash
cargo bench -p bizra-telescript
```

Covers: authority chain verification, permit verification, Gini coefficient, FATE gate, ticket operations.

## Standing on Giants

General Magic (Telescript), Shannon (SNR), Lamport (BFT), Al-Ghazali (Maqasid al-Shariah), Anthropic (Constitutional AI)

## License

MIT
