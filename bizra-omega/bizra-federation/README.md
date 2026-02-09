# bizra-federation

**BIZRA Federation** — Distributed Sovereignty via SWIM Gossip + BFT Consensus.

## Overview

Cryptographically-signed distributed coordination for BIZRA sovereign nodes:

- **Gossip Protocol** — SWIM-based failure detection and membership management
- **Consensus Engine** — BFT voting with Ed25519-signed votes and quorum detection
- **Federation Node** — Lifecycle management, peer discovery, pattern proposals
- **Bootstrap** — Seed node discovery and initial cluster formation

## Security

All federation communication is cryptographically signed:
- Consensus votes: Ed25519 signatures verified before counting
- Gossip messages: Ed25519 signatures prevent spoofing

## Key Types

| Type | Purpose |
|------|---------|
| `GossipProtocol` | SWIM membership with signed ping/ack/leave |
| `ConsensusEngine` | BFT proposal → vote → quorum → commit pipeline |
| `FederationNode` | Top-level node lifecycle (start/stop/propose) |
| `Bootstrapper` | Seed-based peer discovery |

## Standing on Giants

Lamport (BFT), Das (SWIM), Bernstein (Ed25519)

## License

MIT
