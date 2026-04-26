# Minimum Hardware Profile for Private User Nodes

## Purpose

Define the minimum device profile for the first private pilot. These are pilot requirements, not final consumer requirements.

## Tier 0: Operator Node0 Reference

Node0 remains the strongest machine and source of operational truth for the pilot.

Recommended:

- Modern high-performance CPU.
- 64-128 GB RAM for heavy local models and indexing.
- Discrete GPU preferred for local inference.
- 100 GB free disk for repo, models, logs, and evidence.
- Stable broadband connection.
- Linux or WSL2 environment.

## Tier 1: Private Pilot User Node

Minimum:

- 4 CPU cores.
- 16 GB RAM.
- 20 GB free disk.
- Linux, WSL2, or compatible dev environment.
- Stable network connection to Node0 or pilot relay.
- Python 3.11+.
- Rust stable if local Rust build is required.

Recommended:

- 8 CPU cores.
- 32 GB RAM.
- 50 GB free disk.
- Local LLM runtime optional.
- SSD storage.

## Tier 2: Lightweight Receipt-Only Node

A lightweight node may participate in receipt exchange without local model inference.

Minimum:

- 2 CPU cores.
- 8 GB RAM.
- 10 GB free disk.
- Ability to run the node identity and receipt verification path.

Use this tier for early handshake testing only. Do not use it to benchmark mission performance.

## Not Supported in First Pilot

- Mobile-only onboarding.
- Browser-only node execution.
- Shared public machines.
- Devices where the owner cannot control the filesystem or keys.
- Devices that cannot preserve local state across restart.

## Readiness Labels

| Label | Meaning |
|---|---|
| `ready` | Meets pilot profile and passes local health. |
| `degraded` | Can run partial flow but cannot complete handshake or restart recovery. |
| `blocked` | Cannot safely participate. |

## Hardware Claim Discipline

Do not claim BIZRA runs on every device until the installer and receipt path have been tested across those device classes. Say: "The private pilot supports selected Linux/WSL2 devices first; broader device support is planned after evidence is collected."
