# Claim Discipline for Node0 and URP

## Purpose

Protect BIZRA from overclaiming while still communicating the power of the system. Every Node0 or URP claim must be truth-labeled.

## Truth Labels

| Label | Use when |
|---|---|
| MEASURED | There is a receipt, test, audit, or reproducible artifact. |
| PARTIAL | Some components work, but the end-to-end outcome is not complete. |
| PLANNED | Architecture exists as a plan or scaffold, not a proven runtime capability. |
| DIRECTIONAL | It is a vision, principle, or long-term intent. |
| PROHIBITED | It implies proof, scale, safety, or financial result that does not exist. |

## Approved Node0 Language

Safe:

- "Node0 is BIZRA's first sovereign-node foundation."
- "Node0 is documented as GO through internal single-node readiness tiers and NO-GO for external production until remaining gates close."
- "BIZRA is moving from single-node proof toward a private multi-device pilot."
- "The next milestone is a two-node signed-receipt handshake."

Use with evidence link:

- "Node0 emits signed receipts."
- "BIZRA uses BLAKE3-chained and Ed25519-signed receipt concepts."
- "The mission lifecycle is implemented in the Rust workspace."

## Approved URP Language

Safe:

- "URP means Universal Resource Pool."
- "URP is the planned shared resource substrate where sovereign nodes can coordinate without surrendering local authority."
- "The production URP bootnode is not yet proven; the private pilot will test the first signed receipt exchange across devices."

## Prohibited or Unsafe Until Proven

Do not say:

- "BIZRA is fully decentralized today."
- "URP is live at production scale."
- "Genesis 100 is operational."
- "Any device can join today."
- "Trustless network."
- "Guaranteed secure."
- "First in the world" unless independently substantiated.
- "No telemetry" unless privacy policy and implementation evidence are published.
- Any exact latency, cost, benchmark, node-count, or pass-rate claim without a receipt.

## Public Copy Pattern

Use:

> BIZRA is the Seed of Sovereign Intelligence: a human-first AI ecosystem built on meaning, proof, and Ihsan. Node0 proves the seed can live alone; the next private pilot will test how trusted user nodes connect through signed, verifiable receipts.

Avoid:

> BIZRA is already a decentralized AGI network with production-ready trustless federation.

## Upgrade Rule

A claim can move from PLANNED to MEASURED only when:

- The test or receipt exists.
- The methodology is understandable.
- The artifact can be found by a reviewer.
- The language matches exactly what was measured.
