# BIZRA Constitutional Kernel

**Mission:** BIZRA is a decentralized developmental AGI operating system that turns every human into a sovereign node, every node into a living seed, and every verified act of growth into shared intelligence, capability, and value.

**Version:** v1.0.0 | **Anchor:** `bizra-constitution/constitution.toml` v5.0.0-GENESIS (673 lines)

---

## Five Non-Negotiable Invariants

| # | Invariant | Threshold | Enforcement | Source |
|---|-----------|-----------|-------------|--------|
| I-1 | Excellence or rejection | Ihsan >= 0.95 | `UNIFIED_IHSAN_THRESHOLD`, fail-closed gate | `core/integration/constants.py:110` |
| I-2 | Signal or quarantine | SNR >= 0.85 | `UNIFIED_SNR_THRESHOLD`, hard floor | `core/integration/constants.py:198` |
| I-3 | Justice or block | Gini <= 0.35 | `ADL_GINI_THRESHOLD`, transaction rejection | `core/integration/constants.py:243` |
| I-4 | Sovereignty absolute | Keys = LOCAL_ONLY | Ed25519 generated on-device, never transmitted | `core/sovereign/genesis_identity.py` |
| I-5 | Accountability total | Every action = receipt | SHA-256/BLAKE2b hash-chained evidence ledger | `core/proof_engine/evidence_ledger.py` |

Every invariant is machine-enforced. No override. No exception. Change requires constitutional amendment.

---

## Seven-Layer Stack

| Layer | Name | Source | Tests |
|-------|------|--------|-------|
| L0 | Human Seed | الرسالة + البذرة — constitutional anchor | — |
| L1 | Sovereign Node | `core/sovereign/genesis_identity.py` — Ed25519, HD agent keys | 332 |
| L2 | Agentic Development | `core/sovereign/mission.py` — PAT-7 + SAT-5 pipeline | 38 |
| L3 | Verification | `core/proof_engine/evidence_ledger.py` — PoI receipts, hash chains | 50+ |
| L4 | Learning | `core/sovereign/seed_engine.py` — SEED/SPROUT/TREE/FOREST tiers | 46 |
| L5 | Economic | `core/constitutional/algorithms.py` — 15 algorithms, Zakat, SEED/BLOOM | 100+ |
| L6 | Civilizational | `core/federation/node.py` + `core/a2a/` — gossip, BFT, Asabiyyah | 60+ |

Total: 8,237+ tests. 113K LOC Python. 137K LOC Rust. Every layer has code, tests, and evidence.

---

## Node Lifecycle

| Stage | Score | What Happened |
|-------|-------|---------------|
| Seed | 0.00 - 0.10 | Install Node0, generate Ed25519 keypair |
| Node | 0.10 - 0.20 | First mission completed with Ihsan >= 0.85 |
| Apprentice | 0.20 - 0.35 | 10+ qualified episodes, 50%+ qualification rate |
| Builder | 0.35 - 0.55 | First reflex compiled (3 consecutive qualified) |
| Verifier | 0.55 - 0.70 | Trusted to attest others' work |
| Mentor | 0.70 - 0.85 | Skills published to marketplace |
| Catalyst | 0.85 - 1.00 | Network effect multiplier, FOREST tier |

Parallel to agent skill tree (Novice to Grandmaster). Both earned through verified work.

---

## Reward Loop

```
1. EARN    — Complete a mission (work verified by SAT-5 Oracle)
2. VERIFY  — Pass Ihsan gate (6-dim tensor, fail-closed)
3. COMPILE — 3+ consecutive qualified episodes -> reflex precipitation
4. TRADE   — Compiled reflex -> skill on marketplace -> SEED tokens
5. REPEAT  — Each cycle raises sovereignty score, unlocking higher-trust work
```

Zakat (2.5%) flows to Community Fund at every mint. Gini gate prevents concentration. Khaldunian curve throttles minting as inequality rises.

---

## Node Value (KPI)

```
NodeValue = (Potential x Activation x Quality x Compounding x Synergy) ^ (1/5)
```

Geometric mean. All factors normalized [0, 1]. No factor dominates through volume or age.

| Factor | Source | Normalization | Range |
|--------|--------|---------------|-------|
| Potential | `seed_engine.potential().sovereignty_score` | Direct | 0 - 1 |
| Activation | `episodes / days` | `min(DAM / 5.0, 1.0)` | 0 - 1 |
| Quality | Mean Ihsan (6-dim composite) | Direct | 0 - 1 |
| Compounding | Age + streak | `(1 - e^(-days/365)) x (0.7 + 0.3 x streak/10)` | 0 - 1 |
| Synergy | Asabiyyah x attestations | Pre-federation: 1.0 | 0 - 1 |

One number. Five inputs. All bounded. The investor metric, the marketing metric, and the constitutional health metric are the same number. NodeValueEngine is read-only over SeedEngine — single source of truth, no duplicate counters.

---

## The Moat

Every new node brings hardware (compute), data (knowledge), and intelligence (compiled reflexes). Performance improves with growth. Value accrues to nodes, not to the platform.

```
OpenAI:  users -> data -> model -> users     (value accrues to OpenAI)
BIZRA:   nodes -> work -> skills -> nodes     (value accrues to nodes)
```

---

**8,237 tests. 22 Rust crates. 58 Python packages. Every claim above traces to running code.**
