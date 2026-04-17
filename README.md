# BIZRA Sovereign Node (SeedOS)

> بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

**Your Sovereign Work OS** — The first personal operating system where your agents think, act on your machine, prove what they did, turn repeated work into reflexes, and mint value from verified impact.

---

## What BIZRA Is

BIZRA is not "AI + blockchain." It is four interlocking systems in one: a **Personal Operating System** that runs entirely on your hardware; an **Agent Market** where skills become composable, tradeable objects; an **Impact Economy** where value is minted from verified work rather than extracted from data or debt; and a **Constitutional Trust Layer** that enforces ethical constraints in code, not policy. Every agent action is signed, receipted, and auditable. The machine works for the human — not the platform.

The governance fabric has three tiers. **PAT-7** is a seven-member council of agents that lives on the user's local machine and handles day-to-day task delegation. **SAT-5** is a five-member system governance council that enforces constitutional constraints across the full node. The **FATE Gate** is a judicial layer backed by Z3 SMT formal verification, requiring Ihsan ≥ 0.95 and hard zeros on ZANN and RIBA before any consequential action is committed. Together, PAT-7 + SAT-5 form a 12-agent parliament. Two agents — P5 Ethicist and S2 Oracle — are permanently frozen as a Gödelian escape valve: they cannot be modified by any runtime instruction.

---

## Architecture

### 5-Layer Governed Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5 — Proof Surface                                    │
│  Signed receipts · Manifests · Benchmark campaigns         │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 — Operator Surface                                 │
│  CLI · TUI · MCP / A2A protocol endpoints                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Runtime Kernel Bridge                            │
│  PyO3 bridge (3.2 MB) · BYOB LLM router · Sippar ledger   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Sovereign Cognition                              │
│  PAT-7 council · SAT-5 governance · FATE judiciary         │
│  72 Python subpackages · HDA memory · SkillNFT engine      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — Constitutional Core (Rust)                       │
│  6 frozen objects · 2,111 LOC · Constitutional membrane    │
│  Fail-closed · Outward-facing · Monotonic gate maturation  │
└─────────────────────────────────────────────────────────────┘
```

### 4-Tier Cognitive Cascade

```
L0  Reflex   ──  O(1) hash lookup            (< 1 ms)
L1  Pattern  ──  Cosine similarity index     (< 10 ms)
L2  Engram   ──  Confidence-gated retrieval  (< 50 ms)
L3  Full PAT ──  GPU inference (PAT-7)       (full context)
```

Gate maturation is monotonic: **Observe → Flag → Throttle (×5) → Reject**. A gate that tightens never softens.

**BYOB LLM support:** LM Studio (deepseek-r1-32b, qwen2.5-32b, llava-7b, qwen2.5-coder-32b) with Ollama fallback. No cloud dependency required.

---

## Quick Start

```bash
# Clone
git clone https://github.com/BizraInfo/bizra-data-lake.git
cd bizra-data-lake

# Build Rust workspace
cd bizra-omega && cargo build --workspace --release

# Build Python environment
python3.12 -m venv .venv-linux && source .venv-linux/bin/activate
pip install -r requirements.txt

# Run the sovereign binary
./bizra-omega/target/release/bizra-node --genesis
```

---

## System Metrics

Full detail in [METRICS_CANONICAL.md](./METRICS_CANONICAL.md). Key numbers:

| Dimension        | Value                           |
|------------------|---------------------------------|
| Total LOC        | 556K+ (251K Python, 116K Rust)  |
| Test suite       | 12,537 (1,122 Rust + 11,415 Python) |
| Rust crates      | 25 in bizra-omega workspace     |
| CI workflows     | 21 active gates                 |
| Commits          | 763                             |
| Release binaries | bizra-node 2.8 MB · bizra-api 5.1 MB |

---

## Standing on the Shoulders of Giants

BIZRA is built on published research, not invented from nothing. See [GIANTS.md](./GIANTS.md) for the full academic and industry lineage.

Seven key papers that directly shaped the system:

1. Bera et al. (Apr 2025) — Hardware-Accelerated Reflex Memory — 7.55× speedup
2. FormalJudge, Zhou et al. (Feb 2026) — Neuro-symbolic oversight via Z3 SMT
3. Krishnamoorthy (Oct 2024) — Cryptographic seal chains for AI lifecycle integrity
4. Aegis Governance (Mar 2026) — Runtime cryptographic policy enforcement
5. LifeBench (Mar 2026) — Multi-source memory benchmark
6. DeepSeek-V3 (Dec 2024) — Aux-loss-free MoE load balancing
7. Wright (Jun 2025) — Epistemic Integrity in AI Reasoning Systems

---

## Constitutional Thresholds

These values are enforced in code. No runtime instruction can override them.

| Parameter    | Threshold | Meaning                                     |
|--------------|-----------|---------------------------------------------|
| IHSAN        | ≥ 0.95    | Minimum excellence score for commitment     |
| SNR          | ≥ 0.85    | Signal-to-noise floor for agent reasoning   |
| ADL_GINI     | ≤ 0.35    | Maximum wealth concentration (Gini ceiling) |
| ZANN_ZERO    | = 0       | No speculative/unjust transactions          |
| RIBA_ZERO    | = 0       | No interest-bearing debt or extraction      |

The constitutional membrane is outward-facing and fail-closed: any breach halts the action, generates a signed rejection receipt, and escalates to SAT-5.

---

## Build Order

| Phase | Goal |
|-------|------|
| Phase 1 | Win one user on one machine — HDA + PAT-7 + FATE + local wallet |
| Phase 2 | Turn skills into market objects — SkillNFT + Proof of Impact + SEED settlement |
| Phase 3 | Turn nodes into ecosystem — A2A + URP leases + capability tokens |
| Phase 4 | Universalize for 8B reach — 3-tap installer + mobile + multilingual |

---

## Economic Model

- **Dual-token:** SEED (transferable utility) + BLOOM (soulbound governance)
- **Proof of Impact (PoI):** Value derived from verified work, never from lending or extraction
- **Anti-RIBA:** No interest-bearing debt, no data harvesting, no rent-seeking subscriptions
- **Zakat:** 2.5% annual obligation, enforced constitutionally
- **Sippar ledger:** Exact arithmetic via Babylonian regular numbers (485 LOC Rust crate)

---

## Documentation

| Document | Role |
|---|---|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md) | Operator runbook |
| [docs/TESTING.md](docs/TESTING.md) | Testing guide |
| [docs/BIZRA-Handover-v1.md](docs/BIZRA-Handover-v1.md) | Production handover (Cycle-5 scope) |
| [docs/BIZRA-Repo-Inventory-v1.md](docs/BIZRA-Repo-Inventory-v1.md) | Full polyglot repo inventory |
| [docs/bizra-trust-compiler-thesis.md](docs/bizra-trust-compiler-thesis.md) | Category thesis (Verificative AI) |
| [docs/dema-cli-manifesto-v1.md](docs/dema-cli-manifesto-v1.md) | Dema CLI manifesto |
| [docs/why-dema-wins.md](docs/why-dema-wins.md) | Product thesis (1 page) |
| [docs/CI-POLICY-AUDIT-v1.md](docs/CI-POLICY-AUDIT-v1.md) | Fail-closed CI audit (22 workflows) |

---

## License

MIT

---

## Contact

**Mohamed Beshr** — m.beshr@bizra.info — Dubai, UAE

*Solo developer. 3+ years. 15,000+ hours. Started Ramadan 2023.*

> بذرة واحدة تصنع غابة — One seed makes a forest.
