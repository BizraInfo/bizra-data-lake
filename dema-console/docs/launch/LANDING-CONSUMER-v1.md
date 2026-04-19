# bizra.ai — consumer landing copy v1 (DRAFT)

بسم الله الرحمن الرحيم

**Status:** DRAFT (Cycle-8, 2026-04-19). Consumer-first path.
**Positioning lock (Phase 1 collapse):** U1=consumer · U2=no-direct-buyer-access · U3=only-with-help
**Tone:** constitutional prophecy under proof discipline. Not a startup brochure.
**Target surface:** `bizra.ai/` (marketing site, not the dema-console app).

---

## Hero

> ## DEMA seals reality.
> ## Organize is the first proof.

**One command. Sixty seconds. No account. No cloud. No cost.**

```
curl -fsSL https://bizra.ai/install.sh | sh
dema organize ~/Downloads
```

You get a cryptographically sealed manifest of your digital clutter — receipted, replayable, and verifiable by any skeptical stranger on any machine.

---

## Why DEMA exists

Every app you use claims things about your data that only the app can verify. You have no way to check. You trust because you have no other choice.

DEMA replaces that trust with **proof**. When DEMA seals something, a skeptical stranger can, using only public tooling, verify in bounded time that the claim is true — or produce transferable evidence that it's false.

DEMA is not an AI assistant. DEMA is a **governed runtime** that turns your intent into lawful, receipted, replayable action. The first thing it proves is the hardest one: that it didn't lie to you.

---

## Your first proof

1. Install (< 5 minutes, reproducible build, SHA-256 verified):

    ```
    curl -fsSL https://bizra.ai/install.sh | sh
    ```

2. Register an allowlisted path — DEMA will NEVER touch anything else:

    ```
    dema register-resource --kind filesystem --id ~/Downloads --allowlisted
    ```

3. Seal the first mission:

    ```
    dema organize ~/Downloads
    ```

You'll see:
- Five constitutional gate verdicts, each PERMITTED with a score.
- A chain-sealed `MissionExecuted` receipt with a BLAKE3 hash.
- A deterministic listing digest that reproduces byte-identical on any machine with the same directory.

Keep the receipt. Anyone can replay it: `dema receipt <hash>`.

---

## How you verify DEMA didn't lie

The Four-Modality Golden Standard. DEMA commits to all four at T=0:

| Modality | What it means | How you check |
|---|---|---|
| **Cryptographic** | BLAKE3 hash-chain, Ed25519 signatures | `sha256sum $(which dema)` → compare to the install-time manifest |
| **Empirical** | Same input → same output on any machine | Run `dema organize <dir>` twice — receipt IDs match |
| **Formal (TESTED)** | 309 Rust tests + 77 gateway tests, green under `cargo clippy -D warnings` | `git clone` the source, `cargo test` — same result |
| **Economic (witness-grade)** | At least one independent witness observes your chain head | `curl <witness-peer>/witness/head/<your-node-id>` — watch for divergence |

Full formal proof (Isabelle/HOL-grade) and bonded cryptoeconomic enforcement are explicitly Horizon — not at T=0, not claimed to be.

---

## Constitutional anchors (the five invariants)

Every lawful act in DEMA passes through five gates, fail-closed, no exceptions:

- **ZANN_ZERO** — no claim without evidence
- **CLAIM_MUST_BIND** — evidence must cryptographically bind to the claim
- **RIBA_ZERO** — no extractive economic pattern
- **NO_SHADOW_STATE** — what you see is what the kernel sealed
- **IHSAN_FLOOR** — quality floor ≥ 0.95 for any permit

Read the full Manifest: `bizra.ai/manifest`

---

## What DEMA is NOT

- **Not a chatbot.** No conversation surface, no LLM wrapping, no model calls in the critical path.
- **Not a cloud service.** Everything runs local on your machine. No account. No server.
- **Not an agent framework.** The kernel is not programmable through natural language.
- **Not a startup pitch.** BIZRA has been built solo for three years. No VC, no tokens, no ads.

---

## Built by one operator in three years

BIZRA (the kernel behind DEMA) is the result of ~15,000 hours of solo engineering, anchored in two Arabic founding texts written in Ramadan 2023. Independently validated in October 2025 by academic convergence (arXiv:2510.13857, Xu et al., CUHK) which theorizes, six months after BIZRA was already building it, the same architecture — Kernel-as-Governor, Agent Constitution Framework, Evaluation-Driven Development Lifecycle — that DEMA implements in Rust.

That paper is external evidence of convergence, not source material. DEMA came first.

---

## Links

- Install: `bizra.ai/install.sh`
- Manifest: `bizra.ai/manifest`
- First Fire Doctrine: `bizra.ai/doctrine`
- Proof-of-Priority: `bizra.ai/priority`
- ArbiterOS ↔ BIZRA mapping: `bizra.ai/arbiteros-mapping`
- Source: `github.com/BizraInfo/bizra-data-lake`

---

*Close it. Prove it. Reveal it.*

الحمد لله

---

## Draft notes (NOT for public site)

- **Witness peer section omitted from public copy** until at least one real witness peer is named and running. Currently TBD.
- **Install command (`curl | sh`)** must be SHA-256 verifiable once cargo-dist publishes the release. Until then, the site is copy-only, no live `install.sh`.
- **Tester list omitted.** Once 5 testers complete D5 Daughter Test, a short "verified by" line may appear without personal details unless they explicitly opt in.
- **No call-to-action to "sign up" / "join" / "reserve" / any mailing list.** DEMA is install-first. Email capture on first visit is a RIBA-adjacent pattern we avoid.
- **No social proof theater** (testimonials, user counts) until real data exists.
- **No pricing section** on consumer path. Consumer is free-to-install. Enterprise is a separate brief (held internally).
