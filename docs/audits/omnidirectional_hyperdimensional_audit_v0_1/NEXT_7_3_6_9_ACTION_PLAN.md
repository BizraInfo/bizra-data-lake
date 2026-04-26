# Next 7-3-6-9 Action Plan — BIZRA v0.1

**BIZRA 7-3-6-9 structure:** 7 lenses · 3 execution passes · 6 validation checks · 9 next actions.

---

## 7 Lenses

1. **L1 — Architecture** (Node0, DEMA, PAT/SAT, URP, canon separation)
2. **L2 — Security** (secrets, receipts, SSRF, injection, blast radius)
3. **L3 — Performance** (measured vs simulated vs target vs unverified)
4. **L4 — Documentation** (handoff, ADR, DoD, public/private split)
5. **L5 — Supply chain** (lockfiles, SBOM, audit tooling)
6. **L6 — Public claims / Ihsan** (Law of Assumption, claim register)
7. **L7 — Node0 activation readiness** (Tier A-E DoD)

## 3 Execution passes

### Pass 1 — Bulletproof (P0 + low-effort Tier D)

Target: close every HIGH-severity item that does not require a separate lane.

- Maintain zero secret-pattern findings; keep scanner gate current.
- Add pre-commit secret scanner.
- Remove `subprocess(shell=True)`.
- Visual QA of media kit.
- Arabic reviewer pass.
- Operator sign-off on claim-safe launch copy.

**Estimated duration:** 1 operator day.

### Pass 2 — Truth & Site (P2)

Target: make bizra.ai match the internal claim discipline.

- Remove or receipt-ify C4 / C5 / C7 / C9.
- Replace hero with claim-safe English + Arabic.
- Publish privacy policy OR soften C1 / C2.
- Add OG tags to SPA shell.
- Verify PR #50 merged.
- Document operator kill-switch.

**Estimated duration:** 1–2 operator days (blocked on external bizra.ai repo access).

### Pass 3 — Launch & Scale (P3 → P4 → P5)

Target: organic launch → paid ads → Genesis 100 planning.

- Silent profile foundation (Phase 1).
- Launch moment (Phase 2).
- First-week support (Phase 3).
- Once organic telemetry exists → paid-ad preflight (P4.1–P4.6).
- Once organic + paid land → Genesis-100 planning (P5.1–P5.8).

**Estimated duration:** 1 week (Phase 2-3) + multi-week for Phase 4-5.

## 6 Validation checks

1. **Re-run audit engine.** `python3 -m tools.audit.omni_audit.run_audit ...`. Verify:
   - PROHIBITED count stable or down.
   - NEEDS_REWRITE count stable or down.
   - Secret matches down to 0 actual credentials (false-positives allow-listed).
   - Dep gaps reducing (Cargo.lock for filedfs/desktop/rust; SBOM generated).
2. **Cross-language sync.** Re-run CI stage 2 (schema + sync); confirm `constants.py` ↔ `bizra-core/src/lib.rs` parity.
3. **Receipt-chain integrity.** Pick 3 recent receipts from the chain; verify signatures + previous-hash linkage.
4. **Claim vs. reality diff.** Open bizra.ai in a real browser + headless-Chromium; compare against `PUBLIC_CLAIMS_REGISTER.md`. Zero C-class claims live.
5. **Media-kit visual QA pass.** Every raster used on a platform has been human-verified at post size.
6. **Operator load check.** After bulletproofing, operator should have < 3 open "urgent" lanes. More than 3 = drift risk; re-land the plane.

## 9 Next actions (concrete, ordered)

| # | Action | Owner | Dep | Unblocks |
|---|---|---|---|---|
| 1 | Maintain zero secret-pattern findings; keep scanner gate current; rotate any real credential | operator | — | Security GO |
| 2 | Visual QA of 12 concept boards + 11 raster exports | operator | — | Organic launch |
| 3 | Arabic reviewer pass on launch copy §1–§4 | operator + Arabic reviewer | — | Organic launch |
| 4 | Operator sign-off on `CLAIM_SAFE_LAUNCH_COPY.md` §1–§6 | operator | #2, #3 | Organic launch |
| 5 | Remove / receipt-ify bizra.ai C4 / C5 / C7 / C9 | operator + web lead | — | Tier D + paid ads |
| 6 | Publish privacy policy OR soften "no telemetry" + "no cloud" to cloud-optional framing | operator | — | Tier D |
| 7 | Draft Canon Store Ingestion Gate ADR (spec only, no code) | architecture lead | typed-auth | Tier D + E |
| 8 | Hot-path `.unwrap()` audit in receipt / mission crates | runtime lead | — | Tier B watchlist → green |
| 9 | Author node-onboarding runbook + min-hardware profile | operator + runtime lead | — | Tier D + E |

## Stop conditions

- If P0.1 surfaces a real credential leak → **HALT** all other work. Rotate immediately. Document incident.
- If the audit engine re-run shows PROHIBITED or NEEDS_REWRITE count *increasing* → **HALT** public-copy work. Find the regression source.
- If any finding requires mutating canon / runtime / git / public surfaces beyond already-authorized scope → **PAUSE** and get typed authorization.

## Commit to the plane

After 9 next actions close: **land the plane.** Re-run audit quarterly. Don't stack P5 on top of post-landing completion. The land-the-plane discipline is itself one of the golden gems.
