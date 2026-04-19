# BIZRA First Fire Dry-Run Checklist (Cycle-8 Day 11-12)

بسم الله الرحمن الرحيم

**Purpose:** verify the T=0 install + first-use flow on a tester's machine
BEFORE the 10k+ distribution bullet fires. One tester = one full run of this
checklist. The operator (Mumo) watches results; testers need not be technical.

**Prerequisite:** 5 tester names known + confirmed to Mumo. If fewer than 5,
halt pre-fire.

---

## Per-tester checklist

### Phase 0 — Setup (2 min)

- [ ] Tester has Linux (Ubuntu 22.04+), macOS (arm64 or x86_64), or Windows 11.
- [ ] Tester has a working terminal (bash / zsh / pwsh).
- [ ] Tester has `curl` available.
- [ ] Tester has at least 500 MB free in `$HOME`.
- [ ] Tester has a `~/Downloads` folder with at least 3 files (doesn't matter what).

### Phase 1 — Install (< 5 min)

- [ ] Run: `curl -fsSL https://bizra.ai/install.sh | sh`
- [ ] Installer detects platform correctly.
- [ ] Installer fetches the tarball.
- [ ] Installer verifies SHA-256 against manifest (no "SHA MISMATCH" errors).
- [ ] Installer extracts + installs `dema` + `bizra-cognition-gateway` to
      `$HOME/.bizra/bin` (or `$HOME/.cargo/bin` if present).
- [ ] Installer prints the installed version + SHA-256 of each binary.
- [ ] `$HOME/.bizra/bin` is on `$PATH` (or instructions to add it were shown).
- [ ] No unexplained network calls (tester can confirm with `strace` or
      equivalent — optional).

### Phase 2 — First run (< 3 min)

- [ ] Run: `dema --version`
- [ ] Version string prints. Matches the install-phase version.
- [ ] Run: `dema health` (expected to say "gateway not running, start with: dema start")
- [ ] Run: `dema start` (or the documented command to spin up the local gateway)
- [ ] Gateway starts; binds to 127.0.0.1:7421.
- [ ] `dema health` now returns green.
- [ ] Run: `dema register-resource --kind filesystem --id ~/Downloads --allowlisted`
- [ ] Registration succeeds.
- [ ] Run: `dema organize ~/Downloads`
- [ ] Output includes: permitted, receipted, sealed, 5 gate verdicts (all Permit),
      chain_head, listing_digest.
- [ ] Chain head is a 64-char lowercase hex string.

### Phase 3 — Re-verify (< 2 min)

- [ ] Run: `dema chain`
- [ ] Chain length ≥ 2 (principal activation + organize mission).
- [ ] Run: `dema receipt <chain_head>` (copy the chain_head hex from Phase 2).
- [ ] Receipt decodes and shows: kind=MissionExecuted, claim_ref, evidence_hash,
      timestamp, listing digest matching Phase 2.
- [ ] Run: `sha256sum $(which dema)` and confirm the hash matches the
      install-time declared SHA.

### Phase 4 — Witness (if witness peer is configured)

- [ ] Run: `dema chain-head` to get current head.
- [ ] Run: `curl https://<witness-peer>/witness/head/<node-id>` to get the
      witness's observation.
- [ ] Witness-reported chain_head matches the local chain_head.
- [ ] Witness-reported signature verifies against Node0's declared pubkey.

### Phase 5 — Halt-on-failure reporting

If ANY step failed, the tester reports:
- which step failed
- the exact error output
- the platform (OS + arch + kernel version)
- the install-time-declared SHA and the locally-computed SHA

Mumo decides: (a) fix and retry, (b) mark the tester's platform as
"not supported at T=0", or (c) delay the fire.

---

## Aggregation (operator's view)

For the fire to proceed, at minimum:
- [ ] 5 / 5 testers completed Phase 0-3 successfully.
- [ ] 3 / 5 testers completed Phase 4 successfully (witness may not be
      configured for all; minimum 3 is the witness-grade quorum at T=0).
- [ ] 0 / 5 testers reported SHA mismatches (critical — any mismatch means
      the install surface is compromised).
- [ ] 0 / 5 testers reported data leaving the local machine during Phase 2-3
      (confirms local-first posture).

If aggregation fails: **do not fire.** Delay. Investigate. Re-test.

---

## Tester identities (to be filled)

| # | Name | Platform | Contact | Completed |
|---|---|---|---|---|
| 1 | TBD | TBD | TBD | — |
| 2 | TBD | TBD | TBD | — |
| 3 | TBD | TBD | TBD | — |
| 4 | TBD | TBD | TBD | — |
| 5 | TBD | TBD | TBD | — |

**This table MUST be filled before the harness can execute.** 5 tester names is
the minimum human gate for Day 11-12.

---

*Close it. Prove it. Reveal it.*
