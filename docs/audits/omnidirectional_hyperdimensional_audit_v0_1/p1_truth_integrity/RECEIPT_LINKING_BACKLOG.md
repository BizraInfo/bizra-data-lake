# Receipt-Linking Backlog — PROOF_REQUIRED Claims

**Date:** 2026-04-24 GST
**Purpose:** Group the 367 `PROOF_REQUIRED` claims by the kind of evidence
each one needs. This pass **does not rewrite** any PROOF_REQUIRED claim; it
only defines what receipt would close each bucket.

**Constraint:** This document is an engineering-planning artefact only.
Nothing here is to be published.

---

## Distribution (from refreshed `claims_register.json`, 2026-04-24)

| # | Category | Count | % |
|---|----------|-------:|--:|
| 1 | Cryptography claim | 122 | 33.2% |
| 2 | Hashing claim | 122 | 33.2% |
| 3 | Formal-verification claim | 62 | 16.9% |
| 4 | Post-quantum claim | 22 |  6.0% |
| 5 | Ihsan-threshold claim | 17 |  4.6% |
| 6 | Local-only / no-cloud claim | 12 |  3.3% |
| 7 | Relative-performance claim |  8 |  2.2% |
| 8 | Zero-telemetry claim |  2 |  0.5% |
| | **Total** | **367** | 100% |

---

## Evidence requirements by group

### 1. Cryptography claim (122)

**Typical claim shape:** "Ed25519 signature", "ECDSA", "signed with KEY_X", "asymmetric".

**Required receipt:**
1. Commit hash + file path to the actual signing code
   (e.g. `bizra-omega/bizra-core/src/canonical_receipt.rs`).
2. A sample signed receipt hex dump (no keys leaked).
3. The `constants.py` / Rust constants block defining the key size /
   algorithm spec.
4. Chain-of-custody note: who holds the public key, where it is published.

**Receipt artefact to publish:** `/trust/crypto.md` with verifiable sample
+ commit link.

### 2. Hashing claim (122)

**Typical claim shape:** "BLAKE3-chained receipts", "SHA-256 of manifest",
"deterministic hash".

**Required receipt:**
1. Commit + file path to the hash call site.
2. Reproducible example: input + expected output, re-runnable from the
   command line (`python -c "..."` or a test fixture).
3. Test file that enforces hash stability.

**Receipt artefact:** `/trust/hashing.md` with the re-run command + commit.

### 3. Formal-verification claim (62)

**Typical claim shape:** "Z3-verified", "formally verified gate", "Lyapunov
stability proof".

**Required receipt:**
1. The actual Z3 / Dafny / Coq artefact checked into the repo.
2. CI job that re-runs the proof on every PR.
3. A one-paragraph abstract written by a human who can defend the claim.

**Receipt artefact:** `/trust/formal.md` with proof files + CI job link.

### 4. Post-quantum claim (22)

**Typical claim shape:** "ML-DSA post-quantum", "Dilithium", "quantum-safe".

**Required receipt:**
1. Code path showing the PQ primitive is actually wired (not only a
   dependency).
2. Scope note: *which* operations are PQ-signed and which are classical.
3. Reference to the NIST round / standard the primitive implements.

**Receipt artefact:** `/trust/post-quantum.md` with scope + code links.

**Risk:** "quantum-safe by default" is a prohibited absolute; rewrite to
scoped language before publishing.

### 5. Ihsan-threshold claim (17)

**Typical claim shape:** "Ihsan ≥ 0.95", "IHSAN_THRESHOLD = 0.95".

**Required receipt:**
1. `core/integration/constants.py` link with line anchor.
2. Rust-side constant in `bizra-omega/bizra-core/` with cross-language sync
   test reference.
3. A dashboard or API endpoint that publishes the live composite score.

**Receipt artefact:** `/trust/ihsan.md` + status-page block.

### 6. Local-only / no-cloud claim (12)

**Typical claim shape:** "no cloud dependency", "runs entirely locally",
"no upstream call".

**Required receipt:**
1. A `netstat`-style test fixture that binds no remote socket during
   normal operation.
2. Opt-in sharing documented with exactly which endpoints are called and
   when.
3. `/privacy` page (also required by W-H-02 in WEBSITE_PATCH_PLAN).

**Receipt artefact:** `/privacy` + a test badge reflecting the network-free
smoke test.

### 7. Relative-performance claim (8)

**Typical claim shape:** "10x faster than X", "faster than cloud".

**Required receipt:**
1. Benchmark methodology document: hardware, software versions, config.
2. Independent re-run instructions.
3. Link to the raw numbers in a versioned artefact, not prose.

**Receipt artefact:** `/trust/benchmarks.md`.

**Risk:** relative-performance claims against a named competitor are a
platform-policy hotspot; require legal review before publishing.

### 8. Zero-telemetry claim (2)

**Typical claim shape:** "no telemetry", "we never phone home".

**Required receipt:** identical to Local-only (§ 6).

**Risk:** this is the most dangerous absolute on the public site today
(W-H-02). Hold until `/privacy` is live.

---

## Close order

Receipts should land in this order so the earliest-unblocked surfaces
unblock the most downstream copy:

1. **Ihsan** (17) — already cited in constants; easy win.
2. **Hashing** (122) — standard primitive, reproducible in a test.
3. **Cryptography** (122) — standard primitive + sample receipt.
4. **Local-only** (12) + **Zero-telemetry** (2) — packaged with the
   `/privacy` publish.
5. **Formal-verification** (62) — requires a CI job to land on trunk first.
6. **Post-quantum** (22) — scope-qualify first, then receipt.
7. **Relative-performance** (8) — last; needs independent benchmark
   methodology.

## Exit criterion

The backlog closes when every PROOF_REQUIRED claim in the live site copy
has either

1. a documented receipt artefact under `/trust/` with commit-hash anchor, or
2. been rewritten into a PROOF_NOT_REQUIRED directional line, or
3. been moved off public surfaces entirely.

The Flywheel Kernel then re-runs against a refreshed audit. When
`PROOF_REQUIRED` on site-sourced claims is zero, `G-FW-003` can drop from
`BLOCK` to `WARN` (or `PASS` if `PROHIBITED = NEEDS_REWRITE = 0` too).
