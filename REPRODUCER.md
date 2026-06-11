# BIZRA Node0 — External Witness Reproducer

**Audience:** anyone outside the BIZRA project. No prior context required.
**Time:** ~15–30 minutes. **Founder involvement:** zero, by design.

---

## 1. What you are being asked to witness

BIZRA's engineering thesis is _proof before claim_. Three of its four proof axes
(Formal, Cryptographic, Empirical) are verified internally. The fourth —
**Economic** — requires something the founder cannot produce alone: an
independent human spending their own effort to reproduce a result and attest
to it.

You are that fourth axis. The claim under witness is deliberately small and
checkable:

> **Claim (Arc 3):** When `BIZRA_RECEIPT_STORE_PATH` is set, the Node0
> gateway's receipt chain (`GET /chain` length and head) survives a full
> process restart. Default in-memory behavior is unchanged.

You are _not_ asked to endorse BIZRA, its economics, or its philosophy.
You are asked to run one script and report what actually happened —
**including failure**. A reproduction failure is a valid and valued result.

## 2. Prerequisites

- Linux or macOS, ~4 GB free disk
- `git`, `rustc`/`cargo` (stable), `python3` ≥ 3.10
- Optional: `b3sum` (BLAKE3) — falls back to `sha256sum`

## 3. Run

```bash
export BIZRA_WITNESS_COMMIT=PUBLISHED_SHA_HERE   # from the table below
curl -fsSLO https://raw.githubusercontent.com/BizraInfo/bizra-data-lake/PUBLISHED_SHA_HERE/tools/witness/reproduce.sh
chmod +x reproduce.sh && ./reproduce.sh
```

| Field                 | Value                                                 |
| --------------------- | ----------------------------------------------------- |
| Witness target commit | `PUBLISHED_SHA_HERE` — maintainer fills after Gate W0 |
| Expected label        | `NODE0_MISSION_REPLAY_PERSIST_WITNESS_COMPLETE`       |
| Expected field        | `persist_survives_restart: true`                      |

The script is fail-closed: it refuses floating HEADs, dirty trees, and
unverifiable witness output. Everything it writes stays under
`./bizra-witness-run/`. It makes no network calls after the clone and asks
for no credentials.

## 4. Report (any of the three)

1. **Success:** fill `witness_identity` in the generated
   `artifacts/witness/<timestamp>/ATTESTATION.json` and send it back
   (GitHub issue titled `WITNESS: <commit-short>`, or email). The witness
   hash inside it is the attestation — it binds your run to the exact commit.
2. **Failure:** open an issue titled `WITNESS-FAIL: <commit-short>` with
   `witness-stdout.log` attached. This is treated as a first-class finding,
   not an embarrassment.
3. **Refusal with reasons:** if you inspected and declined to run, say why.
   That is also signal.

## 5. What your attestation becomes

Your `ATTESTATION.json` hash is recorded in the project's receipt chain as
the first (or Nth) external witness receipt. It is the project's equivalent
of a second node validating a genesis block: small, boring, and decisive.

## 6. Trust posture / what to inspect first

- `tools/witness/run_witness.py` — the harness itself (~read in 5 min):
  starts the gateway with a persistence path, writes receipts, kills the
  process, restarts, and asserts chain length/head survival.
- `tools/witness/reproduce.sh` — this wrapper. Verify it does what §3 says.
- No step requires sudo. If any step asks for elevation, stop and report.

---

_Maintainer checklist (Gate W0 — must be green before publishing this file):_

- [ ] `run_witness.py` vendored into `tools/witness/`, path-parameterized, no absolute `/data/bizra/...` paths
- [ ] Witness JSON embeds the built commit SHA
- [ ] `reproduce.sh` tested end-to-end on a clean machine by the maintainer
- [ ] `PUBLISHED_SHA_HERE` filled in §3 with a CI-green commit
- [ ] Default-branch security advisories: 0 critical (✅ already true as of 11 Jun)
