# Node0 Genesis Manifest — Redaction Policy v1

Generated: 2026-04-25 GST · Scope: `evidence/node0_genesis_manifest/`

This policy governs what the Node0 Genesis Manifest tool (`tools/node0_genesis_manifest.py`) is allowed to read, hash, and emit. It implements the canon rules from `feedback_secret_triage_redacted_only` (never reproduce raw secret values; only hashes/manifests).

---

## 1. Visibility levels

| Level | What the script does | What the script does NOT do |
|---|---|---|
| `public` | Reads file bytes, computes content hash, records size + path | Never copies/exports bytes outside the file |
| `private` | Reads file bytes, computes content hash, records size only | Does NOT log the path beyond `current_location` hint; does NOT include any file content in any output |
| `hash_only` | Alias for `private` | Same |
| `redacted` | Skips the file entirely; records the asset entry with `content_hash: null`, `redaction_reason` populated | Does NOT open the file. Path may still appear in `current_location` if non-sensitive |

## 2. Hard rules (script-enforced)

1. **No raw private content in any output file.** The hash ledger and run report contain hashes and sizes only — no excerpts, no previews, no headlines, no titles beyond the operator-supplied `title` field.
2. **Titles are operator-controlled.** If an asset's title would itself leak (e.g. "letter to my doctor on 2024-03-12"), the operator MUST sanitize it before adding to the manifest. The script does not validate title content.
3. **Paths to private archives are not echoed.** The asset's `path_relative` is set to `null` when `visibility != public`. The `current_location` field is operator-supplied and may say e.g. "private archive — Mumu local SSD" without disclosing absolute paths.
4. **No external transmission.** The script writes only to `evidence/node0_genesis_manifest/` and does not perform any network call.
5. **No secret values in error messages.** Hash computation errors emit only `{ "asset_id", "error_class", "error_message_redacted" }`. Stack traces are not propagated to the run report.

## 3. Soft rules (operator-controlled)

6. **Family / health / financial / faith-private content stays `redacted` indefinitely** unless the operator explicitly upgrades the visibility. There is no automated upgrade path.
7. **`FOUNDER_STATED` claims** (e.g. "15,000 hours of work since Ramadan 2023") are recorded as PLANNED entries until a separate timestamped artifact (commit log, calendar export, etc.) backs them. The Genesis Manifest does not anchor these by itself.
8. **Cross-model chat-history corpora** (Claude/ChatGPT/Gemini exports) are `private` by default. Each export is hashed once at archive level. Per-conversation hashing is a separate future tool, not part of this v1 scope.

## 4. What this manifest is NOT

- It is not a Claim Registry. Claim/evidence linking with `directional / measured / independently_attested` lifecycle states is **future Phase 3 work** and is explicitly out of scope per the canon patch dated 2026-04-25.
- It is not an attestation service. No third-party signer is invoked.
- It is not a Proof-of-Impact engine. No PoI scoring is computed.
- It does not produce Ed25519 signatures. The Genesis Manifest is content-addressed by hash, not signed. (Signing belongs to the runtime spine at `core/proof_engine/canonical_receipt_adapter.py` and is out of this side-track's scope.)

## 5. P1 risk recorded (for Phase 3 closure, not patched here)

**Adversarial evidence poisoning**: structurally plausible evidence with manipulated empirical/provenance content may pass hash convergence and corrupt downstream memory lessons. Closure: future Phase 3 Claim Registry must require source lineage, provenance validation, and independent attestation before any claim is upgraded from `directional` to `measured` or `independently_attested`. **The Genesis Manifest does not protect against this risk and does not claim to.**

## 6. Future IhsanDecision boundary (recorded, not implemented)

The future Ihsān policy gate will reject any `private → public` visibility upgrade unless an `IhsanDecision` is recorded:

```yaml
IhsanDecision:
  gate: "allow" | "allow_with_notice" | "require_approval" | "deny"
  trigger_reason: <deterministic string explaining the rule/input>
  trace_hash: hash(action_id + policy_version + decision + timestamp)
```

This schema is a reference boundary only. The script does not implement it. Visibility changes today are operator-controlled via direct manifest edit.

---

## Operator checklist before adding a new asset

- [ ] Title is non-leaking
- [ ] `current_location` hint is non-leaking
- [ ] `visibility` correctly reflects sensitivity
- [ ] `proof_status` is the lowest-confidence label that fits (default: `FOUNDER_STATED` or `PLANNED`)
- [ ] If `redacted`, `redaction_reason` is filled in
- [ ] Asset is something whose existence-and-hash is more valuable to publish than its content
