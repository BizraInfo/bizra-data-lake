# Canon Store Ingestion Gate Design

**Status:** Phase-3 design note, not implementation
**Date:** 2026-04-26 GST
**Scope:** human-gated path from candidate canon artifacts to runtime canon stores
**Non-authorization:** this document does not ingest any canon, mutate runtime state, change `MEMORY.md`, or promote the Origin Kernel into runtime canon.

---

## 1. Why This Gate Exists

BIZRA's strongest truth boundary is the refusal to let generated or reviewed material become canon by proximity. A file may be meaningful, reviewed, content-addressed, or spiritually foundational, but it remains outside runtime canon until an explicit human-gated ingestion process admits it.

The gate protects three invariants from `docs/canon/BIZRA_ORIGIN_KERNEL.md`:

| Origin Kernel invariant | Gate implication |
|---|---|
| §4.1 Knowledge -> humility | Candidate packs must declare what they do not prove. |
| §4.2 Symmetric epistemic charity | Reviewer dissent and alternate interpretations must be preserved. |
| §4.3 Law of Assumption with Ihsan | Ingestion must reject unsupported certainty and force declared uncertainty. |

This is the missing bridge between:

`candidate artifact -> human review -> content hash -> operator approval -> signed ingestion receipt -> runtime canon update`

---

## 2. What The Gate Is Not

The Canon Store Ingestion Gate is not:

- An automatic importer.
- A markdown copy/paste workflow.
- A memory write shortcut.
- A runtime startup hook.
- A way to bypass public claim discipline.
- A tool that edits raw origin utterances.
- A substitute for tests, receipts, or human approval.

If an artifact is not explicitly ingested through this gate, it remains **candidate-for-canon** or **reference material**, not runtime canon.

---

## 3. Inputs

The gate accepts only typed ingestion requests with the following fields:

| Field | Required | Purpose |
|---|---:|---|
| `candidate_path` | yes | File or pack directory to evaluate. |
| `candidate_type` | yes | `origin_kernel`, `foundry_pack`, `doctrine_doc`, `constant_set`, `topology_rule`, or `other`. |
| `content_hash` | yes | Stable content identity of the candidate. |
| `issuance_hash` | optional | Promotion-event identity when present, e.g. Cognitive Foundry v0.2.0 packs. |
| `target_store` | yes | Destination canon store or runtime surface requested. |
| `operator_id` | yes | Human approver identity. |
| `operator_statement` | yes | Plain-language reason for ingestion. |
| `truth_label` | yes | `MEASURED`, `PARTIAL`, `PLANNED`, or `DIRECTIONAL`. |
| `rollback_plan` | yes | How to revert if ingestion is later rejected. |

Requests with missing fields fail closed.

---

## 4. Mandatory Validation

Before any write, the gate must validate:

1. **Path boundary:** candidate path is inside the approved repository or evidence root.
2. **Content hash:** current bytes match the declared content hash.
3. **Disposition:** candidate is not superseded by a later preferred pack.
4. **Claim discipline:** exact public or runtime claims are either receipted or truth-labeled.
5. **Origin preservation:** raw source sections marked immutable are not changed.
6. **Review status:** human review status is complete, or the request declares an explicit partial truth label.
7. **Runtime target:** destination store is known, versioned, and rollback-capable.
8. **Operator confirmation:** typed approval matches the candidate hash and target store.

Any failed validation returns a rejection receipt, not a partial write.

---

## 5. Human Approval Ceremony

The gate requires a deliberate human confirmation step:

```text
I, <operator_id>, approve ingesting <content_hash> from <candidate_path>
into <target_store> with truth label <truth_label>.
I understand this will create a signed ingestion receipt and can be rolled back only
through the declared rollback procedure.
```

The confirmation string must be stored verbatim in the ingestion receipt.

---

## 6. Outputs

Successful ingestion emits:

| Output | Purpose |
|---|---|
| `ingestion_receipt.json` | Signed record of exactly what was ingested, by whom, and why. |
| `pre_state_hash` | Hash of target store before mutation. |
| `post_state_hash` | Hash of target store after mutation. |
| `content_hash` | Stable identity of ingested content. |
| `issuance_hash` | Optional promotion-event identity. |
| `rollback_manifest.json` | Revert target store to `pre_state_hash`. |
| `audit_log.jsonl` | Append-only gate event stream. |

Rejected ingestion emits:

| Output | Purpose |
|---|---|
| `rejection_receipt.json` | Signed rejection reason. |
| `failed_validation` | Machine-readable failed gate. |
| `candidate_hash` | What was evaluated. |

---

## 7. Target Store Policy

Initial allowed targets should be narrow:

| Target | Initial status | Notes |
|---|---|---|
| Documentation canon index | allowed after review | Lowest risk; index-only mutation. |
| Claim registry | allowed after review | Must preserve truth labels and evidence links. |
| MEMORY.md / memory anchors | paused | Needs separate memory governance policy. |
| Rust topology canon | paused | Requires tests and code review. |
| Python constants | paused | Requires cross-language sync tests. |
| Runtime canon database | paused | Requires backup, rollback, and signed receipts. |

Default posture: documentation targets first; runtime targets later.

---

## 8. Security And DevOps Requirements

- No secrets may appear in candidate artifacts or receipts.
- Receipts must be append-only and content-addressed.
- Runtime target writes must be atomic.
- Rollback must be tested before the first production ingestion.
- CI should include a dry-run validator before merge.
- Production ingestion should require a signed release or protected branch context.
- The gate must run without network access unless the target store explicitly requires it.

---

## 9. Minimal Implementation Runway

The first implementation should be a dry-run validator, not an ingestion writer. It should prove that BIZRA can reject unsafe canon promotion before it proves that it can write canon.

### Phase K4.1 — Dry-run validator

| Capability | Requirement |
|---|---|
| Input | JSON request file matching the schema in §3. |
| Mutation | None. No target store writes. |
| Output | `dry_run_receipt.json` or `rejection_receipt.json`. |
| Network | Disabled by default. |
| Exit code | `0` for accepted dry run, non-zero for rejection or internal error. |

Recommended command shape:

```bash
python tools/canon_store/ingestion_gate.py validate \
  --request path/to/ingestion_request.json \
  --output artifacts/canon_store/dry_runs/
```

### Phase K4.2 — Documentation-index writer

Only after K4.1 has accepted one valid candidate and rejected one tampered candidate should the gate write to the lowest-risk target: a documentation-only canon candidate index.

Allowed mutation:

```text
docs/canon/CANDIDATE_CANON_INDEX.json
```

Paused mutations:

- `docs/canon/BIZRA_ORIGIN_KERNEL.md`
- `MEMORY.md`
- Rust topology canon
- Python constants
- Any runtime canon database

### Phase K4.3 — ADR promotion

The design can become an ADR only after receipts, rollback manifests, and tamper rejection are proven in CI-safe mode. Runtime canon targets remain out of scope until a separate typed GO names the exact store and rollback procedure.

---

## 10. Request Schema Draft

Initial request files should be explicit and boring:

```json
{
  "candidate_path": "tools/cognitive_foundry/claude_lane/canon_packs/example/",
  "candidate_type": "foundry_pack",
  "content_hash": "sha256:<hex>",
  "issuance_hash": "sha256:<hex-or-null>",
  "target_store": "docs/canon/CANDIDATE_CANON_INDEX.json",
  "operator_id": "human-operator-id",
  "operator_statement": "I approve dry-run validation for this candidate only.",
  "truth_label": "PARTIAL",
  "rollback_plan": "No mutation in dry-run; remove index entry for documentation-index write."
}
```

Schema rules:

- `candidate_path` must resolve inside an approved repository or evidence root.
- `content_hash` and `issuance_hash` must include the algorithm prefix.
- `truth_label` must be one of `MEASURED`, `PARTIAL`, `PLANNED`, or `DIRECTIONAL`.
- `target_store` must be in the allowlist for the current phase.
- `operator_statement` must be stored verbatim in the receipt.

---

## 11. Receipt Schema Draft

Dry-run and rejection receipts should be machine-checkable and human-readable.

```json
{
  "receipt_type": "canon_ingestion_dry_run",
  "schema_version": "0.1",
  "created_at": "2026-04-26T00:00:00Z",
  "candidate_path": "tools/cognitive_foundry/claude_lane/canon_packs/example/",
  "candidate_hash": "sha256:<hex>",
  "target_store": "docs/canon/CANDIDATE_CANON_INDEX.json",
  "truth_label": "PARTIAL",
  "operator_id": "human-operator-id",
  "operator_statement": "verbatim operator approval text",
  "validations": [
    {"name": "path_boundary", "status": "passed"},
    {"name": "content_hash", "status": "passed"},
    {"name": "claim_discipline", "status": "passed"}
  ],
  "decision": "accepted_dry_run",
  "mutation_performed": false,
  "pre_state_hash": null,
  "post_state_hash": null,
  "receipt_hash": "sha256:<hex>",
  "signature": null
}
```

For K4.1, `signature` may remain `null` if no approved signing key path exists. Before K4.2 writes any index, signatures become required and must use managed key material, never hard-coded secrets.

---

## 12. Fail-Closed Exit Codes

The CLI should reserve clear exit codes so CI can act without parsing prose:

| Code | Meaning |
|---:|---|
| `0` | Validation accepted in current mode. |
| `2` | Request schema invalid. |
| `3` | Candidate path outside approved boundary. |
| `4` | Content hash mismatch. |
| `5` | Claim discipline failure. |
| `6` | Target store not allowed in current phase. |
| `7` | Operator confirmation mismatch. |
| `8` | Secret scan failure. |
| `9` | Internal error; no mutation performed. |

Every non-zero exit must emit a rejection receipt.

---

## 13. First Dry-Run Candidate

The first safe dry run should be index-only:

| Field | Value |
|---|---|
| Candidate | `tools/cognitive_foundry/claude_lane/canon_packs/20260424T000948Z_78a12953b97a1085_promoted_20260424T053225Z_cface302f993/` |
| Why | Preferred pack is fully reviewed and already documented as candidate-for-canon. |
| Target | Documentation-only canon candidate index. |
| Runtime mutation | none |
| Success output | signed dry-run receipt plus no target mutation |

The Origin Kernel should not be the first ingestion candidate. It is source authority and has immutable raw text. It should remain read/cite-only until the gate has proven rollback, rejection receipts, and immutable-section protection.

---

## 14. Acceptance Criteria

The design graduates from note to ADR when:

- Input schema is finalized.
- Rejection receipt schema is finalized.
- Ingestion receipt schema is finalized.
- Dry-run mode is implemented with zero target mutation.
- At least one preferred Foundry pack passes dry-run validation.
- One tampered candidate fails validation with a signed rejection receipt.
- Operator approval ceremony is tested end-to-end.
- Runtime targets remain paused until separate typed authorization.

---

## 15. Stop Line

This note authorizes **design only**. It does not authorize:

- Editing `docs/canon/BIZRA_ORIGIN_KERNEL.md`.
- Ingesting the Origin Kernel into runtime canon.
- Writing to `MEMORY.md`.
- Writing to Rust/Python runtime canon stores.
- Opening or pushing a PR.
- Mutating public website claims.

When in doubt, apply Origin Kernel §4.3: refuse assumption; if assumption is unavoidable, declare uncertainty and act with Ihsan.
