# Canon-pack promotion report

- **Tool:** `cognitive_foundry.claude_lane.promote` v0.2.0
- **Workbook:** `/data/bizra/repos/bizra-data-lake/tools/cognitive_foundry/claude_lane/output/20260424T000948Z_78a12953b97a1085/04_review_pack/review_workbook.csv`
- **Origin run_id:** `20260424T000948Z_78a12953b97a1085`
- **Promoted at (UTC):** `2026-04-24T05:32:25Z`
- **Promoter:** `bizra-operating-system`
- **Dry run:** `False`
- **Entries hash (blake2b-32):** `f76d8c1fde61f17e19d50a5a26004155fd6febf6ddd2146903f92eb9fbe8c770`
- **Pack directory:** `/data/bizra/repos/bizra-data-lake/tools/cognitive_foundry/claude_lane/canon_packs/20260424T000948Z_78a12953b97a1085_promoted_20260424T053225Z_cface302f993`

## Partition counts

| Category | Count |
|---|---|
| **Approved AND promote_to_canon=yes (PROMOTED)** | **27** |
| Approved but promote_to_canon not yes | 21 |
| Rejected | 174 |
| Needs follow-up | 137 |
| Pending review (unreviewed) | 0 |
| Other (retired / merged / ...) | 0 |

## Validation errors

None.

## Canon discipline notes

- This pack is NOT yet in any canonical BIZRA store.
- A future, separately-implemented tool must ingest this pack into actual canon (e.g., MEMORY.md entries or a runtime canonical index) — and that tool will require its own confirmation gate.
- The `entries_hash` is content-addressable tamper evidence, not a cryptographic signature. A future promotion tool can upgrade to Ed25519 if required.
- The pipeline NEVER auto-sets `promote_to_canon=yes`. Every row in the PROMOTED count above was explicitly marked by a human reviewer.
