# Canon-pack promotion report

- **Tool:** `cognitive_foundry.claude_lane.promote` v0.1.0
- **Workbook:** `/data/bizra/repos/bizra-data-lake/tools/cognitive_foundry/claude_lane/output/20260424T000948Z_78a12953b97a1085/04_review_pack/review_workbook.csv`
- **Origin run_id:** `20260424T000948Z_78a12953b97a1085`
- **Promoted at (UTC):** `2026-04-24T02:14:05Z`
- **Promoter:** `bizra-operating-system`
- **Dry run:** `False`
- **Entries hash (blake2b-32):** `07bd7dfd7a7627efebbfec3bb40ea820d2ddc41593616efaacf2e4536a3030b4`
- **Pack directory:** `/data/bizra/repos/bizra-data-lake/tools/cognitive_foundry/claude_lane/canon_packs/20260424T000948Z_78a12953b97a1085_promoted_20260424T021405Z_07bd7dfd7a76`

## Partition counts

| Category | Count |
|---|---|
| **Approved AND promote_to_canon=yes (PROMOTED)** | **12** |
| Approved but promote_to_canon not yes | 2 |
| Rejected | 25 |
| Needs follow-up | 32 |
| Pending review (unreviewed) | 288 |
| Other (retired / merged / ...) | 0 |

## Validation errors

None.

## Canon discipline notes

- This pack is NOT yet in any canonical BIZRA store.
- A future, separately-implemented tool must ingest this pack into actual canon (e.g., MEMORY.md entries or a runtime canonical index) — and that tool will require its own confirmation gate.
- The `entries_hash` is content-addressable tamper evidence, not a cryptographic signature. A future promotion tool can upgrade to Ed25519 if required.
- The pipeline NEVER auto-sets `promote_to_canon=yes`. Every row in the PROMOTED count above was explicitly marked by a human reviewer.
