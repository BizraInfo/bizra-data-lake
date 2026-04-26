# Canon-pack promotion report

- **Tool:** `cognitive_foundry.claude_lane.promote` v0.2.0
- **Workbook:** `/data/bizra/repos/bizra-data-lake/tools/cognitive_foundry/claude_lane/output/20260424T000948Z_78a12953b97a1085/04_review_pack/review_workbook.csv`
- **Origin run_id:** `20260424T000948Z_78a12953b97a1085`
- **Promoted at (UTC):** `2026-04-24T03:19:02Z`
- **Promoter:** `bizra-operating-system`
- **Dry run:** `False`
- **Entries hash (blake2b-32):** `9df8f20bc251955ff54ec17d7bf2fbac37d33868586ddcbcd935d37f608c0357`
- **Pack directory:** `/data/bizra/repos/bizra-data-lake/tools/cognitive_foundry/claude_lane/canon_packs/20260424T000948Z_78a12953b97a1085_promoted_20260424T031902Z_67d1f4c85271`

## Partition counts

| Category | Count |
|---|---|
| **Approved AND promote_to_canon=yes (PROMOTED)** | **25** |
| Approved but promote_to_canon not yes | 19 |
| Rejected | 80 |
| Needs follow-up | 78 |
| Pending review (unreviewed) | 157 |
| Other (retired / merged / ...) | 0 |

## Validation errors

None.

## Canon discipline notes

- This pack is NOT yet in any canonical BIZRA store.
- A future, separately-implemented tool must ingest this pack into actual canon (e.g., MEMORY.md entries or a runtime canonical index) — and that tool will require its own confirmation gate.
- The `entries_hash` is content-addressable tamper evidence, not a cryptographic signature. A future promotion tool can upgrade to Ed25519 if required.
- The pipeline NEVER auto-sets `promote_to_canon=yes`. Every row in the PROMOTED count above was explicitly marked by a human reviewer.
