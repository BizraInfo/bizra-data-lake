# Cycle-6 Execution Spec — First Impact Receipt

بسم الله الرحمن الرحيم

**Cycle:** 6
**Niyyah:** First real impact receipt on Mumo's Downloads folder via MCP tool transport
**Authority:** Dema CLI Manifesto v0 §9
**Predecessor:** Cycle-5 closed (commit 8b16762a, 13 commits on origin)
**Node:** NODE0 (Ubuntu 24.04, bizra-omega workspace)

---

## Success condition (before execution begins)

`dema submit "organize my Downloads folder"` produces:
1. A MissionEnvelope with intent="organize my Downloads folder"
2. Admissibility evaluation (5 gates PERMIT at quality ≥ 0.95)
3. Real filesystem operations (categorize + move files in ~/Downloads)
4. Per-operation sub-receipts (each file move is receipted)
5. A final ReceiptArtifact binding all sub-receipts to the parent mission
6. `dema chain` shows the mission + sub-receipts + final receipt
7. `dema receipt <id>` shows the file operations performed
8. An independent replay can verify the operations occurred

---

## Three gates

### G1: MCP tool transport as sub-mission pattern

**What:** Wire MCP (Model Context Protocol) tool calls into the mission pipeline so that each tool invocation becomes a sub-receipt bound to the parent mission.

**Implementation:**
- Add `ToolInvocation` struct to runtime.rs:
  ```rust
  pub struct ToolInvocation {
      pub tool_name: String,        // e.g. "filesystem.move_file"
      pub input_hash: Blake3Hash,   // BLAKE3 of the input parameters
      pub output_hash: Blake3Hash,  // BLAKE3 of the tool's output
      pub parent_mission: Blake3Hash, // mission_id this belongs to
      pub timestamp_ns: u64,
  }
  ```
- Implement `ReceiptPayload for ToolInvocation` (kind: `ReceiptKind::ToolExecution`)
- Add `ReceiptKind::ToolExecution = 0x80` to receipts.rs
- In submit_mission(), after PERMIT, execute tool calls and append each as a sub-receipt
- The final NodeLifecycle receipt's evidence_chain includes all sub-receipt hashes

**Test:** `test_tool_invocation_produces_sub_receipts` — submit mission with 3 mock tool calls, verify chain has 1 mission + 5 gates + 3 tool receipts + 1 final = 10 records.

### G2: Filesystem operation tool with receipt shape

**What:** A concrete tool that categorizes and moves files in a directory.

**Implementation:**
- Create `bizra-cognition/src/tools/filesystem.rs`:
  ```rust
  pub struct FileMoveOp {
      pub source: PathBuf,
      pub destination: PathBuf,
      pub file_hash: Blake3Hash,  // hash of file content before move
      pub category: String,       // e.g. "document", "image", "archive"
  }
  
  pub fn organize_directory(path: &Path) -> Result<Vec<FileMoveOp>, ToolError> {
      // 1. Scan directory
      // 2. Categorize each file by extension (14-category classifier from file_management.rs)
      // 3. Create category subdirectories
      // 4. Move files into categories
      // 5. Return Vec<FileMoveOp> with before/after hashes
  }
  ```
- Each FileMoveOp becomes a ToolInvocation sub-receipt
- The organize_directory function is the "executor" that submit_mission calls

**Test:** Create a temp directory with 5 test files (.pdf, .jpg, .rs, .txt, .zip), run organize_directory, verify:
- Files moved to correct category dirs
- Each move produces a FileMoveOp with valid source/dest/hash
- Original file hashes match (no corruption during move)

### G3: First real impact receipt on Downloads folder

**What:** Run the full pipeline on Mumo's actual ~/Downloads folder.

**Implementation:**
- `dema submit "organize my Downloads folder" --target ~/Downloads`
- The `--target` flag maps to MissionEnvelope's ideal_state
- Gateway POST /mission with tool_target field
- Runtime calls organize_directory(target_path)
- Each file move → sub-receipt → chain
- Final receipt aggregates all sub-receipts
- `dema chain` shows the full operation
- `dema receipt <final_id>` shows what was moved where

**Test:** This is a manual acceptance test (Mumo's real Downloads folder). Automated test uses a temp directory copy.

**Safety:** Before moving any file, hash it. After moving, verify hash matches. If mismatch → abort + DegradedPathReceipt. This is CLAIM_MUST_BIND for filesystem operations.

---

## Deferred (not Cycle-6 scope)

- LLM-powered file categorization (use extension-based classifier first)
- Cloud backup of organized files
- Cross-device sync
- Undo/rollback from receipts (future — receipts enable this but UI doesn't exist)
- Persistent chain across restart (sled-store — can run parallel but not blocking G1-G3)

---

## Build order

1. Add ReceiptKind::ToolExecution to receipts.rs (2 lines)
2. Add ToolInvocation struct + ReceiptPayload impl to runtime.rs (~50 lines)
3. Create tools/ module with filesystem.rs (~150 lines)
4. Wire organize_directory into submit_mission executor path (~30 lines)
5. Add --target flag to dema CLI submit subcommand (~10 lines)
6. Add tool_target to gateway POST /mission DTO (~5 lines)
7. Tests: tool_invocation_sub_receipts + organize_temp_directory + chain_length_with_tools
8. Manual test on ~/Downloads (G3 acceptance)

**Estimated LOC:** ~250 new lines + ~20 modified
**Estimated tests:** 5-8 new
**Estimated time:** 1 focused session

---

## Commit discipline

One focused commit per gate:
- G1: `feat(cognition): ToolInvocation sub-receipt pattern`
- G2: `feat(cognition): filesystem organize tool with receipt shape`
- G3: `feat(dema): --target flag + first impact receipt`

Push after all three pass. CI fires. Cycle-6 closes.

---

## The moment

When G3 passes, type:

```
dema chain
```

The chain will show:
- 1 MissionEnvelope ("organize my Downloads folder")
- 5 GateVerdicts (all PERMIT)
- N ToolInvocations (one per file moved)
- 1 Final receipt (binding everything)

Each file move is independently verifiable. The chain proves the work happened. Any node can replay and confirm. That's the first time the trust compiler produces trust about something that changed the real world.

**Not a demo. Not a simulation. Proof.**
