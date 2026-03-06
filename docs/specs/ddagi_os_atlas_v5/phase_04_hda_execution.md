# Phase 04 — HDA Execution: Brain/Body Split, AHK Bridge, Closed-Loop Verification

> Source: Atlas v5.0 — Diagram D2 (HDA Architecture)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-040: Brain/Body Split-Process Design

The Human-Desktop Automation layer separates concerns into two OS-level
processes. The Body Process handles low-level hardware interaction. The Brain
Process handles reasoning, scoring, and the actuation decision. Communication
between them flows over a JSON-RPC channel (TCP:9742) with Ed25519-signed
message envelopes.

**Body Process (Kinetic Layer)**

| Subsystem              | Responsibility                                         |
|------------------------|--------------------------------------------------------|
| UIA Tree Scanner       | Enumerate live UI Automation tree; extract element IDs, names, bounding rects |
| OCR + Vision           | Screenshot capture, text extraction, visual element location when UIA fails |
| AHK Nervous System     | Execute real keystrokes, mouse clicks, window management via AHK bridge (8 HDA verbs) |
| Postcondition Verifier | Before/after screenshot hash diff; structural UI tree diff; pixel-region comparison |

The Body Process has no reasoning capability. It receives atomic action
instructions and returns observation payloads. It never initiates action
autonomously.

**Brain Process (Cognitive Layer)**

| Subsystem         | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| Entropy Router    | Shannon H(task) with theta=4.5 bits raw (0.68 normalized). Below: S1. Above: S2. (Phase 02, FR-020) |
| System-2 Pipeline | GoT + PBFT quorum voting + Diffusion amplification + Aha moment detection |
| System-1 Pipeline | AscentTracker + O(1) ReflexCache hash match + Direct execution (no GoT/PBFT) |
| SNR Scoring       | Three-level: Atomic (per-step), Edge (step-transition), Path (full plan). Floor: 0.85 |
| Actuation Decision | Weighted logistic gate producing scalar probability of action |

**Actuation Decision Function:**

```
P_A = sigma(4 * PSI + 3 * SNR + 2 * G - 4 + lambda * nabla_SNR)
```

- `PSI` = PSI-AST safety score (binary: 0 or 1)
- `SNR` = composite signal-to-noise [0, 1]
- `G` = FATE gate alignment score [0, 1]
- `lambda` = momentum coefficient (default 0.5)
- `nabla_SNR` = SNR gradient over last 3 cycles
- `sigma(x) = 1 / (1 + exp(-x))`

**Threshold:** `P_A > 0.85`. Below this the mission pauses for human review.

Canonical safe point (PSI=1, SNR=0.95, G=0.95, nabla=0):
`sigma(4 + 2.85 + 1.90 - 4) = sigma(4.75) = 0.991`. If PSI=0, sum drops by
4, making passage nearly impossible. The kill switch is decisive by design.

### FR-041: Security Enclave

All HDA actions pass through a security enclave before reaching the Body
Process. Four constraints:

**PSI-AST Binary Kill Switch.** Every AHK script is parsed into an AST and
walked against a whitelist. Any unknown construct, shell escape, registry
write, or network call triggers `PSI = 0`. No partial score.

Whitelisted: `SendInput`, `Click`, `WinActivate`, `WinWait`, `WinClose`,
`ControlClick`, `ControlSend`, `ControlGetText`, `Sleep`, `MsgBox`, `Clipboard` (read).
Blacklisted: `Run`, `RunWait`, `FileAppend` (system paths), `RegWrite`,
`RegDelete`, `Download`, `URLDownloadToFile`, `ComObjCreate`, `DllCall`
(unless approved DLL whitelist).

**Path Validation (Safe Zones Only).** File operations restricted to:
`${BIZRA_DATA_LAKE_ROOT}/`, `${HOME}/Documents/`, `${HOME}/Downloads/`,
`${TEMP}/bizra_hda/`. Path traversal (`../`, symlink escape) is blocked.

**Privilege Escalation Guard.** Body Process NEVER requests UAC elevation,
writes to `HKLM`, modifies services, or accesses other users' profiles.
Violation triggers immediate Body Process termination + Sentinel alert.

**Cryptographic Audit Trail (Ed25519).** Every instruction and observation
carries an Ed25519 signature over `BLAKE3(payload || timestamp || nonce)`.
Receipts are Merkle-chained and written to the evidence ledger.

### FR-042: Closed-Loop Verification

Every HDA action follows a four-phase protocol:

```
Phase 1: CAPTURE_BEFORE  -> uia_snapshot + screenshot -> BLAKE3 hash
Phase 2: EXECUTE          -> ahk_execute(action) + settle (200ms default)
Phase 3: CAPTURE_AFTER   -> uia_snapshot + screenshot -> BLAKE3 hash
Phase 4: VERIFY           -> tree_diff + pixel_diff + postcondition check
```

**Postcondition types:** `ELEMENT_EXISTS(name, role)`, `ELEMENT_GONE(name, role)`,
`TEXT_CONTAINS(region, substring)`, `WINDOW_ACTIVE(title_pattern)`,
`PIXEL_CHANGED(region, min_pct)`, `HASH_DIFFERENT`, `HASH_SAME`.

**On failure:** PAUSE (halt pipeline) -> REASSESS (re-evaluate) -> ADAPT
(alternative path, max 2 retries) -> ABORT (partial-result receipt, no silent
continuation).

### FR-043: AHK Error Recovery

Four error classes, each following the PAUSE/REASSESS/ADAPT/ABORT pattern:

| Error Class | REASSESS | ADAPT | ABORT Trigger |
|-------------|----------|-------|---------------|
| Window Disappeared | UIA scan by class+PID | `WinActivate` if minimized; `WinWait(5s)` if hidden | Process exited |
| UI Element Moved | Re-scan UIA tree x3 (500ms apart) | Recalculate coords from new bounding rect | Not found after 3 scans |
| App Crashed | Check for crash dialog | Relaunch if in restart-safe list + restore checkpoint | Not in safe list or restart fails |
| State Mismatch | Fresh capture + postcondition eval | Corrective micro-actions if coverage >= 50% | Coverage < 50% after corrections |

All error recovery emits a typed receipt (`error_class` field) appended to
the evidence ledger.

---

## 2. Edge Cases

**EC-040: Body Process Crash During Execution.**
Brain detects broken TCP within heartbeat interval (1s). Emergency receipt
with `status=BODY_PROCESS_LOST`. Body restarts with degraded permit (50%
budget). Three crashes in 5 minutes suspends HDA until human intervention.

**EC-041: Screen Resolution or DPI Change Mid-Session.**
UIA coordinates are logical (DPI-independent). Verifier detects metric change,
invalidates cached bounding rects, forces full UIA rescan, re-baselines
screenshot comparison.

**EC-042: Multiple Monitors — Element on Wrong Display.**
UIA returns absolute coords. Brain validates target coordinate falls within a
known monitor rect. Out-of-bounds coordinates rejected with
`error_class=COORDINATE_OUT_OF_BOUNDS`.

**EC-043: Concurrent HDA Sessions.**
One Body Process per node enforced via file lock (`${BIZRA_STATE}/hda.lock`
with PID). Second session waits or breaks stale lock.

**EC-044: AHK Bridge Version Mismatch.**
Brain requests `GET_VERSION` on startup. Semantic version mismatch blocks all
actuation commands. Emits `VERSION_MISMATCH` receipt.

---

## 3. Pseudocode

### 3.1 hda_execute(mission)

```
FUNCTION hda_execute(mission, brain, body, permit):
    required_caps = extract_capabilities(mission.action_plan)
    FOR cap IN required_caps:
        IF NOT permit.has_capability(cap): RETURN HDAResult(REJECTED, "missing_capability:" + cap.name)
        IF permit.budget_exhausted(cap):   RETURN HDAResult(REJECTED, "budget_exhausted:" + cap.name)
        IF permit.expired():               RETURN HDAResult(REJECTED, "permit_expired")

    receipts = []

    # Brain: compute actuation decision
    snr = brain.snr_scorer.score_plan(mission.action_plan)
    composite_snr = snr.atomic * 0.40 + snr.edge * 0.30 + snr.path * 0.30
    fate_score = brain.fate_gate.evaluate(mission.action_plan)
    psi = psi_ast_validate(mission.action_plan.ahk_script)
    p_a = sigmoid(4*psi + 3*composite_snr + 2*fate_score - 4 + 0.5*brain.snr_gradient(3))

    IF p_a <= 0.85:
        RETURN HDAResult(BLOCKED, p_a=p_a, snr=composite_snr, fate=fate_score)

    FOR step IN mission.action_plan.steps:
        before = body.capture_state()
        exec_result = body.ahk_execute(step)

        IF exec_result.error:
            recovery = ahk_error_recovery(exec_result.error_type, step, body, brain)
            IF recovery.action == ABORT:
                receipts.append(make_receipt(step, before, None, recovery))
                RETURN HDAResult(ABORTED, receipts=receipts, last_error=recovery)
            exec_result = body.ahk_execute(recovery.adapted_step)

        sleep(step.settle_ms OR 200)
        after = body.capture_state()
        verification = closed_loop_verify(before, after, step.postconditions)
        permit.record_action(step.capability)

        receipt = make_receipt(step, before, after, verification)
        receipt.signature = brain.sign_ed25519(receipt.canonical_bytes())
        receipts.append(receipt)

        IF NOT verification.passed:
            IF verification.retry_eligible AND step.retries_remaining > 0:
                step.retries_remaining -= 1; CONTINUE
            RETURN HDAResult(VERIFICATION_FAILED, receipts=receipts, failed_step=step.id)

    poi = brain.compute_poi(mission, receipts)
    final = FinalReceipt(mission.id, receipts, poi, composite_snr, p_a)
    final.signature = brain.sign_ed25519(final.canonical_bytes())
    brain.evidence_ledger.append(final)
    RETURN HDAResult(SUCCESS, receipts=receipts, poi=poi, p_a=p_a)
```

### 3.2 psi_ast_validate(script)

```
FUNCTION psi_ast_validate(ahk_script):
    ast = ahk_parser.parse(ahk_script)
    IF ast IS None OR ast.has_syntax_errors(): RETURN 0

    WHITELIST = {"SendInput", "Click", "ControlClick", "ControlSend", "ControlGetText",
                 "WinActivate", "WinWait", "WinClose", "WinMinimize", "WinMaximize",
                 "WinGetTitle", "WinExist", "Sleep", "MsgBox", "ClipboardGet",
                 "MouseMove", "ImageSearch", "PixelGetColor"}

    BLACKLIST = {"Run", "RunWait", "FileAppend", "FileDelete", "FileCopy", "FileMove",
                 "RegWrite", "RegDelete", "Download", "URLDownloadToFile",
                 "ComObjCreate", "ComObjGet", "DllCall"}

    FOR node IN ast.walk():
        IF node.command IN BLACKLIST:
            IF node.command == "DllCall" AND node.target_dll IN APPROVED_DLLS: CONTINUE
            log_security_event("PSI_KILL", node.command, node.line); RETURN 0
        IF node.command NOT IN WHITELIST:
            log_security_event("PSI_KILL_UNKNOWN", node.command, node.line); RETURN 0
        IF node.has_file_path():
            IF NOT is_within_safe_zones(resolve_path(node.file_path), SAFE_ZONES):
                log_security_event("PSI_KILL_PATH", node.file_path, node.line); RETURN 0
        FOR arg IN node.string_arguments():
            IF contains_shell_escape(arg):
                log_security_event("PSI_KILL_SHELL", arg, node.line); RETURN 0

    RETURN 1  # All nodes validated
```

### 3.3 closed_loop_verify(before, after, postconditions)

```
FUNCTION closed_loop_verify(before_state, after_state, postconditions):
    results = []
    FOR pc IN postconditions:
        MATCH pc.type:
            ELEMENT_EXISTS:  results.append(PostcondResult(pc, after.uia.find(pc.name, pc.role) != None))
            ELEMENT_GONE:    results.append(PostcondResult(pc, after.uia.find(pc.name, pc.role) == None))
            TEXT_CONTAINS:   results.append(PostcondResult(pc, pc.substring IN after.ocr(pc.region)))
            WINDOW_ACTIVE:   results.append(PostcondResult(pc, matches(after.fg_title(), pc.pattern)))
            PIXEL_CHANGED:   results.append(PostcondResult(pc, pixel_diff_pct(before, after, pc.region) >= pc.min_pct))
            HASH_DIFFERENT:  results.append(PostcondResult(pc, before.hash != after.hash))
            HASH_SAME:       results.append(PostcondResult(pc, before.hash == after.hash))

    IF len(postconditions) == 0:
        RETURN VerificationResult(passed=(before.hash != after.hash), note="implicit_hash_check")

    passed = SUM(1 FOR r IN results IF r.passed)
    coverage = passed / len(results)
    retry_eligible = (passed < len(results)) AND (coverage >= 0.50)
    RETURN VerificationResult(passed=(passed == len(results)), coverage=coverage,
                               retry_eligible=retry_eligible, results=results,
                               before_hash=before.hash, after_hash=after.hash)
```

### 3.4 ahk_error_recovery(error_type)

```
FUNCTION ahk_error_recovery(error_type, step, body, brain):
    MATCH error_type:
        CASE WINDOW_DISAPPEARED:
            scan = body.uia_find_window(class_name=step.target_class, pid=step.target_pid)
            IF scan.found AND scan.state == "minimized":
                body.ahk_execute(AHKCommand("WinActivate", scan.hwnd)); sleep(500)
                IF body.uia_verify_window(scan.hwnd): RETURN RecoveryAction(ADAPT, step)
            IF scan.found AND scan.state == "hidden":
                body.ahk_execute(AHKCommand("WinWait", step.target_title, timeout=5000))
                IF body.uia_verify_window(scan.hwnd): RETURN RecoveryAction(ADAPT, step)
            RETURN RecoveryAction(ABORT, "WINDOW_LOST")

        CASE UI_ELEMENT_MOVED:
            FOR attempt IN 1..3:
                sleep(500)
                rescan = body.uia_find_element(id=step.automation_id, name=step.target_name)
                IF rescan.found:
                    adapted = step.clone(); adapted.click_x = rescan.rect.cx; adapted.click_y = rescan.rect.cy
                    RETURN RecoveryAction(ADAPT, adapted)
            RETURN RecoveryAction(ABORT, "ELEMENT_RELOCATED")

        CASE APP_CRASHED:
            dialog = body.uia_find_window(name_contains="stopped working")
            IF dialog.found: body.ahk_execute(AHKCommand("Click", dialog.close_btn))
            IF step.target_app IN RESTART_SAFE_LIST:
                body.ahk_execute(AHKCommand("Run", step.app_path)); sleep(3000)
                IF body.uia_verify_window_by_class(step.target_class):
                    IF brain.checkpoint_exists(step.target_app, step.mission_id):
                        brain.restore_checkpoint(step.target_app, step.mission_id)
                    RETURN RecoveryAction(ADAPT, step)
            RETURN RecoveryAction(ABORT, "APP_CRASHED")

        CASE STATE_MISMATCH:
            fresh = body.capture_state()
            met = [pc FOR pc IN step.postconditions IF evaluate_single(pc, fresh)]
            coverage = len(met) / len(step.postconditions)
            IF coverage >= 0.50:
                FOR pc IN step.postconditions IF pc NOT IN met:
                    correction = brain.suggest_correction(pc, fresh)
                    IF correction: body.ahk_execute(correction); sleep(300)
                fresh2 = body.capture_state()
                IF ALL(evaluate_single(pc, fresh2) FOR pc IN step.postconditions):
                    RETURN RecoveryAction(ADAPT, step)
            RETURN RecoveryAction(ABORT, "STATE_MISMATCH")

        DEFAULT:
            RETURN RecoveryAction(ABORT, "UNKNOWN_ERROR")
```

---

## 4. TDD Anchors

```
TEST hda_execute_rejects_without_permit:
    permit = make_permit(capabilities=[Capability.READ_CLIPBOARD])
    mission = make_mission(required=[Capability.DESKTOP_CLICK])
    result = hda_execute(mission, brain, body, permit)
    ASSERT result.status == REJECTED AND "missing_capability" IN result.reason

TEST hda_execute_blocks_below_actuation_threshold:
    mock_fate_gate(score=0.10); mock_snr_scorer(composite=0.30)
    result = hda_execute(mission, brain, body, valid_permit)
    ASSERT result.status == BLOCKED AND result.p_a < 0.85

TEST psi_ast_validate_kills_on_run_command:
    ASSERT psi_ast_validate("WinActivate, Notepad\nRun, cmd.exe /c del *.*") == 0

TEST psi_ast_validate_passes_clean_script:
    ASSERT psi_ast_validate("WinActivate, Notepad\nSendInput, Hello\nSleep, 100") == 1

TEST psi_ast_validate_kills_on_path_traversal:
    ASSERT psi_ast_validate("FileAppend, x, C:\\Windows\\..\\..\\etc\\passwd") == 0

TEST closed_loop_verify_passes_all_postconditions:
    before = make_state(uia=TREE_WITHOUT_BTN); after = make_state(uia=TREE_WITH_BTN)
    result = closed_loop_verify(before, after, [Postcond(ELEMENT_EXISTS, "Submit", "Button")])
    ASSERT result.passed == True AND result.coverage == 1.0

TEST closed_loop_verify_detects_missing_element:
    result = closed_loop_verify(make_state(uia=WITH_BTN), make_state(uia=WITH_BTN),
                                 [Postcond(ELEMENT_GONE, "Submit", "Button")])
    ASSERT result.passed == False

TEST ahk_recovery_adapts_on_minimized_window:
    mock_uia_find_window(found=True, state="minimized", hwnd=12345)
    recovery = ahk_error_recovery(WINDOW_DISAPPEARED, step, body, brain)
    ASSERT recovery.action == ADAPT AND body.last_cmd.name == "WinActivate"
```

---

## 5. Cross-References

### Python Modules
- `core/sovereign/permit.py` -- `Capability` enum, `Permit`, `MAX_DELEGATION_DEPTH`, HMAC-SHA256. Mirrors Rust `bizra-telescript`.
- `core/bridges/desktop_bridge.py` -- JSON-RPC proxy (TCP:9742), auth envelope, HDA verb dispatch, rate limiting.
- `core/bridges/ghost_ws.py` -- WebSocket overlay (port 9743), RPC proxy, auth header injection.
- `core/integration/constants.py` -- `UNIFIED_IHSAN_THRESHOLD` (0.95), `SNR_THRESHOLD` (0.85), `GATE_FAIL_MODE` ("closed"), `ACTION_BUS_MAX_CONCURRENT` (10), `ACTION_BUS_MAX_PER_HOUR` (100).
- `core/iaas/snr_v2_adapter.py` -- `SNRv2Adapter` for atomic/edge/path scoring.
- `core/proof_engine/evidence_ledger.py` -- Merkle-chained append-only receipt log with cross-process file lock.
- `core/reasoning/entropy_router.py` -- `EntropyRouter`, theta=4.5 bit threshold (Brain Process routing).

### Rust Crates
- `bizra-omega/bizra-agent/src/permit_guard.rs` -- `PermitBudgetConfig`, `PermitUsage`, per-plan budget enforcement.
- `bizra-omega/bizra-agent/src/action_bus.rs` -- Event-driven action dispatch, rate limiting.
- `bizra-omega/bizra-agent/src/action_types.rs` -- `ActionKind`, `ActionPlan`, `PlannedStep`, `ActionChannel`.
- `bizra-omega/bizra-agent/src/sub_agent.rs` -- `SubAgent`, `SubAgentPermit` with degradation support.
- `bizra-omega/bizra-hooks/src/event_bus.rs` -- 8-shard EventBus (FNV-1a), HDA events on `hda/*` namespace.
- `bizra-omega/bizra-hooks/src/ihsan_gate.rs` -- Ihsan scoring gate, final wall before actuation.

### AHK Layer
- `filedfs/ahk_bridge.ahk` -- AHK bridge server (TCP:9742), 8 HDA verbs.
- `scripts/ghost_overlay.ahk` -- Frosted-glass proactive overlay (Windows-native).

### Atlas v5 Phases
- Phase 00 -- System Overview (FR-003: 12-step value loop, steps 4-5 are HDA)
- Phase 01 -- Sovereign Node (FR-010: genesis bootstraps HDA kinetic layer)
- Phase 02 -- Cognition Engine (FR-020: Entropy Router feeds Brain tier; FR-021: Diffusion SNR for actuation)
- Phase 03 -- Agent Orchestration (FR-030: PAT-7 delegates via TeleScript permits; FR-032: negotiated budget constrains HDA)
- Phase 06 -- Governance + Soul (FATE Gate, Ihsan Wall, Crown Verification -- the G term in P_A)

### Standing on Giants
- General Magic (1994): Telescript permits -- capability-scoped mobile agents
- Fitts (1954): Target acquisition law -- click coordinate tolerance
- Norman (1988): Affordance + feedback -- closed-loop as gulf-of-evaluation bridge
- Boyd (1976): OODA -- perceive-orient-decide-act maps to capture-reason-actuate-verify
- Shannon (1948): SNR as the quality metric for actuation worthiness
- Al-Ghazali (1095): Ihsan as the excellence threshold that gates all action
