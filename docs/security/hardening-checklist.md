# BIZRA Hardening Checklist (Stage 1)

Updated: 2026-02-20

| Item | Owner | Verification Method | Status |
|---|---|---|---|
| Enforce fail-closed action validation (unknown/missing payload/executor) | Node0 Core | `cargo test -p bizra-agent action_bus` | Complete |
| Enforce permit capability mapping and resource budgets | Node0 Core | `cargo test -p bizra-agent permit_guard` | Complete |
| Enforce reflex active-mode policy-hash requirement | Node0 Core | runtime tests (`reflex_active_fail_closed_without_policy_hash`) | Complete |
| Quarantine on guardian veto/revalidation failure | Node0 Core | runtime/reflex tests | Complete |
| Add hash-chained `actions.log` receipts | Node0 Core | `cargo test -p bizra-node persistence::save_and_load_action_log_roundtrip` | Complete |
| Load-time receipt chain verification | Node0 Core | persistence tests + tamper simulation (manual) | Complete |
| Require desktop bridge token for action execution | Bridge Integration | runtime action call without token returns fail-closed | Complete |
| Keep bridge localhost-only | Bridge Integration | inspect `core/bridges/desktop_bridge.py` bind host check | Complete |
| Add PWA offline queue and reconnect flush with re-checks | Filedfs | manual QA + queue module tests (manual) | Complete |
| Document Specified vs Implemented vs Verified | Program | `STATUS.md` maintained in repo root | Complete |
| Define external audit gate and scope | Program + Security | `docs/security/audit-stage2-plan.md` | Complete |
| Hardware-backed key storage roadmap (TPM/HSM) | Security | design task + implementation issue | Open |
| Wire protocol authentication for non-local deployment | Security | protocol hardening spec + implementation | Open |
| Python↔Rust guardian unification for full MCP path | Core Integration | integration sprint + regression suite | Open |

