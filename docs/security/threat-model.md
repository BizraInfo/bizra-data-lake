# BIZRA Security Threat Model (Stage 1 Internal)

Updated: 2026-02-20
Scope: Alpha-100 Sprint 2/3 Action Layer (`bizra-agent`, `bizra-node`, desktop bridge integration, PWA shell)

## Assets
1. Sovereign identity keys (`~/.bizra/keys/*`).
2. Policy hash trust anchor (runtime config/env).
3. Knowledge state (`knowledge.seed`), reflex state (`reflex.cache`), action receipts (`actions.log`).
4. Local desktop bridge command surface (`127.0.0.1:9742`).
5. Action execution decisions and receipts.

## Trust Boundaries
1. Node protocol boundary: stdin/stdout wire protocol.
2. Rust node to Python bridge boundary: TCP JSON-RPC on localhost with auth headers.
3. UI boundary: Filedfs browser/PWA client to node bridge.
4. Persistence boundary: on-disk files under `~/.bizra/node-<hash>/`.

## Threats and Current Mitigations
1. Unauthorized desktop command invocation.
   - Mitigation: bridge token required (`BIZRA_BRIDGE_TOKEN`), fail-closed on missing/invalid token, Guardian/Permit validation.
2. Unsafe reflex/action reuse after policy changes.
   - Mitigation: policy hash binding in trigger hash, mismatch quarantine, fail-closed when policy hash missing in active mode.
3. Action log tampering.
   - Mitigation: hash-chained receipts with `prev_receipt_hash` + domain-separated BLAKE3.
4. Agent spawn explosion / uncontrolled parallelism.
   - Mitigation: spawn limits (`max_depth=2`, `max_children=5`, `max_total_active=20`), degraded permit inheritance.
5. Replay/abuse of queued mobile actions.
   - Mitigation: reconnect flush requires fresh runtime checks; queued actions are not auto-approved.

## Known Gaps (Accepted for Alpha)
1. No hardware-backed key storage (TPM/HSM not enforced).
2. Node wire protocol itself is unauthenticated if exposed beyond local process boundary.
3. No external security audit yet.

## Stage-1 Verification Evidence
1. `cargo test -p bizra-agent -p bizra-node` (action + permit + receipt + protocol tests).
2. `tests/core/mcp/test_sovereign_phase46_tools.py` Python regression.
3. Manual fail-closed checks:
   - missing bridge token,
   - missing policy hash in active reflex mode,
   - unavailable desktop bridge.

