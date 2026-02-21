# External Security Audit Plan (Stage 2)

Updated: 2026-02-20

## Trigger (Entry Criteria)
Stage 2 audit begins only after all are true:
1. User Zero action flow is in daily use for at least 7 consecutive days.
2. Action E2E path is stable (`PLAN_ACTION -> RUN_ACTION -> ACTION_STATUS -> ACTION_HISTORY`).
3. `STATUS.md` reflects implemented/verified truth for sprint scope.
4. No critical regressions in `cargo test -p bizra-agent -p bizra-node`.

## Audit Scope
1. `bizra-agent`:
   - runtime gating and quarantine behavior,
   - permit guard/resource limits,
   - sub-agent spawn controls.
2. `bizra-node`:
   - protocol parser/handler safety,
   - action executor bridge boundary and fail-closed behavior,
   - persistence integrity (`reflex.cache`, `actions.log` chain).
3. Python bridge boundary:
   - auth enforcement,
   - localhost binding,
   - method exposure minimization.

## Required Deliverables
1. Threat findings report with severity and reproducibility steps.
2. Exploitability assessment for each high/critical issue.
3. Recommended remediations and retest evidence.
4. Signed attestation of tested commit hash and environment.

## Out of Scope for Stage 2
1. Token economics and federation consensus layer.
2. Cross-node Telescript transport.
3. UI/visual style issues not impacting security or integrity.

## Exit Gate
1. Zero unresolved critical findings.
2. All high findings mitigated or explicitly risk-accepted with owner/date.
3. Retest confirms mitigation effectiveness.

