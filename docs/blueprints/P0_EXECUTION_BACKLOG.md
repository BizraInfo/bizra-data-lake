# P0 Execution Backlog - Integrity Sprint

Version: 1.0
Status: Ready
Duration: 2 weeks
Goal: Align security, policy, and documentation and establish the Integrity Gate.

## Integrity Release Gate (Must Pass)

- No committed secrets in repo; credentials rotated.
- Auth is fail-closed in Rust and Python.
- SAT consensus semantics are consistent in code and docs.
- MCP per-request allowlist is enforced.
- Policy rejections use explicit HTTP status and error codes.
- Request-id is generated and propagated.
- OpenAPI and docs match runtime behavior.
- Parity checks pass and are enforced in CI.

## Backlog Items (Priority Order)

P0-1 Remove committed secrets and rotate credentials
- Owner: Security + Platform
- Actions:
  - Remove .env from VCS history and working tree.
  - Rotate API tokens, DB passwords, and provider keys.
  - Replace with .env.example or secrets manager entries.
- DoD/Evidence:
  - Secrets scan shows no findings.
  - New credentials validated in dev.

P0-2 Confirm veto-only SAT consensus across code and docs
- Owner: Core + Docs
- Actions:
  - Confirm veto-only consensus (all 5 approve).
  - Update src/sat.rs and all docs to match.
  - Update tests to assert the chosen policy.
- DoD/Evidence:
  - SAT unit tests pass for policy behavior.
  - README and API docs match runtime behavior.

P0-3 Enforce fail-closed auth in Rust HTTP
- Owner: Platform
- Actions:
  - If BIZRA_API_TOKEN is missing, do not start or return 503.
  - Remove logging of generated tokens.
  - Keep auth handling consistent with core/main.py.
- DoD/Evidence:
  - Startup fails or blocks when token missing.
  - No secret values logged.

P0-4 Implement per-request MCP tool allowlist
- Owner: Core
- Actions:
  - Enforce mcp_tools_whitelist in EnhancedDualAgenticRequest.
  - Default to deny if whitelist is provided and empty.
  - Update OpenAPI and tests.
- DoD/Evidence:
  - MCP calls outside whitelist are rejected with policy error.
  - OpenAPI matches behavior.

P0-5 Standardize policy rejection HTTP codes
- Owner: Platform
- Actions:
  - Map SAT rejection to 403 or 422 with clear error code.
  - Map Ihsan gate failure to 422 with explanation.
  - Preserve 500 only for server faults.
- DoD/Evidence:
  - API responses differentiate policy vs system errors.
  - Tests cover each rejection path.

P0-6 Add request-id propagation
- Owner: Platform + Core
- Actions:
  - Generate request-id at HTTP entry if missing.
  - Include request-id in logs, receipts, and metrics.
  - Return request-id header in responses.
- DoD/Evidence:
  - Logs and receipts include request-id.
  - OpenAPI documents request-id header.

P0-7 Update docs and OpenAPI for alignment
- Owner: Docs
- Actions:
  - Align README, API docs, and request-lifecycle with runtime.
  - Update OpenAPI error responses and auth behavior.
- DoD/Evidence:
  - Docs reflect chosen SAT policy and auth rules.
  - OpenAPI matches runtime responses.

P0-8 Add CI policy gates
- Owner: DevOps
- Actions:
  - Add scripts/check_parity.py to CI.
  - Add secret scanning to CI (gitleaks or equivalent).
  - Block merge on policy parity failures.
- DoD/Evidence:
  - CI fails on parity or secret scan violations.

P0-9 Configure bind address for container use
- Owner: Platform
- Actions:
  - Make HTTP bind host configurable via env var.
  - Default to loopback for local; 0.0.0.0 for container.
- DoD/Evidence:
  - Container can accept traffic on configured host.

P0-10 Validate Integrity Gate
- Owner: Release Manager
- Actions:
  - Run P0 test suite and policy checks.
  - Capture evidence artifacts (logs, receipts, CI results).
- DoD/Evidence:
  - Integrity Gate checklist complete and attached.

## Sprint Rituals

- Kickoff: confirm policy choice and acceptance criteria.
- Mid-sprint review: verify parity and docs alignment.
- Release review: Integrity Gate checklist and evidence archive.
