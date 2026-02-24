# Submission Campaign (Integrity + Cost-Aware Ranking)

Use this reference for Submit and Analyze stages.

## Campaign Flow
1. Create benchmark submission payload.
2. Run anti-gaming validation.
3. Record score, cost, latency, token usage.
4. Compare against target baseline/SOTA.
5. Emit campaign summary and next-loop backlog.

## Submission Window Policy
- Use deterministic run IDs and target-specific output directories.
- Keep one bundle per target to prevent artifact collisions.
- Include versioned config snapshot in every bundle.

## Anti-Gaming Policy
Require in strict mode:
- null-model probe
- prompt-injection probes
- leak scan indicators
- consistency checks for deterministic settings

Reject submission if anti-gaming validation fails.

## Cost-Aware Ranking
Track:
- raw benchmark score
- normalized score
- cost per run
- latency total
- KAMI-style merit (score adjusted by reliability and cost)

Accept improvements only when score and operational metrics remain inside policy.

## Rollback Receipt Contract
Emit `rollback_receipt.json` with:
- `run_id`
- `target`
- `mode`
- `timestamp_utc`
- `reason_code`
- `failed_gate`
- `trigger_metric`
- `last_good_config`

Use reason codes consistently so CI and operators can automate recovery.
