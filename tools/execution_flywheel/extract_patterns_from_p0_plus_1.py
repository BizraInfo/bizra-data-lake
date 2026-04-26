"""Extract the P0+1 hardening lessons into pattern dicts for persistence.

Emits the five initial patterns compatible with schemas.Pattern.from_dict()
and the patterns.yaml grammar. Default is a dry-run print; pass --write <path>
to persist the patterns as JSON at the given path.

Does not call GitHub. Does not mutate the packaged patterns.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PATTERNS: list[dict] = [
    {
        "pattern_id": "PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT",
        "name": "Verify origin branch before editing from review feedback",
        "severity": "critical",
        "triggers": [
            {"keyword": "review_requests_change", "description": "A review comment requests a code change"},
            {"keyword": "pr_has_commits_after_reviewed_sha", "description": "PR head SHA differs from reviewed SHA"},
            {"keyword": "local_branch_differs_from_pr_head", "description": "Working tree is not on PR branch"},
        ],
        "risks": ["duplicate fix", "divergent implementation", "polluted working tree", "stale review loop"],
        "guard_actions": [
            "fetch origin PR branch",
            "inspect origin/<branch>:<file>",
            "compare reviewed SHA against PR head SHA",
            "abort edit if requested change already exists",
        ],
        "source": ["PR #49", "CodeRabbit stale CHANGES_REQUESTED", "fix commit c09cc95c already pushed"],
    },
    {
        "pattern_id": "AUDIT_YAML_INLINE_COMMENT_PARSE_FAILURE",
        "name": "Audit engine fails on YAML with inline comments near numeric config",
        "severity": "high",
        "default_decision": "REVALIDATE",
        "triggers": [
            {"keyword": "audit_engine_crash", "description": "Audit engine terminates during YAML config load"},
            {"keyword": "yaml_typeerror_int_vs_str", "description": "TypeError comparing int and str during parse"},
            {"keyword": "inline_comment_near_numeric_value", "description": "Inline YAML comments near numeric config values"},
        ],
        "risks": ["false production outage signal", "blocked security audit", "operator distrust of audit tooling"],
        "guard_actions": [
            "sanitize inline comments outside quoted strings",
            "add regression test for YAML loader with inline comments",
        ],
        "source": ["P0+1 hardening"],
    },
    {
        "pattern_id": "SECRET_SCANNER_SNR_NOISE_COLLAPSE",
        "name": "Secret scanner output drowns in self-scan and placeholder noise",
        "severity": "high",
        "default_decision": "REVALIDATE",
        "triggers": [
            {"keyword": "high_secret_finding_count", "description": "Scanner reports many hits that are not true secrets"},
            {"keyword": "self_scan_matches", "description": "Scanner matches its own log or fixtures"},
            {"keyword": "placeholder_and_env_substitution_matches", "description": "Matches on $VARS and placeholders"},
        ],
        "risks": ["true positives hidden in noise", "false rotation events", "wasted triage time"],
        "guard_actions": [
            "dedupe overlapping scanner roots",
            "exclude logs and scanner self-reference",
            "suppress safe placeholders and env substitutions",
            "rerun audit into /tmp for diff comparison",
        ],
        "source": ["P0+1 hardening"],
    },
    {
        "pattern_id": "DEV_DEFAULT_CREDENTIAL_FALLBACK_TRUTH_DEBT",
        "name": "Committed dev-default credential fallback creates truth debt",
        "severity": "critical",
        "default_decision": "ABORT",
        "triggers": [
            {"keyword": "default_dsn_or_redis_or_neo4j_fallback", "description": "Committed default fallback credential for backing stores"},
            {"keyword": "credential_url_printed_in_logs", "description": "Connection URL with credentials appears in logs"},
        ],
        "risks": ["silent credential leak to logs", "prod degrades to dev-default in wrong environment", "rotation false negative"],
        "guard_actions": [
            "require operator-supplied env var",
            "fail closed for strict backend modes",
            "degrade only to local non-network persistence when explicitly safe",
            "mask connection URLs in logs",
        ],
        "source": ["P0+1 hardening"],
    },
    {
        "pattern_id": "BOTTLENECK_SHIFT_AFTER_SECRET_GATE_CLEARS",
        "name": "Priority should shift from secret triage to public claim discipline",
        "severity": "high",
        "default_decision": "REVALIDATE",
        "triggers": [
            {"keyword": "secret_findings_zero", "description": "Scanner reports zero findings after hardening"},
            {"keyword": "rotation_not_required", "description": "No rotation action open"},
            {"keyword": "public_claims_risky", "description": "Public claims include prohibited or proof-required items"},
        ],
        "risks": ["complacency after secret gate closes", "continued focus on closed axis", "missed reputational or compliance risk"],
        "guard_actions": [
            "shift priority from secret triage to public claim discipline",
            "recommend P0.2 website claim cleanup",
        ],
        "source": ["P0+1 hardening"],
    },
]


def build_all_patterns() -> list[dict]:
    return [dict(p) for p in PATTERNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="P0+1 pattern extractor (default: dry-run)")
    parser.add_argument("--write", help="Optional path to write patterns as JSON (opt-in only)")
    args = parser.parse_args()
    patterns = build_all_patterns()
    payload = json.dumps({"patterns": patterns}, indent=2)
    if args.write:
        Path(args.write).write_text(payload, encoding="utf-8")
        print(f"Wrote {len(patterns)} patterns to {args.write}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
