// AUTO-GENERATED from core/dema/csl/labels.py — do not edit by hand.
// Regenerate with: python scripts/dema/dema_csl.py emit-ts --write
//
// Python is the source of truth. The Python drift test
// (tests/scripts/test_dema_csl.py::test_typescript_mirror_matches_python)
// fails CI if this file falls out of sync.

export const CSL_SCHEMA_VERSION = "0.1.0" as const;

export const RECEIPT_TRUTH_LABELS = ["MEASURED", "DERIVED", "PLANNED", "SANDBOX"] as const;
export type ReceiptTruthLabel = (typeof RECEIPT_TRUTH_LABELS)[number];

export const DISPLAY_TRUTH_LABELS = ["MEASURED", "DERIVED", "PLANNED", "SANDBOX", "UNKNOWN"] as const;
export type DisplayTruthLabel = (typeof DISPLAY_TRUTH_LABELS)[number];

export const MISSION_TRUTH_LABELS = ["MEASURED", "DERIVED", "PLANNED", "SANDBOX", "UNKNOWN"] as const;
export type MissionTruthLabel = (typeof MISSION_TRUTH_LABELS)[number];

export const RISK_LEVELS = ["low", "medium", "high"] as const;
export type RiskLevel = (typeof RISK_LEVELS)[number];

export const APPROVAL_STATUSES = ["granted", "pending", "n/a", "denied"] as const;
export type ApprovalStatus = (typeof APPROVAL_STATUSES)[number];

export const DECISION_VERDICTS = ["auto_low_risk", "notify", "require_approval", "forbid"] as const;
export type DecisionVerdict = (typeof DECISION_VERDICTS)[number];
