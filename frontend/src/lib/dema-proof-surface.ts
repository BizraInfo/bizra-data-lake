import type {
  DecisionVerdict,
  DisplayTruthLabel,
} from "./dema-csl";

export interface ProofSurfacePayload {
  schema_version: string;
  surface_id: string;
  claim: string;
  source: string;
  truth_label: DisplayTruthLabel;
  decision: DecisionVerdict;
  decision_reason: string;
  evidence_auditor_verdict: DecisionVerdict;
  converged: boolean;
  receipt_id: string | null;
  receipt_export_ready: boolean;
  evidence_refs: string[];
  reasons: string[];
  sources: string[];
  blocking_sources: string[];
  missing_sources: string[];
}

interface MissionReceiptLike {
  mission_id: string;
  receipt_id: string;
  evidence_receipt_id?: string | null;
  status: "COMPLETE" | "PARTIAL" | "FAILED" | "BLOCKED";
  hash_chain_ref: string;
  reasoning_proof?: {
    receipt_id: string;
    vrg_root: string;
  } | null;
  proof_surface?: ProofSurfacePayload | null;
}

function receiptEvidenceRefs(receipt: MissionReceiptLike): string[] {
  return [
    receipt.evidence_receipt_id,
    receipt.hash_chain_ref,
    receipt.reasoning_proof?.receipt_id,
    receipt.reasoning_proof?.vrg_root,
  ].filter((value): value is string => Boolean(value));
}

function decisionFromStatus(status: MissionReceiptLike["status"]): DecisionVerdict {
  if (status === "FAILED" || status === "BLOCKED") {
    return "forbid";
  }
  return "require_approval";
}

export function proofSurfaceFromMissionReceipt(
  receipt: MissionReceiptLike,
): ProofSurfacePayload {
  const decision = decisionFromStatus(receipt.status);
  const evidenceRefs = receiptEvidenceRefs(receipt);
  const primaryEvidence = receipt.hash_chain_ref || receipt.receipt_id;

  return {
    schema_version: "0.1.0",
    surface_id: `derived:${primaryEvidence}`,
    claim: `Mission ${receipt.mission_id || receipt.receipt_id} produced a receipt-backed outcome.`,
    source: "mission_receipt",
    truth_label: "DERIVED",
    decision,
    decision_reason: "backend_proof_surface_absent; derived_from_mission_receipt",
    evidence_auditor_verdict: decision,
    converged: false,
    receipt_id: receipt.receipt_id || null,
    receipt_export_ready: false,
    evidence_refs: evidenceRefs,
    reasons: ["backend_proof_surface_absent"],
    sources: ["mission_receipt"],
    blocking_sources:
      decision === "forbid" ? ["mission_receipt"] : [],
    missing_sources: ["backend_proof_surface"],
  };
}

export function proofSurfaceForMissionReceipt(
  receipt: MissionReceiptLike,
): ProofSurfacePayload {
  return receipt.proof_surface ?? proofSurfaceFromMissionReceipt(receipt);
}
