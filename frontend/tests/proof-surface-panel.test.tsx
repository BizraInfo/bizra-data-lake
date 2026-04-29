import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import ProofSurfacePanel from "../src/components/terminal/proof-surface-panel";
import type { MissionReceipt } from "../src/hooks/use-sovereign-api";
import {
  proofSurfaceForMissionReceipt,
  proofSurfaceFromMissionReceipt,
  type ProofSurfacePayload,
} from "../src/lib/dema-proof-surface";

afterEach(() => {
  cleanup();
});

const backendSurface: ProofSurfacePayload = {
  schema_version: "0.1.0",
  surface_id: "surface-abc",
  claim: "Sovereignty reveal is backed by a receipt",
  source: "receipt panel",
  truth_label: "MEASURED",
  decision: "notify",
  decision_reason: "proof_surface_ready",
  evidence_auditor_verdict: "notify",
  converged: true,
  receipt_id: "rcpt-1",
  receipt_export_ready: true,
  evidence_refs: ["rcpt-1", "vrg-root"],
  reasons: [],
  sources: ["receipt", "auditor"],
  blocking_sources: [],
  missing_sources: [],
};

const receipt: MissionReceipt = {
  mission_id: "mission-1",
  receipt_id: "rcpt-1",
  evidence_receipt_id: "evidence-1",
  status: "COMPLETE",
  synthesis: "Mission completed.",
  ihsan_score: 0.97,
  snr_score: 0.91,
  duration_ms: 42,
  channels_executed: [],
  execution_path: "SYSTEM_2_NOVEL",
  wallet_delta: {
    seed: 1,
    bloom: 0,
  },
  reflex_delta: {
    compiled: false,
    near_compile: false,
    compile_count: 0,
    threshold: 3,
  },
  memory_delta: {
    episodic: 1,
    semantic: 0,
    procedural: 0,
  },
  hash_chain_ref: "hash-chain-ref",
  action_count: 1,
  reflex_pattern: "",
  reflex_latency_ms: 0,
  comparison_s2_avg_ms: 0,
  reasoning_proof: {
    mode: "verified",
    vrg_root: "vrg-root",
    verified: true,
    receipt_id: "reasoning-rcpt",
    status: "ACCEPTED",
    payload_digest: "digest",
    branch_count: 2,
    surviving_branches: 2,
    detail: "verified",
  },
};

describe("ProofSurfacePanel", () => {
  it("renders backend proof surface payload without downgrading export readiness", () => {
    render(<ProofSurfacePanel surface={backendSurface} />);

    expect(screen.getByTestId("proof-surface-panel")).toBeInTheDocument();
    expect(screen.getByTestId("proof-surface-claim")).toHaveTextContent(
      "Sovereignty reveal is backed by a receipt",
    );
    expect(screen.getByTestId("proof-surface-source")).toHaveTextContent(
      "receipt panel",
    );
    expect(screen.getByTestId("proof-surface-truth-label")).toHaveTextContent(
      "MEASURED",
    );
    expect(screen.getByTestId("proof-surface-decision")).toHaveTextContent(
      "notify",
    );
    expect(screen.getByTestId("proof-surface-auditor")).toHaveTextContent(
      "notify",
    );
    expect(screen.getByTestId("proof-surface-converged")).toHaveTextContent(
      "yes",
    );
    expect(screen.getByTestId("proof-surface-export")).toHaveTextContent(
      "ready",
    );
    expect(screen.getByText("rcpt-1, vrg-root")).toBeInTheDocument();
  });

  it("derives an honest receipt-backed fallback when backend proof_surface is absent", () => {
    const surface = proofSurfaceFromMissionReceipt(receipt);

    render(<ProofSurfacePanel surface={surface} />);

    expect(screen.getByTestId("proof-surface-truth-label")).toHaveTextContent(
      "DERIVED",
    );
    expect(screen.getByTestId("proof-surface-decision")).toHaveTextContent(
      "require_approval",
    );
    expect(screen.getByTestId("proof-surface-converged")).toHaveTextContent(
      "no",
    );
    expect(screen.getByTestId("proof-surface-export")).toHaveTextContent(
      "locked",
    );
    expect(screen.getByText("backend_proof_surface")).toBeInTheDocument();
    expect(screen.getByText(/derived_from_mission_receipt/)).toBeInTheDocument();
    expect(screen.getByTestId("proof-surface-id")).toHaveTextContent(
      "derived:hash-chain-ref",
    );
  });

  it("prefers backend proof_surface when a receipt already carries the contract", () => {
    const surface = proofSurfaceForMissionReceipt({
      ...receipt,
      proof_surface: backendSurface,
    });

    expect(surface).toBe(backendSurface);
  });
});
