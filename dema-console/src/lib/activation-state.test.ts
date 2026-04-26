import test from "node:test";
import assert from "node:assert/strict";

import type { ActivatePrincipalResponseContract } from "@/bindings/ActivatePrincipalResponseContract";
import {
  UNAVAILABLE_VALUE,
  createInactiveTrustState,
  formatOptionalText,
  formatTrustLevel,
  formatTrustScore,
  projectActivatedTrustState,
  summarizeActivationFailure,
  trustScoreProgress,
} from "@/lib/activation-state";

const ACTIVATION: ActivatePrincipalResponseContract = {
  missionId: "mission-123",
  missionReceiptId: "mission-receipt-123",
  principalActivationReceiptId: "activation-receipt-123",
  principalId: "principal-123",
  profileHash: "profile-hash-123",
  chainHead: "a".repeat(64),
  finalStage: "Receipt",
  admissibility: {
    verdict: "Permit",
    gateVerdicts: [],
    rejected: null,
  },
  cache_warning: "stale profile cache",
};

test("projectActivatedTrustState keeps only authoritative activation fields", () => {
  const projected = projectActivatedTrustState("MuMu", ACTIVATION);

  assert.equal(projected.isActive, true);
  assert.equal(projected.principalName, "MuMu");
  assert.equal(projected.principalId, ACTIVATION.principalId);
  assert.equal(projected.chainHead, ACTIVATION.chainHead);
  assert.equal(projected.missionId, ACTIVATION.missionId);
  assert.equal(projected.missionReceiptId, ACTIVATION.missionReceiptId);
  assert.equal(projected.activationReceiptId, ACTIVATION.principalActivationReceiptId);
  assert.equal(projected.finalStage, ACTIVATION.finalStage);
  assert.equal(projected.profileHash, ACTIVATION.profileHash);
  assert.equal(projected.cacheWarning, ACTIVATION.cache_warning);
  assert.equal(projected.level, null);
  assert.equal(projected.score, null);
  assert.equal(projected.maxScore, null);
  assert.equal(projected.sessionId, null);
  assert.equal(projected.lastVerified, null);
});

test("createInactiveTrustState starts fully unavailable and inactive", () => {
  const trustState = createInactiveTrustState();

  assert.equal(trustState.isActive, false);
  assert.equal(trustState.principalName, "");
  assert.equal(trustState.principalId, null);
  assert.equal(trustState.level, null);
  assert.equal(trustState.score, null);
  assert.equal(trustState.maxScore, null);
  assert.equal(trustState.chainHead, null);
});

test("activation helpers render unavailable values honestly", () => {
  assert.equal(formatOptionalText(null), UNAVAILABLE_VALUE);
  assert.equal(formatOptionalText(""), UNAVAILABLE_VALUE);
  assert.equal(formatTrustLevel(null), UNAVAILABLE_VALUE);
  assert.equal(formatTrustScore(null, null), UNAVAILABLE_VALUE);
  assert.equal(trustScoreProgress(null, null), 0);
  assert.equal(formatTrustScore(95, 100), "95/100");
  assert.equal(trustScoreProgress(95, 100), 95);
});

test("summarizeActivationFailure distinguishes refused and unreachable outcomes", () => {
  assert.equal(
    summarizeActivationFailure({
      kind: "refused",
      status: 403,
      error: {
        code: "PATH_NOT_ALLOWLISTED",
        message: "Path must be registered",
        domain: "gateway",
        admissibility: null,
      },
    }),
    "PATH_NOT_ALLOWLISTED: Path must be registered",
  );
  assert.equal(
    summarizeActivationFailure({
      kind: "unreachable",
      reason: "connect ECONNREFUSED",
    }),
    "Cognition gateway unreachable: connect ECONNREFUSED",
  );
});
