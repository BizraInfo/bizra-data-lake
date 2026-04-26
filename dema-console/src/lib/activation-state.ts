import type { ActivatePrincipalResponseContract } from "@/bindings/ActivatePrincipalResponseContract";
import type { GatewayOutcome } from "@/lib/gateway-client";
import type { TrustState } from "@/lib/types";

export const UNAVAILABLE_VALUE = "—";

export type ActivationStatus =
  | "idle"
  | "activating"
  | "refused"
  | "unreachable"
  | "activated";

export function createInactiveTrustState(): TrustState {
  return {
    principalId: null,
    principalName: "",
    level: null,
    score: null,
    maxScore: null,
    lastVerified: null,
    sessionId: null,
    isActive: false,
    chainHead: null,
    missionId: null,
    missionReceiptId: null,
    activationReceiptId: null,
    finalStage: null,
    profileHash: null,
    cacheWarning: null,
  };
}

export function projectActivatedTrustState(
  principalName: string,
  activation: ActivatePrincipalResponseContract,
): TrustState {
  return {
    ...createInactiveTrustState(),
    isActive: true,
    principalId: activation.principalId,
    principalName,
    chainHead: activation.chainHead,
    missionId: activation.missionId,
    missionReceiptId: activation.missionReceiptId,
    activationReceiptId: activation.principalActivationReceiptId,
    finalStage: activation.finalStage,
    profileHash: activation.profileHash,
    cacheWarning: activation.cache_warning,
  };
}

export function summarizeActivationFailure(
  outcome: GatewayOutcome<ActivatePrincipalResponseContract>,
): string {
  switch (outcome.kind) {
    case "refused":
      return `${outcome.error.code}: ${outcome.error.message}`;
    case "unreachable":
      return `Cognition gateway unreachable: ${outcome.reason}`;
    case "ok":
      return "";
  }
}

export function formatOptionalText(value: string | null | undefined): string {
  const normalized = value?.trim();
  return normalized ? normalized : UNAVAILABLE_VALUE;
}

export function formatTrustLevel(level: TrustState["level"]): string {
  return level ?? UNAVAILABLE_VALUE;
}

export function formatTrustScore(
  score: TrustState["score"],
  maxScore: TrustState["maxScore"],
): string {
  if (typeof score !== "number" || typeof maxScore !== "number" || maxScore <= 0) {
    return UNAVAILABLE_VALUE;
  }
  return `${score}/${maxScore}`;
}

export function trustScoreProgress(
  score: TrustState["score"],
  maxScore: TrustState["maxScore"],
): number {
  if (typeof score !== "number" || typeof maxScore !== "number" || maxScore <= 0) {
    return 0;
  }
  return (score / maxScore) * 100;
}
