/**
 * Dema Goal Surface — derives the §9 four-state model from the live
 * sovereign API surface.
 *
 *   Current State → Ideal State → Gap → Next Admissible Action
 *
 * Truth-labeled, no fake metrics. When backend data is unavailable, fields
 * fall back to PLANNED (operator-defined intent) or UNKNOWN (no source).
 *
 * Phase A1 v0.1 — there is no /v1/goal endpoint yet, so:
 *   - Current   = MEASURED iff health + chain data are present
 *   - Ideal     = PLANNED  (operator-defined; will move to MEASURED when a
 *                 dedicated goal endpoint ships)
 *   - Gap       = DERIVED  (computed from Current vs Ideal)
 *   - Next      = DERIVED  (admissibility-gated suggestion)
 */

import {
  useChainLatest,
  useNodeLifecycle,
  useSovereignHealth,
} from "./use-sovereign-api";

import type { GoalTruthLabel } from "../components/terminal/goal-truth-badge";

export interface GoalStateField {
  body: string;
  truthLabel: GoalTruthLabel;
  hint?: string;
}

export interface DemaGoalSnapshot {
  current: GoalStateField;
  ideal: GoalStateField;
  gap: GoalStateField;
  nextAdmissibleAction: GoalStateField;
  trust: {
    chainHeadShort: string | null;
    chainLength: number | null;
    receiptKind: string | null;
    ihsanBand: "ideal" | "warn" | "halt" | "unknown";
    ihsanScore: number | null;
    snrScore: number | null;
    gini: number | null;
    fateActive: boolean;
    truthLabel: GoalTruthLabel;
  };
  loading: boolean;
  error: string | null;
}

const IHSAN_IDEAL = 0.95;
const IHSAN_WARN = 0.85;

function ihsanBand(score: number | null): "ideal" | "warn" | "halt" | "unknown" {
  if (score === null) {
    return "unknown";
  }
  if (score >= IHSAN_IDEAL) {
    return "ideal";
  }
  if (score >= IHSAN_WARN) {
    return "warn";
  }
  return "halt";
}

export function useDemaGoalState(): DemaGoalSnapshot {
  const { data: health, error: healthError, loading: healthLoading } =
    useSovereignHealth();
  const { data: lifecycle, error: lifecycleError } = useNodeLifecycle();
  const { data: chainLatest, error: chainError } = useChainLatest();

  // The useSovereignHealth hook returns a fallback object when /v1/health is
  // unreachable; that fallback has ihsan_score=0/snr_score=0/gini=0 by
  // design, but those zeros do NOT represent measured truth. Only treat the
  // numeric fields as real when the upstream status reports healthy AND no
  // error fired.
  const healthIsLive =
    Boolean(health) && !healthError && health?.status === "healthy";
  const ihsanScore =
    healthIsLive && typeof health?.ihsan_score === "number"
      ? health.ihsan_score
      : null;
  const snrScore =
    healthIsLive && typeof health?.snr_score === "number"
      ? health.snr_score
      : null;
  const gini =
    healthIsLive && typeof health?.gini === "number" ? health.gini : null;
  const band = ihsanBand(ihsanScore);

  const chainHead = chainLatest?.head ?? "";
  const chainLength = chainLatest?.length ?? 0;
  const chainHeadShort = chainHead ? chainHead.slice(0, 8) : null;
  const receiptKind = chainLatest?.latestReceipt?.kind ?? null;

  const trustHasMeasuredData = Boolean(
    health && (ihsanScore !== null || snrScore !== null || gini !== null),
  );

  // ── Current State ─────────────────────────────────────────────────
  let currentBody = "";
  let currentLabel: GoalTruthLabel = "UNKNOWN";
  if (healthIsLive && health) {
    const lifecycleStage = lifecycle?.current_stage ?? "—";
    const liveStatus = health.live_status ?? (health.running ? "LIVE" : "OFFLINE");
    const ihsanText =
      ihsanScore === null ? "ihsān unknown" : `ihsān ${ihsanScore.toFixed(2)}`;
    const giniText = gini === null ? "gini unknown" : `gini ${gini.toFixed(2)}`;
    currentBody = [
      `node: ${liveStatus}`,
      `lifecycle: ${lifecycleStage}`,
      ihsanText,
      giniText,
      chainHead
        ? `chain head ${chainHeadShort} (#${chainLength})`
        : "chain at genesis",
    ].join("\n");
    currentLabel = trustHasMeasuredData ? "MEASURED" : "DERIVED";
  } else {
    currentBody = "node state not available — start Dema service or check /v1/health";
    currentLabel = "UNKNOWN";
  }

  // ── Ideal State ───────────────────────────────────────────────────
  // No /v1/goal endpoint yet — this stays PLANNED until the operator can
  // write an ideal-state target. v0.1 reflects the Masterplan §1 default.
  const idealBody = [
    "Sovereign Node0 alive end-to-end with measured ihsān ≥ 0.95.",
    "Two-device pilot proven. PoI sandbox-only.",
  ].join(" ");
  const idealLabel: GoalTruthLabel = "PLANNED";

  // ── Gap ───────────────────────────────────────────────────────────
  let gapBody = "";
  let gapLabel: GoalTruthLabel = "DERIVED";
  if (!healthIsLive) {
    gapBody = "service offline — cannot compute gap";
    gapLabel = "UNKNOWN";
  } else if (band === "halt") {
    gapBody = "ihsān below halt floor — admissibility blocked";
  } else if (band === "warn") {
    gapBody = "ihsān in warn band — bring Current toward ≥ 0.95 before next mission";
  } else if (band === "ideal" && chainLength === 0) {
    gapBody = "chain at genesis — first mission receipt not yet recorded";
  } else if (band === "ideal") {
    gapBody = "no measurable gap on the constitutional axis right now";
  } else {
    gapBody = "ihsān unknown — measurement pending";
    gapLabel = "UNKNOWN";
  }

  // ── Next Admissible Action ────────────────────────────────────────
  let nextBody = "";
  let nextLabel: GoalTruthLabel = "DERIVED";
  if (!healthIsLive) {
    nextBody = "start Dema ambient service: scripts/dema/dema_daemon.py --once";
    nextLabel = "PLANNED";
  } else if (band === "halt") {
    nextBody = "halt: do not submit new missions until ihsān recovers";
  } else if (band === "warn") {
    nextBody = "submit a low-risk mission to recover ihsān (Mission tab)";
  } else if (chainLength === 0) {
    nextBody = "submit first mission to seal genesis receipt (Mission tab)";
  } else {
    nextBody = "submit next mission via Mission tab; verify chain head advances";
  }

  // ── Trust summary ─────────────────────────────────────────────────
  const trust = {
    chainHeadShort,
    chainLength: chainHead ? chainLength : null,
    receiptKind,
    ihsanBand: band,
    ihsanScore,
    snrScore,
    gini,
    fateActive: band !== "halt" && band !== "unknown",
    truthLabel: trustHasMeasuredData
      ? ("MEASURED" as GoalTruthLabel)
      : ("UNKNOWN" as GoalTruthLabel),
  };

  return {
    current: { body: currentBody, truthLabel: currentLabel },
    ideal: {
      body: idealBody,
      truthLabel: idealLabel,
      hint: "operator-defined target until /v1/goal ships",
    },
    gap: { body: gapBody, truthLabel: gapLabel },
    nextAdmissibleAction: { body: nextBody, truthLabel: nextLabel },
    trust,
    loading: healthLoading,
    error: healthError ?? lifecycleError ?? chainError ?? null,
  };
}
