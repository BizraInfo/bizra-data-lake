// ═══════════════════════════════════════════════════════════════
// DEMA — Mission Lifecycle Store
// Constitutional Generative UI state management
// ═══════════════════════════════════════════════════════════════

import { create } from "zustand";
import type {
  Mission,
  MissionStage,
  MissionType,
  MissionUrgency,
  MissionQuality,
  MissionScope,
  GateEvaluation,
  GateStatus,
  SurfaceType,
  StageTransition,
  Receipt,
  MissionActionPlan,
  GATE_DEFINITIONS as GateDefs,
} from "./types";
import { GATE_DEFINITIONS } from "./types";
// Option A session 2 — real gateway binding for organize missions.
import { submitOrganize, type GatewayOutcome } from "./gateway-client";
import type { AdmissibilityContract } from "@/bindings/AdmissibilityContract";

function genId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function generateHash(): string {
  return "0x" + Array.from({ length: 64 }, () =>
    Math.floor(Math.random() * 16).toString(16)
  ).join("");
}

interface MissionStore {
  // ─── State ──────────────────────────────────────────
  currentStage: MissionStage;
  activeMission: Mission | null;
  missionHistory: Mission[];
  stageTransitions: StageTransition[];
  isProcessing: boolean;
  memoryViewOpen: boolean;

  // ─── Actions ────────────────────────────────────────
  beginMission: (data: {
    intent: string;
    currentState: string;
    desiredState: string;
    missionType: MissionType;
    urgency: MissionUrgency;
    quality: MissionQuality;
    scope: MissionScope;
  }) => void;

  advanceToAdmissibility: () => void;
  evaluateGates: () => Promise<void>;
  /**
   * Option A session 2 — real gateway binding.
   *
   * When missionType === "organize" and the intent string looks like a
   * filesystem path, POST it through /missions/organize and render the
   * actual AdmissibilityContract instead of the random-block simulation.
   * Other mission types keep the existing simulation for now (explicit
   * non-goal of this session per surface-catalog-v1.md §10).
   */
  evaluateGatesReal: () => Promise<void>;
  advanceToAction: (actionPlan: MissionActionPlan) => void;
  confirmAction: () => Promise<void>;
  completeMission: (receipt: Receipt) => void;
  blockMission: (blockedGate: GateEvaluation) => void;
  retreatToIntent: () => void;
  cancelMission: () => void;
  resetToIdle: () => void;
  toggleMemoryView: () => void;

  // ─── Computed ───────────────────────────────────────
  getCurrentSurface: () => SurfaceType;
  getStageProgress: () => { current: number; total: number; label: string };
}

export const useMissionStore = create<MissionStore>((set, get) => ({
  currentStage: "idle" as MissionStage,
  activeMission: null,
  missionHistory: [],
  stageTransitions: [],
  isProcessing: false,
  memoryViewOpen: false,

  beginMission: (data) => {
    const mission: Mission = {
      id: genId("msn"),
      intent: data.intent,
      currentState: data.currentState,
      desiredState: data.desiredState,
      missionType: data.missionType,
      urgency: data.urgency,
      quality: data.quality,
      scope: data.scope,
      stage: "intent",
      gates: GATE_DEFINITIONS.map((g) => ({ ...g })),
      selectedResources: [],
      actionPlan: null,
      sealedReceipt: null,
      createdAt: new Date().toISOString(),
      sealedAt: null,
    };
    set({
      activeMission: mission,
      currentStage: "intent",
      stageTransitions: [],
    });
  },

  advanceToAdmissibility: () => {
    const { activeMission } = get();
    if (!activeMission) return;
    set({
      activeMission: { ...activeMission, stage: "admissibility" },
      currentStage: "admissibility",
      stageTransitions: [
        ...get().stageTransitions,
        {
          from: "intent" as MissionStage,
          to: "admissibility" as MissionStage,
          label: "Mission bounded",
          timestamp: new Date().toISOString(),
        },
      ],
    });
  },

  evaluateGates: async () => {
    set({ isProcessing: true });
    const { activeMission } = get();
    if (!activeMission) return;

    // Simulate gate evaluation with sequential delays
    const updatedGates = [...activeMission.gates];

    for (let i = 0; i < updatedGates.length; i++) {
      updatedGates[i] = { ...updatedGates[i], status: "evaluating" as GateStatus };
      set({
        activeMission: { ...activeMission, gates: [...updatedGates] },
      });
      await new Promise((r) => setTimeout(r, 400 + Math.random() * 300));

      // Simulate: 4 pass, 1 might block (20% chance for drama, but mostly pass)
      const blocked = i === 4 && Math.random() < 0.15;
      updatedGates[i] = {
        ...updatedGates[i],
        status: blocked ? ("blocked" as GateStatus) : ("passed" as GateStatus),
        detail: blocked
          ? "Ihsan score 0.82 — below 0.85 threshold. Review evidence quality."
          : "Invariant satisfied. No violations detected.",
      };
      set({
        activeMission: {
          ...activeMission,
          gates: [...updatedGates],
        },
      });
    }

    await new Promise((r) => setTimeout(r, 300));

    const allPassed = updatedGates.every((g) => g.status === "passed");
    if (allPassed) {
      set({
        activeMission: {
          ...get().activeMission!,
          stage: "action",
        },
        currentStage: "action",
        stageTransitions: [
          ...get().stageTransitions,
          {
            from: "admissibility" as MissionStage,
            to: "action" as MissionStage,
            label: "All gates passed",
            timestamp: new Date().toISOString(),
          },
        ],
        isProcessing: false,
      });
    } else {
      set({
        activeMission: {
          ...get().activeMission!,
          stage: "blocked",
        },
        currentStage: "blocked",
        stageTransitions: [
          ...get().stageTransitions,
          {
            from: "admissibility" as MissionStage,
            to: "blocked" as MissionStage,
            label: "Gate blocked",
            timestamp: new Date().toISOString(),
          },
        ],
        isProcessing: false,
      });
    }
  },

  evaluateGatesReal: async () => {
    const { activeMission } = get();
    if (!activeMission) return;
    set({ isProcessing: true });

    // Transition all gates to evaluating so the UI shows activity during
    // the network call. If the gateway returns quickly, the visible
    // "evaluating" flash is acceptable; if it hangs, the operator at
    // least sees progress rather than a frozen UI.
    const evaluatingGates = activeMission.gates.map((g) => ({
      ...g,
      status: "evaluating" as GateStatus,
    }));
    set({
      activeMission: { ...activeMission, gates: evaluatingGates },
    });

    // Intent is the filesystem path for organize missions. The lawful
    // prerequisite (allowlist registration) is NOT yet surfaced in the
    // Composer — if the path isn't registered, the gateway returns a
    // 403 pre-gate refusal and Gate Ladder renders NotAllowlisted.
    const outcome = await submitOrganize({
      path: activeMission.intent.trim(),
      qualityScore: 0.98,
    });

    const renderFromAdmissibility = (a: AdmissibilityContract) => {
      const byScorer = new Map(a.gateVerdicts.map((gv) => [gv.scorerId, gv]));
      return activeMission.gates.map((g) => {
        const gv = byScorer.get(g.id);
        if (!gv) {
          return {
            ...g,
            status: "pending" as GateStatus,
            detail: "no verdict returned from gateway for this gate",
          };
        }
        const passed = gv.verdict === "Permit";
        const score = gv.score !== null && gv.score !== undefined ? gv.score : null;
        const scoreLabel = score !== null ? ` (score ${score.toFixed(4)})` : "";
        return {
          ...g,
          status: passed ? ("passed" as GateStatus) : ("blocked" as GateStatus),
          detail: passed
            ? `Invariant satisfied${scoreLabel}.`
            : `${gv.reason}${scoreLabel}`,
        };
      });
    };

    if (outcome.kind === "ok") {
      // Happy path: organize executed. Render all gates from the real
      // admissibility contract, advance to action stage.
      const updated = renderFromAdmissibility(outcome.data.admissibility);
      set({
        activeMission: {
          ...activeMission,
          gates: updated,
          stage: "action",
        },
        currentStage: "action",
        stageTransitions: [
          ...get().stageTransitions,
          {
            from: "admissibility" as MissionStage,
            to: "action" as MissionStage,
            label: "All gates passed (gateway sealed receipt)",
            timestamp: new Date().toISOString(),
          },
        ],
        isProcessing: false,
      });
      return;
    }

    if (outcome.kind === "refused") {
      // Gateway returned 400/403/422. Two cases matter for Gate Ladder:
      //   - 403 PATH_NOT_ALLOWLISTED: constitutional pre-gate refusal,
      //     leaves no gate verdicts (admissibility never ran). Mark all
      //     gates blocked with the remediation message.
      //   - 422 ADMISSIBILITY_REJECTED: gateway did run admissibility
      //     and returned real verdicts + the failing RejectedClaim.
      //     Render from the contract exactly like the permit path.
      const preGate = !outcome.error.admissibility;
      const updated = preGate
        ? activeMission.gates.map((g) => ({
            ...g,
            status: "blocked" as GateStatus,
            detail: `${outcome.error.code}: ${outcome.error.message}`,
          }))
        : renderFromAdmissibility(outcome.error.admissibility!);
      set({
        activeMission: {
          ...activeMission,
          gates: updated,
          stage: "blocked",
        },
        currentStage: "blocked",
        stageTransitions: [
          ...get().stageTransitions,
          {
            from: "admissibility" as MissionStage,
            to: "blocked" as MissionStage,
            label: preGate
              ? `Pre-gate refused: ${outcome.error.code}`
              : "Admissibility rejected by gateway",
            timestamp: new Date().toISOString(),
          },
        ],
        isProcessing: false,
      });
      return;
    }

    // outcome.kind === "unreachable" — the gateway is offline or
    // returning non-JSON. Refuse honestly; do NOT simulate a verdict.
    const updated = activeMission.gates.map((g) => ({
      ...g,
      status: "blocked" as GateStatus,
      detail: `cognition gateway unreachable: ${outcome.reason}`,
    }));
    set({
      activeMission: {
        ...activeMission,
        gates: updated,
        stage: "blocked",
      },
      currentStage: "blocked",
      stageTransitions: [
        ...get().stageTransitions,
        {
          from: "admissibility" as MissionStage,
          to: "blocked" as MissionStage,
          label: "Gateway unreachable",
          timestamp: new Date().toISOString(),
        },
      ],
      isProcessing: false,
    });
  },

  advanceToAction: (actionPlan) => {
    const { activeMission } = get();
    if (!activeMission) return;
    set({
      activeMission: {
        ...activeMission,
        stage: "confirmation",
        actionPlan,
      },
      currentStage: "confirmation",
      stageTransitions: [
        ...get().stageTransitions,
        {
          from: "action" as MissionStage,
          to: "confirmation" as MissionStage,
          label: "Action plan ready",
          timestamp: new Date().toISOString(),
        },
      ],
    });
  },

  confirmAction: async () => {
    set({ isProcessing: true });
    const { activeMission } = get();
    if (!activeMission) return;

    // Simulate execution
    await new Promise((r) => setTimeout(r, 1500 + Math.random() * 1000));

    const receipt: Receipt = {
      id: genId("rcp"),
      missionId: activeMission.id,
      type: "completion",
      status: "verified",
      title: `${activeMission.missionType.charAt(0).toUpperCase() + activeMission.missionType.slice(1)} mission executed`,
      description: `Mission "${activeMission.intent}" completed successfully. All constitutional invariants verified.`,
      evidence: JSON.stringify({
        missionId: activeMission.id,
        gates: activeMission.gates.map((g) => ({
          id: g.id,
          status: g.status,
        })),
        stepsExecuted: activeMission.actionPlan?.steps.length ?? 0,
        resourcesUsed: activeMission.selectedResources.length,
        contentHash: generateHash(),
        parentHash: generateHash(),
        manifestId: genId("mft"),
      }),
      issuedAt: new Date().toISOString(),
      verifiedAt: new Date().toISOString(),
      expiresAt: null,
    };

    set({
      activeMission: {
        ...activeMission,
        stage: "receipt",
        sealedReceipt: receipt,
        sealedAt: new Date().toISOString(),
      },
      currentStage: "receipt",
      stageTransitions: [
        ...get().stageTransitions,
        {
          from: "confirmation" as MissionStage,
          to: "receipt" as MissionStage,
          label: "Mission sealed",
          timestamp: new Date().toISOString(),
        },
      ],
      isProcessing: false,
    });
  },

  completeMission: (receipt) => {
    const { activeMission, missionHistory } = get();
    if (!activeMission) return;
    const completed = {
      ...activeMission,
      stage: "receipt" as MissionStage,
      sealedReceipt: receipt,
      sealedAt: new Date().toISOString(),
    };
    set({
      missionHistory: [completed, ...missionHistory],
    });
  },

  blockMission: (blockedGate) => {
    const { activeMission } = get();
    if (!activeMission) return;
    const updatedGates = activeMission.gates.map((g) =>
      g.id === blockedGate.id ? { ...g, status: "blocked" as GateStatus, detail: blockedGate.detail } : g
    );
    set({
      activeMission: {
        ...activeMission,
        stage: "blocked",
        gates: updatedGates,
      },
      currentStage: "blocked",
      isProcessing: false,
    });
  },

  retreatToIntent: () => {
    const { activeMission } = get();
    if (!activeMission) return;
    set({
      activeMission: {
        ...activeMission,
        stage: "intent",
        gates: GATE_DEFINITIONS.map((g) => ({ ...g })),
        actionPlan: null,
        sealedReceipt: null,
      },
      currentStage: "intent",
      stageTransitions: [],
    });
  },

  cancelMission: () => {
    const { activeMission, missionHistory } = get();
    if (activeMission) {
      set({
        missionHistory: [activeMission, ...missionHistory],
      });
    }
    set({
      activeMission: null,
      currentStage: "idle",
      stageTransitions: [],
      isProcessing: false,
    });
  },

  resetToIdle: () => {
    set({
      activeMission: null,
      currentStage: "idle",
      stageTransitions: [],
      isProcessing: false,
    });
  },

  toggleMemoryView: () => {
    set({ memoryViewOpen: !get().memoryViewOpen });
  },

  getCurrentSurface: (): SurfaceType => {
    const { currentStage } = get();
    const map: Record<MissionStage, SurfaceType> = {
      idle: "mission-composer",
      intent: "mission-composer",
      admissibility: "gate-ladder",
      action: "organize-preview",
      confirmation: "organize-preview",
      receipt: "receipt-reveal",
      blocked: "reject-remediation",
    };
    return map[currentStage];
  },

  getStageProgress: () => {
    const { currentStage } = get();
    const stages: MissionStage[] = [
      "intent",
      "admissibility",
      "action",
      "confirmation",
      "receipt",
    ];
    const currentIndex = stages.indexOf(currentStage);
    const labels: Record<string, string> = {
      idle: "Ready",
      intent: "Intent",
      admissibility: "Gates",
      action: "Action",
      confirmation: "Confirm",
      receipt: "Sealed",
      blocked: "Blocked",
    };
    return {
      current: currentIndex >= 0 ? currentIndex + 1 : 0,
      total: stages.length,
      label: labels[currentStage] || currentStage,
    };
  },
}));
