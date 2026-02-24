// ============================================================
// useNode — React hook bridging to bizra-node
// ============================================================
// Two modes:
//   TAURI:   invoke() → Rust → Node.execute()
//   BROWSER: simulated bridge (demo / development)
//
// The hook auto-detects which mode based on window.__TAURI__
// React components never know the difference.
// ============================================================

import { useState, useEffect, useCallback, useRef } from "react";
import {
  countQueuedActions,
  enqueueAction,
  listQueuedActions,
  removeQueuedAction,
} from "./offline/queue";

// Is Tauri available?
const isTauri = () => typeof window !== "undefined" && window.__TAURI_INTERNALS__;

// ============================================================
// TAURI BRIDGE — calls Rust backend via invoke
// ============================================================
const createTauriBridge = () => {
  // Dynamic import to avoid errors in non-Tauri environments
  let invoke = null;

  const ensureInvoke = async () => {
    if (!invoke) {
      const api = await import("@tauri-apps/api/core");
      invoke = api.invoke;
    }
    return invoke;
  };

  return {
    mode: "tauri",

    send: async (verb, args = {}) => {
      const inv = await ensureInvoke();
      const sanitize = (v) => String(v ?? "").replace(/\t/g, " ").replace(/\n/g, " ");

      switch (verb) {
        case "RECEIVE":
          return inv("node_receive", {
            content: args.content || "",
            timestamp: args.timestamp || Date.now(),
          });
        case "TEACH":
          return inv("node_teach", {
            kind: args.kind || "fact",
            content: args.content || "",
            confidence: args.confidence || 9000,
            timestamp: args.timestamp || Date.now(),
          });
        case "SYNTHESIZE":
          return inv("node_synthesize", {
            timestamp: args.timestamp || Date.now(),
          });
        case "QUERY":
          return inv("node_query", { key: args.key || "" });
        case "PROFILE":
          return inv("node_profile");
        case "KNOWS_ME":
          return inv("node_knows_me");
        case "HEALTH":
          return inv("node_health");
        case "IHSAN":
          return inv("node_ihsan", { score: args.score || 9900 });
        case "START_SESSION":
          return inv("node_start_session", {
            timestamp: args.timestamp || Date.now(),
          });
        case "END_SESSION":
          return inv("node_end_session", {
            timestamp: args.timestamp || Date.now(),
          });
        case "PING":
          return inv("node_ping");
        case "VERSION":
          return inv("node_version");
        case "PLAN_ACTION":
          return inv("node_execute", {
            command: `PLAN_ACTION\t${sanitize(args.payload_json || "{}")}`,
          });
        case "RUN_ACTION":
          return inv("node_execute", {
            command: `RUN_ACTION\t${sanitize(args.plan_id || "")}\t${sanitize(args.payload_json || "{}")}`,
          });
        case "ACTION_STATUS":
          return inv("node_execute", {
            command: `ACTION_STATUS\t${sanitize(args.action_id || "")}`,
          });
        case "ACTION_HISTORY":
          return inv("node_execute", {
            command: `ACTION_HISTORY\t${sanitize(args.limit || 20)}\t${sanitize(args.cursor || "")}`,
          });
        case "GET_CONTEXT":
          return inv("node_execute", {
            command: `GET_CONTEXT\t${sanitize(args.plaintext_titles || "false")}`,
          });
        // 8 HDA skills — routed through node_execute
        case "HDA_OPEN_APP":
        case "HDA_SWITCH_WINDOW":
        case "HDA_TYPE_TEXT":
        case "HDA_CLICK_ELEMENT":
        case "HDA_SCREENSHOT":
        case "HDA_READ_CLIPBOARD":
        case "HDA_FILE_OPEN":
        case "HDA_BROWSER_NAVIGATE":
          return inv("node_execute", {
            command: `${verb}\t${sanitize(JSON.stringify(args))}`,
          });
        // SAP v0 protocol verbs
        case "SAP_MEET_OPEN":
          return inv("node_execute", {
            command: `SAP_MEET_OPEN\t${sanitize(args.profile || "sap-ads-retail-v0")}\t${sanitize(args.initiator_role || "visitor")}\t${Date.now()}`,
          });
        case "SAP_MESSAGE":
          return inv("node_execute", {
            command: `SAP_MESSAGE\t${sanitize(args.session_id || "")}\t${sanitize(args.content || "")}\t${Date.now()}`,
          });
        case "SAP_DISCLOSURE":
          return inv("node_execute", {
            command: `SAP_DISCLOSURE\t${sanitize(args.session_id || "")}`,
          });
        case "SAP_CONSENT_REQUEST":
          return inv("node_execute", {
            command: `SAP_CONSENT_REQUEST\t${sanitize(args.session_id || "")}\t${sanitize(JSON.stringify(args.scopes || []))}`,
          });
        case "SAP_CONSENT_REVOKE":
          return inv("node_execute", {
            command: `SAP_CONSENT_REVOKE\t${sanitize(args.session_id || "")}\t${sanitize(args.receipt_id || "")}`,
          });
        case "SAP_SESSION_CLOSE":
          return inv("node_execute", {
            command: `SAP_SESSION_CLOSE\t${sanitize(args.session_id || "")}\t${Date.now()}`,
          });
        default:
          return inv("node_execute", { command: verb });
      }
    },
  };
};

// ============================================================
// BROWSER BRIDGE — simulated for development / demo
// ============================================================
const createBrowserBridge = () => {
  const state = {
    knowsMe: 0.0,
    ihsan: 9900,
    messages: 0,
    fragments: 0,
    insights: 0,
    traits: [],
    vetoes: 0,
  };

  const extractFragments = (text) => {
    let count = 0;
    const lower = text.toLowerCase();
    const patterns = [
      [/i (?:prefer|like|love|enjoy)/, "preference", /(?:prefer|like|love|enjoy)\s+(.+?)(?:\.|,|$)/i],
      [/i(?:'m| am) (?:a|an) /, "identity", /(?:i'm|i am)\s+(?:a|an)\s+(.+?)(?:\.|,|$)/i],
      [/my goal/, "goal", /my goal\s+(?:is\s+)?(?:to\s+)?(.+?)(?:\.|$)/i],
      [/i (?:specialize|work with|work at)/, "expertise", /(?:specialize in|work with|work at)\s+(.+?)(?:\.|$)/i],
      [/i live in/, "location", /live in\s+(.+?)(?:\.|$)/i],
    ];

    for (const [detector, label, extractor] of patterns) {
      if (detector.test(lower)) {
        count++;
        const match = text.match(extractor);
        if (match) {
          state.traits = state.traits.filter((t) => t.label !== label);
          state.traits.push({
            label,
            value: match[1].trim().slice(0, 40),
            confidence: label === "location" ? 9500 : label === "expertise" ? 9200 : 8500,
          });
        }
      }
    }
    return count;
  };

  const classify = (text) => {
    if (/\b(code|function|implement|debug)\b/.test(text)) return ["Code", 3];
    if (/\b(what|why|how|explain)\b/.test(text)) return ["Question", 3];
    if (/\b(create|write|generate|design)\b/.test(text)) return ["Create", 3];
    if (/\b(analyze|compare|evaluate)\b/.test(text)) return ["Analyze", 3];
    if (/\b(plan|strategy|roadmap)\b/.test(text)) return ["Plan", 3];
    return ["Chat", 2];
  };

  return {
    mode: "browser",

    send: async (verb, args = {}) => {
      // Simulate async like real Tauri calls
      await new Promise((r) => setTimeout(r, 50 + Math.random() * 100));

      switch (verb) {
        case "RECEIVE": {
          const frags = extractFragments(args.content || "");
          state.messages++;
          state.fragments += frags;
          state.knowsMe = Math.min(1.0, state.knowsMe + frags * 0.008 + 0.002);
          const [intent, agents] = classify((args.content || "").toLowerCase());

          const hasTraits = state.traits.length > 0 && state.knowsMe > 0.05;
          const context = state.traits.map((t) => t.value).join(", ");
          const content = hasTraits
            ? `Building on what I know — ${context} — ${
                { Question: "let me research this.", Code: "I'll implement this your way.", Create: "I'll craft something for your vision.", Analyze: "analyzing with your context.", Plan: "mapping this strategically." }[intent] || "tell me more."
              }`
            : { Question: "Let me look into that.", Code: "I'll work on it.", Create: "I'll start crafting that." }[intent] || "I hear you. Tell me more.";

          return {
            ok: true,
            fields: {
              content,
              confidence: String(hasTraits ? 0.85 + state.knowsMe * 0.1 : 0.75),
              agents_consulted: String(agents),
              fragments_extracted: String(frags),
              guardian_approved: "true",
              knows_me: String(state.knowsMe.toFixed(4)),
              intent,
            },
          };
        }
        case "TEACH": {
          state.fragments++;
          state.knowsMe = Math.min(1.0, state.knowsMe + 0.012);
          state.traits = state.traits.filter((t) => t.label !== args.kind);
          state.traits.push({ label: args.kind, value: (args.content || "").slice(0, 40), confidence: args.confidence || 9000 });
          return { ok: true, fields: { taught: args.content, kind: args.kind, confidence: String(args.confidence || 9000) } };
        }
        case "SYNTHESIZE": {
          const n = Math.min(3, Math.floor(state.fragments / 3));
          state.insights += n;
          state.knowsMe = Math.min(1.0, state.knowsMe + n * 0.015);
          return { ok: true, fields: { insights_generated: String(n), knows_me: state.knowsMe.toFixed(4) } };
        }
        case "HEALTH":
          return {
            ok: true,
            fields: {
              state: "Ready",
              ihsan: String(state.ihsan),
              knows_me: state.knowsMe.toFixed(4),
              agents_registered: "7",
              messages: String(state.messages),
              fragments: String(state.fragments),
              insights: String(state.insights),
              vetoes: String(state.vetoes),
            },
          };
        case "KNOWS_ME":
          return { ok: true, fields: { score: state.knowsMe.toFixed(4) } };
        case "PROFILE": {
          const traits = state.traits.map((t) => `${t.label}:${t.value}:${t.confidence}`).join("|");
          return { ok: true, fields: { traits, count: String(state.traits.length) } };
        }
        case "PING":
          return { ok: true, fields: { pong: "true" } };
        case "VERSION":
          return { ok: true, fields: { node: "bizra-node", version: "0.1.0", protocol: "1.0" } };
        case "PLAN_ACTION": {
          const planId = `pln_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
          return {
            ok: true,
            fields: {
              planned: "true",
              plan_id: planId,
              steps: "1",
              created_at: String(Date.now()),
              method: JSON.parse(args.payload_json || "{}").method || "unknown",
              permit_status: "APPROVED",
              budget_remaining: "29",
            },
          };
        }
        case "RUN_ACTION": {
          const actionId = `act_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
          const preHash = Array.from({ length: 32 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join("");
          const postHash = Array.from({ length: 32 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join("");
          const outcomeHash = Array.from({ length: 32 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join("");
          return {
            ok: true,
            fields: {
              ran: "true",
              action_id: actionId,
              plan_id: args.plan_id || "",
              status: "completed",
              message: "Action executed with proof",
              pre_hash: preHash,
              post_hash: postHash,
              outcome_hash: outcomeHash,
              state_changed: "true",
              outcome_confirmed: "true",
              confidence: "0.95",
              verification_timestamp: String(Date.now()),
            },
          };
        }
        case "ACTION_STATUS":
          return {
            ok: true,
            fields: {
              found: "true",
              action_id: args.action_id || "act_demo",
              status: "completed",
              message: "ok",
              outcome_confirmed: "true",
              confidence: "0.95",
            },
          };
        case "ACTION_HISTORY":
          return {
            ok: true,
            fields: {
              count: "0",
              next_cursor: "",
              rows: "",
            },
          };
        case "GET_CONTEXT":
          return {
            ok: true,
            fields: {
              schema_version: "2.0",
              source: "browser_simulated",
              privacy_mode: args.plaintext_titles ? "plaintext" : "hashed",
              timestamp: String(Date.now()),
              foreground: JSON.stringify({ title: "Demo App", title_hashed: false }),
              process_count: "5",
              clipboard_hash: Array.from({ length: 32 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join(""),
              clipboard_length: "42",
            },
          };
        // 8 HDA skills — simulated with receipt-shaped responses
        case "HDA_OPEN_APP":
        case "HDA_SWITCH_WINDOW":
        case "HDA_TYPE_TEXT":
        case "HDA_CLICK_ELEMENT":
        case "HDA_SCREENSHOT":
        case "HDA_READ_CLIPBOARD":
        case "HDA_FILE_OPEN":
        case "HDA_BROWSER_NAVIGATE": {
          const hdaId = `hda_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
          const hdaPre = Array.from({ length: 32 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join("");
          const hdaPost = Array.from({ length: 32 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join("");
          return {
            ok: true,
            fields: {
              action_id: hdaId,
              method: verb,
              status: "completed",
              pre_hash: hdaPre,
              post_hash: hdaPost,
              outcome_hash: Array.from({ length: 32 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, "0")).join(""),
              state_changed: "true",
              outcome_confirmed: "true",
              confidence: "0.95",
            },
          };
        }

        // SAP v0 protocol simulation
        case "SAP_MEET_OPEN": {
          const sessionId = `sap_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
          return {
            ok: true,
            fields: {
              session_id: sessionId,
              profile: args.profile || "sap-ads-retail-v0",
              disclosure: JSON.stringify({
                claims: [
                  "This agent is compiled from real multi-platform conversations",
                  "BIZRA is in Alpha-100 stage",
                ],
                uncertainty: [
                  "Compilation score is approximate",
                  "Alpha software — features may change",
                ],
                source_refs: ["specs/sap-v0/01-core-primitives.md"],
                compliance_assertions: [{ standard: "SAP_v0", status: "conformant", evidence: "22/22 tests" }],
              }),
              ihsan_score: "0.97",
              agent_card: JSON.stringify({
                agent_id: "node0-user-zero",
                owner_node_id: "node0",
                role: "sovereign_personal",
                version: "0.1.0",
                policy_hash: "504145f781412a4103249f78f46d61609eb1d02f",
                capabilities: ["chat", "teach", "synthesize", "disclose"],
                compilation: {
                  genesis_version: "GENESIS",
                  ihsan_threshold: 0.95,
                  compiled_reflex_count: 81,
                  compilation_coverage: 0.92,
                },
              }),
            },
          };
        }
        case "SAP_MESSAGE": {
          state.messages++;
          const frags = extractFragments(args.content || "");
          state.fragments += frags;
          state.knowsMe = Math.min(1.0, state.knowsMe + frags * 0.008 + 0.002);
          return {
            ok: true,
            fields: {
              session_id: args.session_id || "",
              content: state.knowsMe > 0.05
                ? "I understand your interest. Let me share what I know about BIZRA's architecture."
                : "Welcome to BIZRA. I'm a sovereign agent — compiled, not prompted. Ask me anything.",
              disclosure: JSON.stringify({
                claims: ["Response generated from compiled reflexes"],
                uncertainty: ["I may not have information on features still in development"],
              }),
              ihsan_score: String(0.95 + Math.random() * 0.04),
              receipt_hash: Array.from({ length: 16 }, () => Math.floor(Math.random() * 16).toString(16)).join(""),
            },
          };
        }
        case "SAP_DISCLOSURE":
          return {
            ok: true,
            fields: {
              session_id: args.session_id || "",
              disclosure: JSON.stringify({
                claims: [
                  "This agent is compiled from 7000+ real conversations across 8 platforms",
                  "All responses pass a constitutional Ihsan gate (threshold >= 0.95)",
                  "SAP v0 conformance: 22/22 tests passing",
                ],
                uncertainty: [
                  "Compilation score is 0.92 — some gaps may exist",
                  "Alpha software — production readiness in progress",
                ],
                source_refs: [
                  "specs/sap-v0/01-core-primitives.md",
                  "tests/conformance/sap_v0/",
                  "schemas/sap/v0/disclosure.schema.json",
                ],
              }),
            },
          };
        case "SAP_CONSENT_REQUEST":
          return {
            ok: true,
            fields: {
              session_id: args.session_id || "",
              status: "pending",
              scopes: JSON.stringify(args.scopes || []),
              message: "Consent requested. No data shared until explicitly granted.",
            },
          };
        case "SAP_CONSENT_REVOKE":
          return {
            ok: true,
            fields: {
              session_id: args.session_id || "",
              revoked: "true",
              message: "Consent revoked. All associated data processing stopped.",
            },
          };
        case "SAP_SESSION_CLOSE":
          return {
            ok: true,
            fields: {
              session_id: args.session_id || "",
              closed: "true",
              final_receipt_hash: Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join(""),
              message: "Session closed. You can revoke any granted consent at any time.",
            },
          };

        default:
          return { ok: true, fields: {} };
      }
    },

    getState: () => ({ ...state }),
  };
};

// ============================================================
// THE HOOK
// ============================================================
export function useNode() {
  const bridgeRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState("detecting");
  const [health, setHealth] = useState(null);
  const [lastSeenTs, setLastSeenTs] = useState(null);
  const [queuedActions, setQueuedActions] = useState(0);
  const [nodeReachable, setNodeReachable] = useState(false);

  const flushQueue = useCallback(async () => {
    if (!bridgeRef.current || !nodeReachable) return;
    const rows = await listQueuedActions(50);
    for (const row of rows) {
      try {
        await bridgeRef.current.send(row.command, row.payload);
        await removeQueuedAction(row.id);
      } catch {
        // Stop flush on first failure to avoid tight loops.
        break;
      }
    }
    setQueuedActions(await countQueuedActions());
  }, [nodeReachable]);

  // Initialize bridge
  useEffect(() => {
    const bridge = isTauri() ? createTauriBridge() : createBrowserBridge();
    bridgeRef.current = bridge;
    setMode(bridge.mode);

    // Boot sequence
    const boot = async () => {
      try {
        await bridge.send("PING");
        setConnected(true);
        setNodeReachable(true);
        setLastSeenTs(Date.now());

        // Get initial health
        const h = await bridge.send("HEALTH");
        if (h.ok) setHealth(h.fields);
        setQueuedActions(await countQueuedActions());
        await flushQueue();
      } catch (err) {
        console.error("[useNode] Boot failed:", err);
        setConnected(false);
        setNodeReachable(false);
        setQueuedActions(await countQueuedActions());
      }
    };

    boot();
  }, [flushQueue]);

  // Stable send function
  const send = useCallback(async (verb, args = {}) => {
    if (!bridgeRef.current) return { ok: false, error: "Not initialized" };
    try {
      const result = await bridgeRef.current.send(verb, args);
      setConnected(true);
      setNodeReachable(true);
      setLastSeenTs(Date.now());
      return result;
    } catch (err) {
      setConnected(false);
      setNodeReachable(false);
      const queueEligible = verb === "PLAN_ACTION" || verb === "RUN_ACTION";
      if (queueEligible) {
        await enqueueAction({ command: verb, payload: args });
        setQueuedActions(await countQueuedActions());
        return {
          ok: false,
          queued: true,
          error: "Node unreachable, action queued for reconnect",
        };
      }
      return { ok: false, error: "Node unreachable" };
    }
  }, []);

  // Convenience methods
  const receive = useCallback(
    (content) => send("RECEIVE", { content, timestamp: Date.now() }),
    [send]
  );

  const teach = useCallback(
    (kind, content, confidence = 9000) =>
      send("TEACH", { kind, content, confidence, timestamp: Date.now() }),
    [send]
  );

  const synthesize = useCallback(() => send("SYNTHESIZE", { timestamp: Date.now() }), [send]);

  const knowsMe = useCallback(() => send("KNOWS_ME"), [send]);

  const refreshHealth = useCallback(async () => {
    const h = await send("HEALTH");
    if (h.ok) setHealth(h.fields);
    return h;
  }, [send]);

  // SAP v0 convenience methods
  const sapMeetOpen = useCallback(
    (profile = "sap-ads-retail-v0", initiatorRole = "visitor") =>
      send("SAP_MEET_OPEN", { profile, initiator_role: initiatorRole }),
    [send]
  );

  const sapMessage = useCallback(
    (sessionId, content) =>
      send("SAP_MESSAGE", { session_id: sessionId, content }),
    [send]
  );

  const sapDisclosure = useCallback(
    (sessionId) => send("SAP_DISCLOSURE", { session_id: sessionId }),
    [send]
  );

  const sapSessionClose = useCallback(
    (sessionId) => send("SAP_SESSION_CLOSE", { session_id: sessionId }),
    [send]
  );

  // HDA action round-trip: PLAN -> RUN -> verify receipt
  const planAction = useCallback(
    (method, params = {}) =>
      send("PLAN_ACTION", {
        payload_json: JSON.stringify({ method, params }),
      }),
    [send]
  );

  const runAction = useCallback(
    (planId, params = {}) =>
      send("RUN_ACTION", {
        plan_id: planId,
        payload_json: JSON.stringify(params),
      }),
    [send]
  );

  const actionStatus = useCallback(
    (actionId) => send("ACTION_STATUS", { action_id: actionId }),
    [send]
  );

  const actionHistory = useCallback(
    (limit = 20, cursor = "") =>
      send("ACTION_HISTORY", { limit, cursor }),
    [send]
  );

  const getContext = useCallback(
    (plaintextTitles = false) =>
      send("GET_CONTEXT", { plaintext_titles: plaintextTitles }),
    [send]
  );

  // Lightweight heartbeat + queue flush loop.
  useEffect(() => {
    const id = setInterval(async () => {
      if (!bridgeRef.current) return;
      try {
        await bridgeRef.current.send("PING");
        setConnected(true);
        setNodeReachable(true);
        setLastSeenTs(Date.now());
        await flushQueue();
      } catch {
        setConnected(false);
        setNodeReachable(false);
      }
    }, 5000);
    return () => clearInterval(id);
  }, [flushQueue]);

  return {
    connected,
    mode,
    health,
    nodeReachable,
    queuedActions,
    lastSeenTs,
    send,
    receive,
    teach,
    synthesize,
    knowsMe,
    refreshHealth,
    // SAP v0
    sapMeetOpen,
    sapMessage,
    sapDisclosure,
    sapSessionClose,
    // HDA action round-trip (Task 1.3)
    planAction,
    runAction,
    actionStatus,
    actionHistory,
    getContext,
  };
}
