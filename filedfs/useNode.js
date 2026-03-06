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

        // Get initial health
        const h = await bridge.send("HEALTH");
        if (h.ok) setHealth(h.fields);
      } catch (err) {
        console.error("[useNode] Boot failed:", err);
      }
    };

    boot();
  }, []);

  // Stable send function
  const send = useCallback(async (verb, args = {}) => {
    if (!bridgeRef.current) return { ok: false, error: "Not initialized" };
    const result = await bridgeRef.current.send(verb, args);
    return result;
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

  return {
    connected,
    mode,
    health,
    send,
    receive,
    teach,
    synthesize,
    knowsMe,
    refreshHealth,
  };
}
