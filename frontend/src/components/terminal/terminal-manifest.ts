"use client";

export type TerminalViewId =
  | "dashboard"
  | "mission"
  | "timeline"
  | "memory"
  | "skills"
  | "network"
  | "settings"
  | "cockpit";

export interface TerminalViewMeta {
  id: TerminalViewId;
  label: string;
  shortcut: string;
  emoji: string;
  accentHex: string;
  description: string;
}

export interface TerminalAgentDef {
  id: string;
  name: string;
  call: string;
  role: string;
  domain: string;
  emoji: string;
  temperature: number;
  status: "active" | "idle" | "standby";
  team: "PAT" | "SAT";
  raidRole: string;
}

export interface TerminalTierDef {
  name: "Novice" | "Adept" | "Expert" | "Master";
  min_actions: number;
  min_ihsan: number;
  unlocks: string[];
  color: string;
}

export interface LifecycleStageDef {
  name: "Seedling" | "Sprout" | "Sapling" | "Branch" | "Canopy" | "Catalyst";
  threshold: number;
  emoji: string;
}

export const TERMINAL_THEME = {
  background: "#030810",
  backgroundElevated: "#08121F",
  backgroundSoft: "#111827",
  gold: "#C9A962",
  goldBright: "#E8D5A3",
  goldDeep: "#8B7340",
  text: "#F8F6F1",
  textMuted: "rgba(248,246,241,0.72)",
  textDim: "rgba(248,246,241,0.45)",
  line: "rgba(255,255,255,0.08)",
  success: "#22C55E",
  info: "#3B82F6",
  alert: "#F59E0B",
  danger: "#EF4444",
} as const;

export const TERMINAL_VIEW_META: TerminalViewMeta[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    shortcut: "1",
    emoji: "◈",
    accentHex: "#C9A962",
    description: "Node readiness and constitutional pulse.",
  },
  {
    id: "mission",
    label: "Mission",
    shortcut: "2",
    emoji: "◆",
    accentHex: "#22C55E",
    description: "Submit once. Approve once. Receive one proof.",
  },
  {
    id: "timeline",
    label: "Timeline",
    shortcut: "3",
    emoji: "⋯",
    accentHex: "#3B82F6",
    description: "Event-native narrative of action, receipt, and tick.",
  },
  {
    id: "memory",
    label: "Memory",
    shortcut: "4",
    emoji: "◉",
    accentHex: "#06B6D4",
    description: "Personal continuity across missions, receipts, and reflexes.",
  },
  {
    id: "skills",
    label: "Agents / Skills",
    shortcut: "5",
    emoji: "✦",
    accentHex: "#A855F7",
    description: "PAT-7, SAT-5, tiering, and reflex inventory.",
  },
  {
    id: "network",
    label: "Network / Forest",
    shortcut: "6",
    emoji: "⟡",
    accentHex: "#EAB308",
    description: "Node value, diffusion readiness, and growth horizon.",
  },
  {
    id: "settings",
    label: "Settings / Sovereignty",
    shortcut: "7",
    emoji: "☗",
    accentHex: "#F97316",
    description: "Identity, routing, and trust defaults.",
  },
  {
    id: "cockpit",
    label: "Cockpit",
    shortcut: "8",
    emoji: "⬡",
    accentHex: "#1D9E75",
    description: "Constitutional pipeline: Intent → Evidence.",
  },
];

export const PAT_AGENT_MANIFEST: TerminalAgentDef[] = [
  {
    id: "P1",
    name: "Atlas",
    call: "ATLAS",
    role: "Planner",
    domain: "Strategic Planning",
    emoji: "🗺️",
    temperature: 0.2,
    status: "active",
    team: "PAT",
    raidRole: "Strategist",
  },
  {
    id: "P2",
    name: "Oracle",
    call: "ORACLE",
    role: "Researcher",
    domain: "Knowledge Discovery",
    emoji: "🔭",
    temperature: 0.7,
    status: "active",
    team: "PAT",
    raidRole: "Scout",
  },
  {
    id: "P3",
    name: "Forge",
    call: "FORGE",
    role: "Coder",
    domain: "Implementation",
    emoji: "⚒️",
    temperature: 0.3,
    status: "active",
    team: "PAT",
    raidRole: "Builder",
  },
  {
    id: "P4",
    name: "Judge",
    call: "JUDGE",
    role: "Evaluator",
    domain: "Quality Gates",
    emoji: "⚖️",
    temperature: 0.1,
    status: "active",
    team: "PAT",
    raidRole: "Verifier",
  },
  {
    id: "P5",
    name: "Crown",
    call: "CROWN",
    role: "Ethicist",
    domain: "Constitutional Guard",
    emoji: "👑",
    temperature: 0.1,
    status: "active",
    team: "PAT",
    raidRole: "Guardian",
  },
  {
    id: "P6",
    name: "Herald",
    call: "HERALD",
    role: "Publisher",
    domain: "Delivery",
    emoji: "📢",
    temperature: 0.4,
    status: "idle",
    team: "PAT",
    raidRole: "Messenger",
  },
  {
    id: "P7",
    name: "DEMA",
    call: "NEXUS",
    role: "Integrator",
    domain: "Orchestration",
    emoji: "💜",
    temperature: 0.2,
    status: "active",
    team: "PAT",
    raidRole: "Raid Leader",
  },
];

export const SAT_AGENT_MANIFEST: TerminalAgentDef[] = [
  {
    id: "S1",
    name: "Sentinel",
    call: "SENTINEL",
    role: "Security",
    domain: "Threat Detection",
    emoji: "🛡️",
    temperature: 0.05,
    status: "active",
    team: "SAT",
    raidRole: "Shield",
  },
  {
    id: "S2",
    name: "Oracle",
    call: "ORACLE-S",
    role: "Forest Health",
    domain: "Ihsan Scoring",
    emoji: "🌳",
    temperature: 0.3,
    status: "active",
    team: "SAT",
    raidRole: "Oracle",
  },
  {
    id: "S3",
    name: "Ledger",
    call: "LEDGER",
    role: "Economy",
    domain: "Evidence Chain",
    emoji: "📊",
    temperature: 0.05,
    status: "active",
    team: "SAT",
    raidRole: "Record Keeper",
  },
  {
    id: "S4",
    name: "Conductor",
    call: "CONDUCTOR",
    role: "Capacity",
    domain: "S1 / S2 Boundary",
    emoji: "🎵",
    temperature: 0.2,
    status: "active",
    team: "SAT",
    raidRole: "Traffic Control",
  },
  {
    id: "S5",
    name: "Ambassador",
    call: "AMBASSADOR",
    role: "Federation",
    domain: "Network Sync",
    emoji: "🤝",
    temperature: 0.4,
    status: "standby",
    team: "SAT",
    raidRole: "Diplomat",
  },
];

export const TERMINAL_TIER_DEFS: TerminalTierDef[] = [
  {
    name: "Novice",
    min_actions: 0,
    min_ihsan: 0,
    unlocks: ["Read files", "Clipboard", "Basic queries"],
    color: "text-slate-400",
  },
  {
    name: "Adept",
    min_actions: 10,
    min_ihsan: 0.85,
    unlocks: ["Write files", "Local scripts", "Browser automation"],
    color: "text-teal-400",
  },
  {
    name: "Expert",
    min_actions: 100,
    min_ihsan: 0.9,
    unlocks: ["Network access", "API calls", "Multi-app orchestration"],
    color: "text-amber-400",
  },
  {
    name: "Master",
    min_actions: 1000,
    min_ihsan: 0.95,
    unlocks: ["Unsandboxed processes", "Marketplace publish", "Mentor others"],
    color: "text-purple-400",
  },
];

export const TERMINAL_LIFECYCLE_STAGES: LifecycleStageDef[] = [
  { name: "Seedling", threshold: 0, emoji: "🌱" },
  { name: "Sprout", threshold: 0.15, emoji: "🌿" },
  { name: "Sapling", threshold: 0.3, emoji: "🌲" },
  { name: "Branch", threshold: 0.5, emoji: "🌳" },
  { name: "Canopy", threshold: 0.75, emoji: "🏔️" },
  { name: "Catalyst", threshold: 0.95, emoji: "⭐" },
];
