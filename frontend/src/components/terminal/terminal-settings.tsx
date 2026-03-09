"use client";

import { useState, useCallback } from "react";
import { useSovereignHealth } from "@/hooks/use-sovereign-api";

// ─── Types ──────────────────────────────────────────────────────

interface ModelRoute {
  id: string;
  name: string;
  provider: string;
  size: string;
  status: "active" | "available" | "downloading";
  selected: boolean;
}

interface PermissionPolicy {
  name: string;
  description: string;
  enabled: boolean;
  tier_required: string;
}

// ─── Demo Data ──────────────────────────────────────────────────

const DEMO_IDENTITY = {
  node_id: "0x4A2F7B3E9C5D1A8F",
  public_key_prefix: "ed25519:4A2F7B3E...9C5D1A8F",
  created_at: "2026-03-08T10:00:00Z",
  genesis_block: "0xA3B5C7D9E1F2...48 chars",
  evidence_chain_length: 47,
  total_seed: 24.67,
  total_bloom: 12.40,
};

const DEMO_MODELS: ModelRoute[] = [
  { id: "phi3", name: "Phi-3 Mini", provider: "Ollama", size: "3.8B Q4_K", status: "active", selected: true },
  { id: "llama8b", name: "Llama 3.1 8B", provider: "Ollama", size: "8B Q4_K", status: "available", selected: false },
  { id: "tinyllama", name: "TinyLlama", provider: "Ollama", size: "1.1B Q4_K", status: "available", selected: false },
  { id: "qwen14b", name: "Qwen 2.5 14B", provider: "LM Studio", size: "14B Q4_K", status: "available", selected: false },
];

const DEMO_PERMISSIONS: PermissionPolicy[] = [
  { name: "File Read", description: "Read files from local filesystem", enabled: true, tier_required: "Novice" },
  { name: "File Write", description: "Write files to local filesystem", enabled: true, tier_required: "Adept" },
  { name: "Network Access", description: "Make HTTP requests to external APIs", enabled: false, tier_required: "Expert" },
  { name: "Process Execution", description: "Execute system processes unsandboxed", enabled: false, tier_required: "Master" },
  { name: "Marketplace Publish", description: "Publish reflexes to forest pool", enabled: false, tier_required: "Expert" },
];

// ─── Helpers ────────────────────────────────────────────────────

function truncateHash(hash: string, len: number = 16): string {
  if (hash.length <= len) return hash;
  return hash.slice(0, len) + "...";
}

// ─── Sub-Components ─────────────────────────────────────────────

function IdentityCard() {
  const [showFull, setShowFull] = useState(false);

  return (
    <div className="border border-slate-700/50 rounded-lg p-4 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
        Node Identity
      </h3>

      <div className="space-y-2">
        {/* §10.7: Shows node ID + public key prefix */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Node ID</span>
          <button
            onClick={() => setShowFull(!showFull)}
            className="text-xs text-teal-400 font-mono hover:text-teal-300 transition-colors"
          >
            {showFull ? DEMO_IDENTITY.node_id : truncateHash(DEMO_IDENTITY.node_id)}
          </button>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Public Key</span>
          <span className="text-xs text-slate-300 font-mono">
            {DEMO_IDENTITY.public_key_prefix}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Genesis</span>
          <span className="text-xs text-slate-300">
            {new Date(DEMO_IDENTITY.created_at).toLocaleDateString()}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Evidence Chain</span>
          <span className="text-xs text-slate-300 font-mono">
            {DEMO_IDENTITY.evidence_chain_length} blocks
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Genesis Block</span>
          <span className="text-[10px] text-slate-500 font-mono">
            {truncateHash(DEMO_IDENTITY.genesis_block, 20)}
          </span>
        </div>

        <div className="border-t border-slate-800 pt-2 mt-2 flex items-center justify-between">
          <span className="text-xs text-slate-500">Lifetime</span>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-amber-400 font-bold">{DEMO_IDENTITY.total_seed.toFixed(1)} SEED</span>
            <span className="text-purple-400 font-bold">{DEMO_IDENTITY.total_bloom.toFixed(1)} BLOOM</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ModelRouting({ models, onSelect }: { models: ModelRoute[]; onSelect: (id: string) => void }) {
  return (
    <div className="border border-slate-700/50 rounded-lg p-4 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
        Model Routing
      </h3>

      <div className="space-y-1.5">
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => onSelect(m.id)}
            className={`w-full flex items-center justify-between p-2.5 rounded-lg border transition-all ${
              m.selected
                ? "border-teal-500/50 bg-teal-950/30"
                : "border-slate-700/30 bg-slate-800/20 hover:border-slate-600"
            }`}
          >
            <div className="flex items-center gap-2.5">
              {m.selected && (
                <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
              )}
              {!m.selected && (
                <span className="w-2 h-2 rounded-full bg-slate-700" />
              )}
              <div className="text-left">
                <div className="text-sm text-slate-200 font-medium">{m.name}</div>
                <div className="text-[10px] text-slate-500">
                  {m.provider} · {m.size}
                </div>
              </div>
            </div>
            <span
              className={`text-[10px] px-1.5 py-0.5 rounded ${
                m.status === "active"
                  ? "bg-emerald-900/50 text-emerald-300"
                  : m.status === "downloading"
                    ? "bg-amber-900/50 text-amber-300"
                    : "bg-slate-800 text-slate-500"
              }`}
            >
              {m.status}
            </span>
          </button>
        ))}
      </div>

      <p className="text-[10px] text-slate-600 mt-2">
        Select a model to route inference. Larger models need more RAM but produce higher-quality reasoning.
      </p>
    </div>
  );
}

function EnvironmentInfo() {
  const { data: health } = useSovereignHealth();
  const env = health?.env || "development";
  const isProduction = env === "production";

  return (
    <div className="border border-slate-700/50 rounded-lg p-4 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
        Environment
      </h3>

      <div className="space-y-2">
        {/* §10.7: Shows dev/prod mode indicator */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Mode</span>
          <span
            className={`text-xs font-bold px-2 py-0.5 rounded ${
              isProduction
                ? "bg-emerald-900/50 text-emerald-300"
                : "bg-amber-900/50 text-amber-300"
            }`}
          >
            {isProduction ? "PRODUCTION" : "DEVELOPMENT"}
          </span>
        </div>

        {/* §10.7: Shows auth state */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Auth</span>
          <span
            className={`text-xs ${
              isProduction
                ? "text-emerald-400"
                : "text-amber-400"
            }`}
          >
            {isProduction ? "🔐 Authenticated" : "🔓 Anonymous (dev)"}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Heartbeat</span>
          <span className="text-xs text-emerald-400">
            ♥ Every 60s
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Version</span>
          <span className="text-xs text-slate-300 font-mono">v3.0.0-GENESIS</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">Ihsān Gate</span>
          <span className="text-xs text-slate-300">≥ 0.95 (production)</span>
        </div>
      </div>
    </div>
  );
}

function PermissionDefaults({
  permissions,
  onToggle,
}: {
  permissions: PermissionPolicy[];
  onToggle: (name: string) => void;
}) {
  return (
    <div className="border border-slate-700/50 rounded-lg p-4 mb-3">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
        Permission Defaults
      </h3>

      <div className="space-y-1">
        {permissions.map((p) => (
          <div
            key={p.name}
            className="flex items-center justify-between py-2 border-b border-slate-800/30 last:border-0"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-200">{p.name}</span>
                <span className="text-[10px] text-slate-600 bg-slate-800 px-1.5 rounded">
                  {p.tier_required}+
                </span>
              </div>
              <p className="text-[10px] text-slate-500">{p.description}</p>
            </div>
            <button
              onClick={() => onToggle(p.name)}
              className={`ml-3 w-9 h-5 rounded-full transition-colors flex-shrink-0 ${
                p.enabled ? "bg-teal-600" : "bg-slate-700"
              }`}
            >
              <div
                className={`w-4 h-4 rounded-full bg-white transition-transform ${
                  p.enabled ? "translate-x-4.5 ml-[18px]" : "translate-x-0.5 ml-[2px]"
                }`}
              />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────

export default function TerminalSettings() {
  const [models, setModels] = useState(DEMO_MODELS);
  const [permissions, setPermissions] = useState(DEMO_PERMISSIONS);

  // §10.7: Editable: model routing preferences
  const handleModelSelect = useCallback((id: string) => {
    setModels((prev) =>
      prev.map((m) => ({
        ...m,
        selected: m.id === id,
        status: m.id === id ? "active" : m.status === "active" ? "available" : m.status,
      }))
    );
  }, []);

  const handlePermissionToggle = useCallback((name: string) => {
    setPermissions((prev) =>
      prev.map((p) =>
        p.name === name ? { ...p, enabled: !p.enabled } : p
      )
    );
  }, []);

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Settings</h2>
          <p className="text-xs text-slate-500">Sovereign node configuration</p>
        </div>
      </div>

      {/* §10.7: Shows node ID + public key prefix */}
      <IdentityCard />

      {/* §10.7: Shows current model routing + editable */}
      <ModelRouting models={models} onSelect={handleModelSelect} />

      {/* §10.7: Shows dev/prod mode, auth state */}
      <EnvironmentInfo />

      {/* §10.7: Shows permission policy defaults */}
      <PermissionDefaults permissions={permissions} onToggle={handlePermissionToggle} />

      {/* Sovereignty guarantee */}
      <div className="text-center mt-4 py-3 border border-emerald-900/30 rounded-lg bg-emerald-950/10">
        <p className="text-[10px] text-emerald-500">
          🔒 Your keys. Your data. Your compute. Sovereign by design.
        </p>
        <p className="text-[9px] text-slate-700 mt-1">
          Ed25519 identity · BLAKE2b evidence chain · Zero cloud dependency
        </p>
      </div>
    </div>
  );
}
