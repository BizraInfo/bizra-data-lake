"use client";

import { useEffect, useState } from "react";
import {
  useModelRoutingSettings,
  useSignatureInfo,
  useSovereignHealth,
} from "@/hooks/use-sovereign-api";

function sameRouting(
  left: Record<string, string>,
  right: Record<string, string>,
): boolean {
  const leftEntries = Object.entries(left);
  const rightEntries = Object.entries(right);
  if (leftEntries.length !== rightEntries.length) {
    return false;
  }
  return leftEntries.every(([key, value]) => right[key] === value);
}

function truncate(value: string, length = 18): string {
  if (!value) {
    return "Unavailable";
  }
  if (value.length <= length) {
    return value;
  }
  return `${value.slice(0, length)}...`;
}

export default function TerminalSettings() {
  const { data: health } = useSovereignHealth();
  const { data: signature } = useSignatureInfo();
  const {
    modelRouting,
    permissionDefaults,
    authState,
    runtimeMode,
    save,
    saving,
    error,
  } = useModelRoutingSettings();
  const [draftRouting, setDraftRouting] = useState<Record<string, string>>({});
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (!dirty) {
      setDraftRouting((previous) => (sameRouting(previous, modelRouting) ? previous : modelRouting));
    }
  }, [modelRouting, dirty]);

  const permissionRows = [
    {
      name: "Filesystem",
      value: permissionDefaults.filesystem.join(", ") || "none",
    },
    {
      name: "Applications",
      value: permissionDefaults.applications.join(", ") || "none",
    },
    {
      name: "Network",
      value: permissionDefaults.network.join(", ") || "none",
    },
    {
      name: "Sensitivity",
      value: permissionDefaults.data_sensitivity,
    },
    {
      name: "Budget",
      value: `$${permissionDefaults.spend_budget_usd.toFixed(2)} / ${permissionDefaults.time_budget_seconds}s`,
    },
    {
      name: "Escalation",
      value: permissionDefaults.escalation,
    },
  ];

  const handleSave = async () => {
    const saved = await save(draftRouting);
    if (saved) {
      setDirty(false);
      setDraftRouting(saved);
    }
  };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Settings</h2>
          <p className="text-xs text-slate-500">
            Trust boundaries, identity, and routing preferences
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_1fr] gap-4">
        <section className="border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
            Node Identity
          </h3>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-500">Node ID</span>
              <span className="text-slate-200 font-mono">
                {truncate(signature.node_id)}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-500">Public key</span>
              <span className="text-slate-200 font-mono">
                {truncate(signature.public_key, 24)}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-500">Signing</span>
              <span className="text-slate-300">
                {signature.algorithms.signing}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-500">Audit chain</span>
              <span className="text-slate-300">
                {signature.algorithms.audit_chain}
              </span>
            </div>
          </div>
        </section>

        <section className="border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
            Environment
          </h3>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Mode</span>
              <span className="text-slate-300 uppercase">{runtimeMode}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Auth state</span>
              <span className="text-slate-300">{authState}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Runtime</span>
              <span className="text-slate-300">{health.status || "unknown"}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Tick interval</span>
              <span className="text-slate-300">{health.tick_interval_s ?? 60}s</span>
            </div>
          </div>
        </section>
      </div>

      <section className="border border-slate-700/50 rounded-lg p-4 mt-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Model Routing
          </h3>
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-3 py-1.5 rounded border border-teal-700/50 bg-teal-950/20 text-[10px] text-teal-300 disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save Routing"}
          </button>
        </div>
        <div className="space-y-2">
          {Object.entries(draftRouting).map(([role, model]) => (
            <label
              key={role}
              className="flex items-center gap-3 border border-slate-800 rounded-lg p-3"
            >
              <span className="w-28 text-[10px] uppercase tracking-wider text-slate-500">
                {role}
              </span>
              <input
                value={model}
                onChange={(event) => {
                  setDirty(true);
                  setDraftRouting((previous) => ({
                    ...previous,
                    [role]: event.target.value,
                  }));
                }}
                className="flex-1 bg-slate-900/70 border border-slate-700 rounded px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-teal-500"
              />
            </label>
          ))}
        </div>
        {error && <p className="text-xs text-red-400 mt-3">{error}</p>}
      </section>

      <section className="border border-slate-700/50 rounded-lg p-4 mt-4">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          Permission Defaults
        </h3>
        <div className="space-y-2">
          {permissionRows.map((permission) => (
            <div
              key={permission.name}
              className="flex items-center justify-between gap-3 border-b border-slate-800/40 pb-2 last:border-0 last:pb-0"
            >
              <div className="text-sm text-slate-200">{permission.name}</div>
              <div className="text-[10px] text-slate-400 text-right max-w-[60%]">
                {permission.value}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
