import { useEffect, useRef, useState } from "react";
import type { ProofSurfacePayload } from "@/lib/dema-proof-surface";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

export interface PermissionEnvelope {
  filesystem: string[];
  applications: string[];
  network: string[];
  data_sensitivity: string;
  spend_budget_usd: number;
  time_budget_seconds: number;
  escalation: string;
  audit_verbosity: string;
}

export interface ChannelResult {
  channel: string;
  success: boolean;
  duration_ms: number;
}

export interface MissionReceipt {
  mission_id: string;
  receipt_id: string;
  evidence_receipt_id?: string | null;
  status: "COMPLETE" | "PARTIAL" | "FAILED" | "BLOCKED";
  synthesis: string;
  ihsan_score: number;
  snr_score: number;
  duration_ms: number;
  channels_executed: ChannelResult[];
  execution_path: string;
  wallet_delta: {
    seed: number;
    bloom: number;
  };
  reflex_delta: {
    compiled: boolean;
    near_compile: boolean;
    compile_count: number;
    threshold: number;
  };
  memory_delta: {
    episodic: number;
    semantic: number;
    procedural: number;
  };
  hash_chain_ref: string;
  action_count: number;
  reflex_pattern: string;
  reflex_latency_ms: number;
  comparison_s2_avg_ms: number;
  reasoning_proof?: {
    mode: string;
    vrg_root: string;
    verified: boolean;
    receipt_id: string;
    status: string;
    payload_digest: string;
    branch_count: number;
    surviving_branches: number;
    detail: string;
  } | null;
  proof_surface?: ProofSurfacePayload | null;
}

export interface CriticalAcknowledgmentReceipt {
  acknowledgement_id: string;
  receipt_id: string;
  status: "ACKNOWLEDGED";
  hash_chain_ref: string;
  acknowledged_event_hash: string;
  acknowledged_topic: string;
  mission_id: string;
  timestamp: string;
  synthesis: string;
}

export interface SovereignHealth {
  status: string;
  tier?: string;
  version?: string;
  uptime_s?: number;
  snr_score?: number;
  ihsan_score?: number;
  env?: string;
  live_status?: string;
  running?: boolean;
  gini?: number;
  asabiyyah?: number;
  last_tick_timestamp?: string;
  tick_interval_s?: number;
  wallet_snapshot?: {
    seed: number;
    bloom: number;
  };
  last_mission_summary?: string;
  auth_state?: string;
  runtime_mode?: string;
  model_routing?: Record<string, string>;
  permission_defaults?: PermissionEnvelope;
  critical_subsystems?: Record<string, string>;
  seed_engine?: Record<string, unknown>;
  node_value?: Record<string, unknown>;
}

export interface SeedPotential {
  sovereignty_score: number;
  tier: string;
  tier_progress: number;
  episodes_total: number;
  episodes_qualified: number;
  qualification_rate: number;
  reward_ema: number;
  streak: number;
  compiled: boolean;
  converged: boolean;
  chain_valid: boolean;
  potential_unlocked: number;
  potential_remaining: number;
  weakest_dimension: string | null;
  growth_velocity: number;
  last_receipt_hash: string;
}

export interface ConstitutionalStatus {
  status: string;
  wallets: number;
  events: number;
  reflexes: number;
  pending_receipts: number;
  pending_proposals: number;
  gini: number;
  asabiyyah: number;
  last_tick_timestamp: string;
  tick_interval_s: number;
}

export interface SeedEpisode {
  index: number;
  timestamp: string;
  snr: number;
  ihsan: number;
  reward: number;
  qualified: boolean;
  tier: string;
  sovereignty_score: number;
  receipt_hash: string;
}

export interface TerminalBriefing {
  time_since_last_mission_s: number;
  active_project: string;
  last_mission_summary: string;
  near_compile_patterns: string[];
  quality_trend: string;
  next_action_suggestion: string;
  wallet_snapshot: {
    seed: number;
    bloom: number;
  };
}

export interface MemoryStats {
  episodic_count: number;
  semantic_count: number;
  procedural_count: number;
  total_entries: number;
  db_size_mb: number;
}

export interface MemoryProfileMission {
  mission_id: string;
  description: string;
  status: "COMPLETE" | "PARTIAL" | "FAILED" | "BLOCKED";
  ihsan_score: number;
  seed_earned: number;
  timestamp: string;
  receipt_hash: string;
}

export interface MemoryProfilePattern {
  name: string;
  count: number;
  threshold: number;
  avg_ihsan: number;
}

export interface MemoryProfileCompiledReflex {
  name: string;
  avg_ihsan: number;
  execution_count: number;
  avg_latency_ms: number;
  compiled_at: string;
  last_hit_at: string;
}

export interface MemoryProfileProject {
  name: string;
  last_activity: string;
  mission_count: number;
}

export interface MemoryProfile {
  privacy_note: string;
  briefing: TerminalBriefing;
  semantic_profile: {
    preferred_domains: string[];
    active_hours: string;
    vocabulary_signature: string;
    work_window: string;
  };
  missions: MemoryProfileMission[];
  active_projects: MemoryProfileProject[];
  work_streak: number;
  near_compile_patterns: MemoryProfilePattern[];
  compiled_reflex_summary: MemoryProfileCompiledReflex[];
  stats: MemoryStats;
}

export interface TerminalStateSnapshot {
  state: string;
  execution_path: string;
  mission_id: string;
  restart_required?: boolean;
}

export interface Node0Readiness {
  status: "green" | "yellow" | "red";
  generated_at: string;
  product_shell: {
    available: boolean;
    version: string;
    default_view: string;
  };
  proof_surface: {
    available: boolean;
    source: string;
  };
  runtime: {
    live: boolean;
    state: string;
    ihsan_score: number;
    snr_score: number;
  };
  boot_service: {
    status: "booted" | "not_booted" | "unavailable" | "error";
    booted: boolean;
    node_id: string;
    total_breaths: number;
    chain_hash: string;
    error: string;
  };
  memory_import: {
    available: boolean;
    status: "ready" | "unavailable";
    mode: string;
    imported_records: number;
    requires_consent: boolean;
    source: string;
    truth_label: string;
  };
  voice_input: {
    available: boolean;
    status: "browser_required" | "ready" | "unsupported";
    mode: string;
    requires_user_gesture: boolean;
    stores_audio: boolean;
    auto_submit: boolean;
    truth_label: string;
  };
  desktop_browser_action: {
    available: boolean;
    status: "preview_only" | "unavailable";
    mode: string;
    allowed_actions: string[];
    requires_user_confirmation: boolean;
    server_executes: boolean;
    truth_label: string;
  };
  local_action_executor: {
    available: boolean;
    status: "browser_client_ready" | "unavailable";
    mode: string;
    allowed_actions: string[];
    requires_user_confirmation: boolean;
    server_executes: boolean;
    records_receipts: boolean;
    truth_label: string;
  };
  always_on_daemon: {
    available: boolean;
    status: "running" | "stopped" | "stale_pid" | "unavailable";
    mode: string;
    pid: number;
    lock_path: string;
    root: string;
    last_tick_at: string;
    last_receipt_id: string;
    requires_operator_confirmation: boolean;
    server_executes: boolean;
    no_public_network_listener: boolean;
    writes_under_state_dir: boolean;
    status_command: string;
    start_command: string;
    truth_label: string;
  };
  spearpoint: {
    status: "pass" | "fail" | "unknown";
    artifact_status: string;
    run_id: string;
    mode: string;
    timestamp_utc: string;
    targets_completed: number;
    official_submission: boolean;
    classification: string;
    reason?: string;
  };
  next_action: string;
}

export interface Node0ActionIntentResult {
  action_id: string;
  accepted: boolean;
  status: string;
  action_type: "open_url" | "copy_text";
  label: string;
  target: string;
  target_preview: string;
  target_hash: string;
  execution_mode: string;
  handoff_method: string;
  server_executed: boolean;
  requires_user_confirmation: boolean;
  truth_label: string;
  source_label: string;
  next_action: string;
}

export interface Node0LocalActionReceipt {
  receipt_id: string;
  action_id: string;
  recorded: boolean;
  status: "executed" | "blocked" | "failed";
  action_type: "open_url" | "copy_text";
  execution_channel: "browser_client";
  server_executed: boolean;
  target_preview: string;
  target_hash: string;
  recorded_at: string;
  truth_label: string;
  source_label: string;
  next_action: string;
  error?: string;
}

export interface TerminalStreamEvent {
  topic: string;
  severity: "info" | "notice" | "warning" | "critical";
  mission_id: string;
  receipt_id: string;
  event_hash: string;
  prev_hash: string;
  timestamp: string;
  payload: Record<string, unknown>;
  source?: string;
}

export interface TokenBalance {
  seed: number;
  bloom: number;
  staked: number;
}

export interface NodeValue {
  composite: number;
  [key: string]: unknown;
}

export interface NodeLifecycle {
  current_stage: string;
  next_stage?: string;
  sovereignty_score?: number;
  [key: string]: unknown;
}

export interface NetworkEffect {
  nodes: number;
  skills_available: number;
  compute_tflops: number;
  latency_factor: number;
  intelligence_density: number;
  cost_per_node: number;
}

export interface NetworkMilestone {
  nodes: number;
  skills: number;
  tflops: number;
  latency_factor: number;
}

export interface SignatureInfo {
  node_id: string;
  public_key: string;
  algorithms: {
    signing: string;
    hashing: string;
    canonicalization: string;
    audit_chain: string;
  };
}

export interface FetchResult<T> {
  data: T;
  error: string | null;
  loading: boolean;
}

function apiUrl(path: string): string {
  const base = API_BASE.endsWith("/") ? API_BASE.slice(0, -1) : API_BASE;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${base}${normalized}`;
}

const SESSION_STORAGE_KEY = "bizra-terminal-session-id";

function getSessionId(): string {
  if (typeof window === "undefined") {
    return "terminal-server";
  }
  const existing = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const created =
    window.crypto?.randomUUID?.() ??
    `terminal-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  window.sessionStorage.setItem(SESSION_STORAGE_KEY, created);
  return created;
}

function websocketUrl(path: string): string {
  const httpUrl = apiUrl(path);
  const absolute = /^https?:\/\//.test(httpUrl)
    ? httpUrl
    : `${window.location.origin}${httpUrl.startsWith("/") ? httpUrl : `/${httpUrl}`}`;
  return absolute.replace(/^http/i, "ws");
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("X-Session-ID", getSessionId());
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(apiUrl(path), {
    ...init,
    headers,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function useFetch<T>(
  path: string,
  fallback: T,
  options?: {
    intervalMs?: number;
    transform?: (payload: unknown) => T;
    resetOnError?: boolean;
  },
): FetchResult<T> {
  const [data, setData] = useState<T>(fallback);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const intervalMs = options?.intervalMs ?? 30000;
  const resetOnError = options?.resetOnError ?? false;

  // Inline object/function literals at call sites (e.g. useChainLatest's
  // fallback) get a new identity each render. Capturing them in refs keeps
  // the polling effect stable instead of re-firing on every render, which
  // would compound under setInterval mocks that don't honor clearInterval.
  const fallbackRef = useRef(fallback);
  fallbackRef.current = fallback;
  const transformRef = useRef(options?.transform);
  transformRef.current = options?.transform;

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      try {
        const payload = await requestJson<unknown>(path);
        if (!mounted) {
          return;
        }
        const transform = transformRef.current;
        setData(transform ? transform(payload) : (payload as T));
        setError(null);
      } catch (err) {
        if (!mounted) {
          return;
        }
        if (resetOnError) {
          setData(fallbackRef.current);
        }
        setError(err instanceof Error ? err.message : "Request failed");
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    void load();
    const timer = window.setInterval(() => {
      void load();
    }, intervalMs);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [intervalMs, path, resetOnError]);

  return { data, error, loading };
}

function asObject(payload: unknown): Record<string, unknown> {
  return payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function normalizePermissionEnvelope(payload: unknown): PermissionEnvelope {
  const data = asObject(payload);
  return {
    filesystem: asStringArray(data.filesystem),
    applications: asStringArray(data.applications),
    network: asStringArray(data.network),
    data_sensitivity: asString(data.data_sensitivity, "standard"),
    spend_budget_usd: asNumber(data.spend_budget_usd, 0),
    time_budget_seconds: asNumber(data.time_budget_seconds, 900),
    escalation: asString(data.escalation, "ask-on-boundary-cross"),
    audit_verbosity: asString(data.audit_verbosity, "standard"),
  };
}

function normalizeSovereignHealth(payload: unknown): SovereignHealth {
  const data = asObject(payload);
  const wallet = asObject(data.wallet_snapshot);
  return {
    status: asString(data.status, "unknown"),
    tier: asString(data.tier, "terminal"),
    version: asString(data.version),
    uptime_s: asNumber(data.uptime_s),
    snr_score: asNumber(data.snr_score),
    ihsan_score: asNumber(data.ihsan_score),
    env: asString(data.env, "development"),
    live_status: asString(data.live_status, "OFFLINE"),
    running: Boolean(data.running),
    gini: asNumber(data.gini),
    asabiyyah: asNumber(data.asabiyyah),
    last_tick_timestamp: asString(data.last_tick_timestamp),
    tick_interval_s: asNumber(data.tick_interval_s, 60),
    wallet_snapshot: {
      seed: asNumber(wallet.seed),
      bloom: asNumber(wallet.bloom),
    },
    last_mission_summary: asString(data.last_mission_summary),
    auth_state: asString(data.auth_state, "unavailable"),
    runtime_mode: asString(data.runtime_mode, "development"),
    model_routing: asObject(data.model_routing) as Record<string, string>,
    permission_defaults: normalizePermissionEnvelope(data.permission_defaults),
    critical_subsystems: asObject(data.critical_subsystems) as Record<string, string>,
    seed_engine: asObject(data.seed_engine),
    node_value: asObject(data.node_value),
  };
}

function normalizeTokenBalance(payload: unknown): TokenBalance {
  const data = asObject(payload);
  const balances = asObject(data.balances);
  const seed = asObject(balances.SEED);
  const bloom = asObject(balances.BLOOM);

  return {
    seed: asNumber(seed.balance),
    bloom: asNumber(bloom.balance),
    staked: asNumber(seed.staked) + asNumber(bloom.staked),
  };
}

function normalizeConstitutionalStatus(payload: unknown): ConstitutionalStatus {
  const data = asObject(payload);
  return {
    status: asString(data.status, "unknown"),
    wallets: asNumber(data.wallets),
    events: asNumber(data.events),
    reflexes: asNumber(data.reflexes),
    pending_receipts: asNumber(data.pending_receipts),
    pending_proposals: asNumber(data.pending_proposals),
    gini: asNumber(data.gini, asNumber(data.network_gini)),
    asabiyyah: asNumber(data.asabiyyah, asNumber(data.network_asabiyyah)),
    last_tick_timestamp: asString(data.last_tick_timestamp),
    tick_interval_s: asNumber(data.tick_interval_s, 60),
  };
}

function normalizeSeedEpisodes(payload: unknown): SeedEpisode[] {
  const data = asObject(payload);
  const episodes = Array.isArray(data.episodes) ? data.episodes : payload;

  if (!Array.isArray(episodes)) {
    return [];
  }

  return episodes.map((episode) => {
    const item = asObject(episode);
    return {
      index: asNumber(item.index),
      timestamp: asString(item.timestamp),
      snr: asNumber(item.snr),
      ihsan: asNumber(item.ihsan),
      reward: asNumber(item.reward),
      qualified: Boolean(item.qualified),
      tier: asString(item.tier),
      sovereignty_score: asNumber(item.sovereignty_score),
      receipt_hash: asString(item.receipt_hash),
    };
  });
}

function normalizeTerminalBriefing(payload: unknown): TerminalBriefing {
  const data = asObject(payload);
  const wallet = asObject(data.wallet_snapshot);

  return {
    time_since_last_mission_s: asNumber(data.time_since_last_mission_s),
    active_project: asString(data.active_project),
    last_mission_summary: asString(data.last_mission_summary),
    near_compile_patterns: asStringArray(data.near_compile_patterns),
    quality_trend: asString(data.quality_trend, "stable"),
    next_action_suggestion: asString(data.next_action_suggestion),
    wallet_snapshot: {
      seed: asNumber(wallet.seed),
      bloom: asNumber(wallet.bloom),
    },
  };
}

function normalizeMemoryStats(payload: unknown): MemoryStats {
  const data = asObject(payload);
  return {
    episodic_count: asNumber(data.episodic_count),
    semantic_count: asNumber(data.semantic_count),
    procedural_count: asNumber(data.procedural_count),
    total_entries: asNumber(data.total_entries),
    db_size_mb: asNumber(data.db_size_mb),
  };
}

function normalizeNode0Readiness(payload: unknown): Node0Readiness {
  const data = asObject(payload);
  const productShell = asObject(data.product_shell);
  const proofSurface = asObject(data.proof_surface);
  const runtime = asObject(data.runtime);
  const bootService = asObject(data.boot_service);
  const memoryImport = asObject(data.memory_import);
  const voiceInput = asObject(data.voice_input);
  const desktopBrowserAction = asObject(data.desktop_browser_action);
  const localActionExecutor = asObject(data.local_action_executor);
  const alwaysOnDaemon = asObject(data.always_on_daemon);
  const spearpoint = asObject(data.spearpoint);
  const status = asString(data.status, "red");
  const bootStatus = asString(bootService.status, "unavailable");
  const memoryImportStatus = asString(memoryImport.status, "unavailable");
  const voiceInputStatus = asString(voiceInput.status, "unsupported");
  const desktopBrowserActionStatus = asString(
    desktopBrowserAction.status,
    "unavailable",
  );
  const localActionExecutorStatus = asString(
    localActionExecutor.status,
    "unavailable",
  );
  const alwaysOnDaemonStatus = asString(alwaysOnDaemon.status, "unavailable");
  const spearpointStatus = asString(spearpoint.status, "unknown");

  return {
    status: ["green", "yellow", "red"].includes(status)
      ? (status as Node0Readiness["status"])
      : "red",
    generated_at: asString(data.generated_at),
    product_shell: {
      available: Boolean(productShell.available),
      version: asString(productShell.version),
      default_view: asString(productShell.default_view, "node0"),
    },
    proof_surface: {
      available: Boolean(proofSurface.available),
      source: asString(proofSurface.source, "mission_receipt"),
    },
    runtime: {
      live: Boolean(runtime.live),
      state: asString(runtime.state, "unknown"),
      ihsan_score: asNumber(runtime.ihsan_score),
      snr_score: asNumber(runtime.snr_score),
    },
    boot_service: {
      status: ["booted", "not_booted", "unavailable", "error"].includes(bootStatus)
        ? (bootStatus as Node0Readiness["boot_service"]["status"])
        : "unavailable",
      booted: Boolean(bootService.booted),
      node_id: asString(bootService.node_id),
      total_breaths: asNumber(bootService.total_breaths),
      chain_hash: asString(bootService.chain_hash),
      error: asString(bootService.error),
    },
    memory_import: {
      available: Boolean(memoryImport.available),
      status: ["ready", "unavailable"].includes(memoryImportStatus)
        ? (memoryImportStatus as Node0Readiness["memory_import"]["status"])
        : "unavailable",
      mode: asString(memoryImport.mode, "single_user_provided_record"),
      imported_records: asNumber(memoryImport.imported_records),
      requires_consent: memoryImport.requires_consent !== false,
      source: asString(memoryImport.source, "agent_db"),
      truth_label: asString(memoryImport.truth_label, "[ENFORCEMENT: WIRED]"),
    },
    voice_input: {
      available: Boolean(voiceInput.available),
      status: ["browser_required", "ready", "unsupported"].includes(voiceInputStatus)
        ? (voiceInputStatus as Node0Readiness["voice_input"]["status"])
        : "unsupported",
      mode: asString(voiceInput.mode, "browser_speech_recognition"),
      requires_user_gesture: voiceInput.requires_user_gesture !== false,
      stores_audio: Boolean(voiceInput.stores_audio),
      auto_submit: Boolean(voiceInput.auto_submit),
      truth_label: asString(voiceInput.truth_label, "[ENFORCEMENT: WIRED]"),
    },
    desktop_browser_action: {
      available: Boolean(desktopBrowserAction.available),
      status: ["preview_only", "unavailable"].includes(desktopBrowserActionStatus)
        ? (desktopBrowserActionStatus as Node0Readiness["desktop_browser_action"]["status"])
        : "unavailable",
      mode: asString(desktopBrowserAction.mode, "client_handoff_only"),
      allowed_actions: asStringArray(desktopBrowserAction.allowed_actions),
      requires_user_confirmation:
        desktopBrowserAction.requires_user_confirmation !== false,
      server_executes: Boolean(desktopBrowserAction.server_executes),
      truth_label: asString(
        desktopBrowserAction.truth_label,
        "[ENFORCEMENT: WIRED]",
      ),
    },
    local_action_executor: {
      available: Boolean(localActionExecutor.available),
      status: ["browser_client_ready", "unavailable"].includes(localActionExecutorStatus)
        ? (localActionExecutorStatus as Node0Readiness["local_action_executor"]["status"])
        : "unavailable",
      mode: asString(localActionExecutor.mode, "explicit_user_gesture"),
      allowed_actions: asStringArray(localActionExecutor.allowed_actions),
      requires_user_confirmation:
        localActionExecutor.requires_user_confirmation !== false,
      server_executes: Boolean(localActionExecutor.server_executes),
      records_receipts: Boolean(localActionExecutor.records_receipts),
      truth_label: asString(
        localActionExecutor.truth_label,
        "[ENFORCEMENT: WIRED]",
      ),
    },
    always_on_daemon: {
      available: Boolean(alwaysOnDaemon.available),
      status: ["running", "stopped", "stale_pid", "unavailable"].includes(
        alwaysOnDaemonStatus,
      )
        ? (alwaysOnDaemonStatus as Node0Readiness["always_on_daemon"]["status"])
        : "unavailable",
      mode: asString(alwaysOnDaemon.mode, "local_ambient_loop"),
      pid: asNumber(alwaysOnDaemon.pid),
      lock_path: asString(alwaysOnDaemon.lock_path),
      root: asString(alwaysOnDaemon.root),
      last_tick_at: asString(alwaysOnDaemon.last_tick_at),
      last_receipt_id: asString(alwaysOnDaemon.last_receipt_id),
      requires_operator_confirmation:
        alwaysOnDaemon.requires_operator_confirmation !== false,
      server_executes: Boolean(alwaysOnDaemon.server_executes),
      no_public_network_listener:
        alwaysOnDaemon.no_public_network_listener !== false,
      writes_under_state_dir: alwaysOnDaemon.writes_under_state_dir !== false,
      status_command: asString(
        alwaysOnDaemon.status_command,
        "python scripts/dema/dema_service.py status --root sovereign_state/dema",
      ),
      start_command: asString(
        alwaysOnDaemon.start_command,
        "python scripts/dema/dema_daemon.py --loop --interval-seconds 60 --root sovereign_state/dema",
      ),
      truth_label: asString(alwaysOnDaemon.truth_label, "[ENFORCEMENT: WIRED]"),
    },
    spearpoint: {
      status: ["pass", "fail", "unknown"].includes(spearpointStatus)
        ? (spearpointStatus as Node0Readiness["spearpoint"]["status"])
        : "unknown",
      artifact_status: asString(spearpoint.artifact_status, "missing"),
      run_id: asString(spearpoint.run_id),
      mode: asString(spearpoint.mode),
      timestamp_utc: asString(spearpoint.timestamp_utc),
      targets_completed: asNumber(spearpoint.targets_completed),
      official_submission: Boolean(spearpoint.official_submission),
      classification: asString(spearpoint.classification, "internal_strict_harness"),
      reason: asString(spearpoint.reason),
    },
    next_action: asString(data.next_action, "start or repair Dema service"),
  };
}

function normalizeMemoryProfileMission(payload: unknown): MemoryProfileMission {
  const data = asObject(payload);
  return {
    mission_id: asString(data.mission_id),
    description: asString(data.description),
    status: asString(data.status, "PARTIAL") as MemoryProfileMission["status"],
    ihsan_score: asNumber(data.ihsan_score),
    seed_earned: asNumber(data.seed_earned),
    timestamp: asString(data.timestamp),
    receipt_hash: asString(data.receipt_hash),
  };
}

function normalizeMemoryProfilePattern(payload: unknown): MemoryProfilePattern {
  const data = asObject(payload);
  return {
    name: asString(data.name),
    count: asNumber(data.count),
    threshold: asNumber(data.threshold, 3),
    avg_ihsan: asNumber(data.avg_ihsan),
  };
}

function normalizeMemoryProfileProject(payload: unknown): MemoryProfileProject {
  const data = asObject(payload);
  return {
    name: asString(data.name),
    last_activity: asString(data.last_activity),
    mission_count: asNumber(data.mission_count),
  };
}

function normalizeMemoryProfileCompiledReflex(
  payload: unknown,
): MemoryProfileCompiledReflex {
  const data = asObject(payload);
  return {
    name: asString(data.name),
    avg_ihsan: asNumber(data.avg_ihsan),
    execution_count: asNumber(data.execution_count),
    avg_latency_ms: asNumber(data.avg_latency_ms),
    compiled_at: asString(data.compiled_at),
    last_hit_at: asString(data.last_hit_at),
  };
}

function normalizeMemoryProfile(payload: unknown): MemoryProfile {
  const data = asObject(payload);
  const semantic = asObject(data.semantic_profile);
  const missions = Array.isArray(data.missions) ? data.missions : [];
  const projects = Array.isArray(data.active_projects) ? data.active_projects : [];
  const nearCompilePatterns = Array.isArray(data.near_compile_patterns)
    ? data.near_compile_patterns
    : [];
  const compiledReflexSummary = Array.isArray(data.compiled_reflex_summary)
    ? data.compiled_reflex_summary
    : [];

  return {
    privacy_note: asString(data.privacy_note, "All data is local"),
    briefing: normalizeTerminalBriefing(data.briefing),
    semantic_profile: {
      preferred_domains: asStringArray(semantic.preferred_domains),
      active_hours: asString(semantic.active_hours),
      vocabulary_signature: asString(semantic.vocabulary_signature),
      work_window: asString(semantic.work_window),
    },
    missions: missions.map(normalizeMemoryProfileMission),
    active_projects: projects.map(normalizeMemoryProfileProject),
    work_streak: asNumber(data.work_streak),
    near_compile_patterns: nearCompilePatterns.map(normalizeMemoryProfilePattern),
    compiled_reflex_summary: compiledReflexSummary.map(
      normalizeMemoryProfileCompiledReflex,
    ),
    stats: normalizeMemoryStats(data.stats),
  };
}

function normalizeNetworkMilestones(payload: unknown): NetworkMilestone[] {
  const data = asObject(payload);
  const milestones = Array.isArray(data.milestones) ? data.milestones : [];
  return milestones.map((item) => {
    const milestone = asObject(item);
    return {
      nodes: asNumber(milestone.nodes),
      skills: asNumber(milestone.skills),
      tflops: asNumber(milestone.tflops),
      latency_factor: asNumber(milestone.latency_factor),
    };
  });
}

function defaultPermissionEnvelope(): PermissionEnvelope {
  return {
    filesystem: ["workspace/**"],
    applications: ["terminal", "editor", "browser"],
    network: [],
    data_sensitivity: "standard",
    spend_budget_usd: 3,
    time_budget_seconds: 900,
    escalation: "ask-on-boundary-cross",
    audit_verbosity: "standard",
  };
}

export function useSovereignHealth() {
  return useFetch<SovereignHealth>(
    "/v1/health",
    {
      status: "unknown",
      tier: "terminal",
      version: "",
      uptime_s: 0,
      snr_score: 0,
      ihsan_score: 0,
      live_status: "OFFLINE",
      running: false,
      gini: 0,
      asabiyyah: 0,
      last_tick_timestamp: "",
      tick_interval_s: 60,
      wallet_snapshot: { seed: 0, bloom: 0 },
      last_mission_summary: "",
      auth_state: "unavailable",
      runtime_mode: "development",
      model_routing: {},
      permission_defaults: defaultPermissionEnvelope(),
    },
    { transform: normalizeSovereignHealth },
  );
}

export function useSeedPotential() {
  return useFetch<SeedPotential>(
    "/v1/seed/potential",
    {
      sovereignty_score: 0,
      tier: "SEED",
      tier_progress: 0,
      episodes_total: 0,
      episodes_qualified: 0,
      qualification_rate: 0,
      reward_ema: 0,
      streak: 0,
      compiled: false,
      converged: false,
      chain_valid: false,
      potential_unlocked: 0,
      potential_remaining: 1,
      weakest_dimension: null,
      growth_velocity: 0,
      last_receipt_hash: "",
    },
  );
}

export function useTokenBalance() {
  return useFetch<TokenBalance>(
    "/v1/token/balance",
    {
      seed: 0,
      bloom: 0,
      staked: 0,
    },
    { transform: normalizeTokenBalance },
  );
}

export function useConstitutionalStatus() {
  return useFetch<ConstitutionalStatus>(
    "/v1/constitutional/status",
    {
      status: "unknown",
      wallets: 0,
      events: 0,
      reflexes: 0,
      pending_receipts: 0,
      pending_proposals: 0,
      gini: 0,
      asabiyyah: 0,
      last_tick_timestamp: "",
      tick_interval_s: 60,
    },
    { transform: normalizeConstitutionalStatus },
  );
}

export function useNetworkEffect(nodes = 1000) {
  return useFetch<NetworkEffect>(
    `/v1/network/effect?nodes=${nodes}`,
    {
      nodes,
      skills_available: 0,
      compute_tflops: 0,
      latency_factor: 0,
      intelligence_density: 0,
      cost_per_node: 0,
    },
  );
}

export function useNetworkMilestones() {
  return useFetch<NetworkMilestone[]>(
    "/v1/network/milestones",
    [],
    { transform: normalizeNetworkMilestones },
  );
}

export function useNodeLifecycle() {
  return useFetch<NodeLifecycle>(
    "/v1/node/lifecycle",
    {
      current_stage: "Seedling",
    },
  );
}

export function useNodeValue() {
  return useFetch<NodeValue>(
    "/v1/node/value",
    {
      composite: 0,
    },
  );
}

export function useSeedEpisodes() {
  return useFetch<SeedEpisode[]>(
    "/v1/seed/episodes",
    [],
    { transform: normalizeSeedEpisodes },
  );
}

export function useTerminalBriefing() {
  return useFetch<TerminalBriefing>(
    "/v1/terminal/briefing",
    {
      time_since_last_mission_s: 0,
      active_project: "",
      last_mission_summary: "",
      near_compile_patterns: [],
      quality_trend: "stable",
      next_action_suggestion: "",
      wallet_snapshot: {
        seed: 0,
        bloom: 0,
      },
    },
    { transform: normalizeTerminalBriefing },
  );
}

export function useMemoryStats() {
  return useFetch<MemoryStats>(
    "/v1/memory/stats",
    {
      episodic_count: 0,
      semantic_count: 0,
      procedural_count: 0,
      total_entries: 0,
      db_size_mb: 0,
    },
    { transform: normalizeMemoryStats },
  );
}

// ═══════════════════════════════════════════════════════════════════
// TRUST SURFACE — row 6 (Node0 Closure Sprint, 2026-04-21)
// ═══════════════════════════════════════════════════════════════════
//
// Authoritative receipt chain head proxied from the Rust cognition-gateway
// via /v1/chain. This is Dema's public truth-surface binding: the chain
// IS the evidence of lawful operation; the web face reveals it verbatim.
// When the proxy returns 503 (gateway unreachable), useChainHead's `error`
// state is set — the UI MUST show that honestly, never fabricate a head.

export interface ChainHead {
  head: string; // 64-char hex, or "" when gateway unreachable
  length: number;
  latestTimestamp: number | null;
  sovereignEnvelopes?: number;
  sovereignEntries?: number;
}

function normalizeChainHead(payload: unknown): ChainHead {
  const data = asObject(payload);
  return {
    head: asString(data.head, ""),
    length: asNumber(data.length, 0),
    latestTimestamp:
      typeof data.latestTimestamp === "number" ? data.latestTimestamp : null,
    sovereignEnvelopes:
      typeof data.sovereignEnvelopes === "number"
        ? data.sovereignEnvelopes
        : undefined,
    sovereignEntries:
      typeof data.sovereignEntries === "number"
        ? data.sovereignEntries
        : undefined,
  };
}

export function useChainHead() {
  return useFetch<ChainHead>(
    "/v1/chain",
    { head: "", length: 0, latestTimestamp: null },
    { transform: normalizeChainHead },
  );
}

// Latest receipt detail — kind + timestamp from the head of the chain.
// Sourced from /v1/chain/latest which combines Rust gateway's
// GET /chain + GET /chain/{head} into one authoritative payload.
// When chain is at genesis (length=0) or gateway unreachable,
// latestReceipt is null — UI must render that as honest absence.

export interface LatestReceipt {
  id: string;
  kind: string; // e.g. "MissionApproved", "PrincipalActivation"
  timestamp: number | null;
  // Additional fields vary by receipt kind — kept loose for forward compat.
  [key: string]: unknown;
}

export interface ChainLatest {
  head: string;
  length: number;
  latestTimestamp: number | null;
  latestReceipt: LatestReceipt | null;
  latestReceiptError: {
    upstream_status: number;
    detail: string;
  } | null;
}

function normalizeLatestReceipt(payload: unknown): LatestReceipt | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const data = asObject(payload);
  const id = asString(data.id, "");
  const kind = asString(data.kind, "");
  if (!id && !kind) {
    return null;
  }
  const timestamp =
    typeof data.timestamp === "number" ? data.timestamp : null;
  // Preserve the full upstream payload for forward-compat surfaces; the
  // id/kind/timestamp triple is the canonical minimum contract this hook
  // promises.
  return { ...data, id, kind, timestamp } as LatestReceipt;
}

function normalizeChainLatest(payload: unknown): ChainLatest {
  const data = asObject(payload);
  const latestReceiptError = asObject(data.latestReceiptError);
  return {
    head: asString(data.head, ""),
    length: asNumber(data.length, 0),
    latestTimestamp:
      typeof data.latestTimestamp === "number" ? data.latestTimestamp : null,
    latestReceipt: normalizeLatestReceipt(data.latestReceipt),
    latestReceiptError:
      Object.keys(latestReceiptError).length === 0
        ? null
        : {
            upstream_status: asNumber(latestReceiptError.upstream_status),
            detail: asString(latestReceiptError.detail),
          },
  };
}

export function useChainLatest() {
  return useFetch<ChainLatest>(
    "/v1/chain/latest",
    {
      head: "",
      length: 0,
      latestTimestamp: null,
      latestReceipt: null,
      latestReceiptError: null,
    },
    { transform: normalizeChainLatest, resetOnError: true },
  );
}

export function useMemoryProfile() {
  return useFetch<MemoryProfile>(
    "/v1/memory/profile",
    {
      privacy_note: "All data is local",
      briefing: normalizeTerminalBriefing({}),
      semantic_profile: {
        preferred_domains: [],
        active_hours: "",
        vocabulary_signature: "",
        work_window: "",
      },
      missions: [],
      active_projects: [],
      work_streak: 0,
      near_compile_patterns: [],
      compiled_reflex_summary: [],
      stats: normalizeMemoryStats({}),
    },
    { transform: normalizeMemoryProfile },
  );
}

export function useTerminalState() {
  return useFetch<TerminalStateSnapshot>(
    "/v1/terminal/state",
    {
      state: "ready",
      execution_path: "SYSTEM_2_NOVEL",
      mission_id: "",
      restart_required: false,
    },
    { intervalMs: 5000 },
  );
}

export function useNode0Readiness() {
  return useFetch<Node0Readiness>(
    "/v1/node0/readiness",
    {
      status: "red",
      generated_at: "",
      product_shell: {
        available: false,
        version: "",
        default_view: "node0",
      },
      proof_surface: {
        available: false,
        source: "mission_receipt",
      },
      runtime: {
        live: false,
        state: "unknown",
        ihsan_score: 0,
        snr_score: 0,
      },
      boot_service: {
        status: "unavailable",
        booted: false,
        node_id: "",
        total_breaths: 0,
        chain_hash: "",
        error: "",
      },
      memory_import: {
        available: false,
        status: "unavailable",
        mode: "single_user_provided_record",
        imported_records: 0,
        requires_consent: true,
        source: "agent_db",
        truth_label: "[ENFORCEMENT: WIRED]",
      },
      voice_input: {
        available: false,
        status: "unsupported",
        mode: "browser_speech_recognition",
        requires_user_gesture: true,
        stores_audio: false,
        auto_submit: false,
        truth_label: "[ENFORCEMENT: WIRED]",
      },
      desktop_browser_action: {
        available: false,
        status: "unavailable",
        mode: "client_handoff_only",
        allowed_actions: [],
        requires_user_confirmation: true,
        server_executes: false,
        truth_label: "[ENFORCEMENT: WIRED]",
      },
      local_action_executor: {
        available: false,
        status: "unavailable",
        mode: "explicit_user_gesture",
        allowed_actions: [],
        requires_user_confirmation: true,
        server_executes: false,
        records_receipts: false,
        truth_label: "[ENFORCEMENT: WIRED]",
      },
      always_on_daemon: {
        available: false,
        status: "unavailable",
        mode: "local_ambient_loop",
        pid: 0,
        lock_path: "",
        root: "",
        last_tick_at: "",
        last_receipt_id: "",
        requires_operator_confirmation: true,
        server_executes: false,
        no_public_network_listener: true,
        writes_under_state_dir: true,
        status_command: "python scripts/dema/dema_service.py status --root sovereign_state/dema",
        start_command:
          "python scripts/dema/dema_daemon.py --loop --interval-seconds 60 --root sovereign_state/dema",
        truth_label: "[ENFORCEMENT: WIRED]",
      },
      spearpoint: {
        status: "unknown",
        artifact_status: "missing",
        run_id: "",
        mode: "",
        timestamp_utc: "",
        targets_completed: 0,
        official_submission: false,
        classification: "internal_strict_harness",
      },
      next_action: "start or repair Dema service",
    },
    { transform: normalizeNode0Readiness, intervalMs: 10000, resetOnError: true },
  );
}

export async function createNode0ActionIntent(args: {
  actionType: "open_url" | "copy_text";
  target: string;
  label?: string;
}): Promise<Node0ActionIntentResult> {
  return requestJson<Node0ActionIntentResult>("/v1/node0/action-intent", {
    method: "POST",
    body: JSON.stringify({
      action_type: args.actionType,
      target: args.target,
      label: args.label ?? "",
      consent: true,
    }),
  });
}

export async function recordNode0LocalActionReceipt(args: {
  actionId: string;
  actionType: "open_url" | "copy_text";
  result: "executed" | "blocked" | "failed";
  targetPreview: string;
  targetHash: string;
  error?: string;
}): Promise<Node0LocalActionReceipt> {
  return requestJson<Node0LocalActionReceipt>("/v1/node0/local-action/receipt", {
    method: "POST",
    body: JSON.stringify({
      action_id: args.actionId,
      action_type: args.actionType,
      result: args.result,
      execution_channel: "browser_client",
      user_confirmed: true,
      target_preview: args.targetPreview,
      target_hash: args.targetHash,
      error: args.error ?? "",
    }),
  });
}

export function useSignatureInfo() {
  return useFetch<SignatureInfo>(
    "/v1/verify/signature",
    {
      node_id: "",
      public_key: "",
      algorithms: {
        signing: "Ed25519",
        hashing: "BLAKE3",
        canonicalization: "RFC 8785",
        audit_chain: "HMAC-SHA256",
      },
    },
  );
}

export function useMissionPlanner() {
  const { data: health } = useSovereignHealth();
  const [receipt, setReceipt] = useState<MissionReceipt | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const defaultEnvelope = health.permission_defaults ?? defaultPermissionEnvelope();

  const submitMission = async (args: {
    description: string;
    source?: string;
    permissionEnvelope?: PermissionEnvelope;
    proofMode?: "auto" | "verified" | "standard";
  }): Promise<MissionReceipt | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await requestJson<MissionReceipt>("/v1/plan", {
        method: "POST",
        body: JSON.stringify({
          description: args.description,
          source: args.source ?? "terminal",
          permission_envelope: args.permissionEnvelope ?? defaultEnvelope,
          proof_mode: args.proofMode ?? "verified",
        }),
      });
      setReceipt(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Mission failed";
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return {
    submitMission,
    receipt,
    loading,
    error,
    clearReceipt: () => setReceipt(null),
    defaultPermissionEnvelope: defaultEnvelope,
  };
}

function normalizeTerminalStreamEvent(payload: unknown): TerminalStreamEvent {
  const data = asObject(payload);
  return {
    topic: asString(data.topic),
    severity: asString(
      data.severity,
      "info",
    ) as TerminalStreamEvent["severity"],
    mission_id: asString(data.mission_id),
    receipt_id: asString(data.receipt_id),
    event_hash: asString(data.event_hash),
    prev_hash: asString(data.prev_hash),
    timestamp: asString(data.timestamp),
    payload: asObject(data.payload),
    source: asString(data.source),
  };
}

function normalizeCriticalAcknowledgmentReceipt(
  payload: unknown,
): CriticalAcknowledgmentReceipt {
  const data = asObject(payload);
  return {
    acknowledgement_id: asString(data.acknowledgement_id),
    receipt_id: asString(data.receipt_id),
    status: "ACKNOWLEDGED",
    hash_chain_ref: asString(data.hash_chain_ref),
    acknowledged_event_hash: asString(data.acknowledged_event_hash),
    acknowledged_topic: asString(data.acknowledged_topic),
    mission_id: asString(data.mission_id),
    timestamp: asString(data.timestamp),
    synthesis: asString(data.synthesis),
  };
}

export function useTerminalStream(topics: string[] = ["*"], historyLimit = 100) {
  const [events, setEvents] = useState<TerminalStreamEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const topicsRef = useRef(topics);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const retriesRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);
  const topicsKey = JSON.stringify(topics);

  useEffect(() => {
    topicsRef.current = topics;
  }, [topics]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    let active = true;

    const connect = () => {
      if (!active) {
        return;
      }

      const url = new URL(websocketUrl("/v1/stream"));
      url.searchParams.set("session_id", getSessionId());
      const ws = new WebSocket(url.toString());
      wsRef.current = ws;

      ws.onopen = () => {
        retriesRef.current = 0;
        setConnected(true);
        ws.send(JSON.stringify({ type: "subscribe", topics: topicsRef.current }));
        ws.send(
          JSON.stringify({
            type: "history",
            topics: topicsRef.current,
            limit: historyLimit,
          }),
        );
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as Record<string, unknown>;
          if (payload.type === "history" && Array.isArray(payload.events)) {
            setEvents(
              payload.events
                .map(normalizeTerminalStreamEvent)
                .slice(-historyLimit)
                .reverse(),
            );
            return;
          }
          if (payload.type === "event") {
            const nextEvent = normalizeTerminalStreamEvent(payload.event);
            setEvents((previous) => {
              const next = [
                nextEvent,
                ...previous.filter((item) => item.event_hash !== nextEvent.event_hash),
              ];
              return next.slice(0, historyLimit);
            });
          }
        } catch {
          // Ignore malformed stream payloads.
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
        setConnected(false);
        if (!active) {
          return;
        }
        const delay = Math.min(1000 * 2 ** retriesRef.current, 30_000);
        retriesRef.current += 1;
        reconnectTimerRef.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      active = false;
      setConnected(false);
      window.clearTimeout(reconnectTimerRef.current);
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [historyLimit, topicsKey]);

  return { events, connected };
}

export function useCriticalEventAcknowledger() {
  const [acknowledgingId, setAcknowledgingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const acknowledge = async (args: {
    eventHash: string;
    topic: string;
    summary: string;
    missionId?: string;
    receiptId?: string;
  }): Promise<CriticalAcknowledgmentReceipt | null> => {
    setAcknowledgingId(args.eventHash);
    setError(null);

    try {
      const payload = await requestJson<unknown>(
        "/v1/terminal/critical-acknowledgments",
        {
          method: "POST",
          body: JSON.stringify({
            event_hash: args.eventHash,
            topic: args.topic,
            summary: args.summary,
            mission_id: args.missionId ?? "",
            receipt_id: args.receiptId ?? "",
          }),
        },
      );
      return normalizeCriticalAcknowledgmentReceipt(payload);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to acknowledge critical event";
      setError(message);
      return null;
    } finally {
      setAcknowledgingId(null);
    }
  };

  return {
    acknowledge,
    acknowledgingId,
    error,
    clearError: () => setError(null),
  };
}

export function useModelRoutingSettings() {
  const { data: health } = useSovereignHealth();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async (modelRouting: Record<string, string>) => {
    setSaving(true);
    setError(null);
    try {
      const response = await requestJson<{ model_routing: Record<string, string> }>(
        "/v1/settings/model-routing",
        {
          method: "PUT",
          body: JSON.stringify(modelRouting),
        },
      );
      return response.model_routing;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to save model routing";
      setError(message);
      return null;
    } finally {
      setSaving(false);
    }
  };

  return {
    modelRouting: health.model_routing ?? {},
    permissionDefaults: health.permission_defaults ?? defaultPermissionEnvelope(),
    authState: health.auth_state ?? "unavailable",
    runtimeMode: health.runtime_mode ?? "development",
    save,
    saving,
    error,
  };
}
