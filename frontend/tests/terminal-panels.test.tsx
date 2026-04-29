/**
 * Terminal Panel Tests — Coverage for terminal-* components.
 * One render per component, multiple assertions per render.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';

const acknowledgeCriticalEventMock = vi.fn().mockResolvedValue({
  acknowledgement_id: 'ack-001',
  receipt_id: 'ack-receipt-001',
  status: 'ACKNOWLEDGED',
  hash_chain_ref: 'ack-hash-001',
  acknowledged_event_hash: 'event-hash-005',
  acknowledged_topic: 'ihsan.breach',
  mission_id: '',
  timestamp: '2026-03-11T10:01:30Z',
  synthesis: 'Critical event acknowledged',
});

const terminalPanelMockData = vi.hoisted(() => ({
  seedEpisodes: [
    {
      index: 12,
      timestamp: '2026-03-11T09:58:00Z',
      snr: 0.92,
      ihsan: 0.97,
      reward: 1.5,
      qualified: true,
      tier: 'SPROUT',
      sovereignty_score: 0.78,
      receipt_hash: 'receipt-hash-001',
    },
  ],
}));

vi.mock('../src/hooks/use-sovereign-api', () => ({
  useSovereignHealth: () => ({
    data: {
      status: 'ready',
      uptime_s: 3600,
      version: '3.0.0-GENESIS',
      ihsan_score: 0.97,
      snr_score: 0.92,
      env: 'development',
      live_status: 'LIVE',
      running: true,
      gini: 0.28,
      last_tick_timestamp: '2026-03-11T10:00:00Z',
      tick_interval_s: 60,
      wallet_snapshot: { seed: 42.5, bloom: 1.23 },
      last_mission_summary: 'Last mission quality Ihsan 0.97 SNR 0.92',
      auth_state: 'anonymous-dev',
      runtime_mode: 'development',
      model_routing: { planner: 'gpt-4.1', executor: 'gpt-4.1-mini' },
      permission_defaults: {
        filesystem: ['workspace/**'],
        applications: ['terminal', 'editor'],
        network: [],
        data_sensitivity: 'standard',
        spend_budget_usd: 3,
        time_budget_seconds: 900,
        escalation: 'ask-on-boundary-cross',
        audit_verbosity: 'standard',
      },
    },
  }),
  useSeedPotential: () => ({
    data: { sovereignty_score: 0.78, tier: 'SPROUT', tier_progress: 0.12, streak: 3, episodes_total: 12, reward_ema: 0.97 },
  }),
  useConstitutionalStatus: () => ({
    data: { status: 'active', gini: 0.28, asabiyyah: 0.61, tick_interval_s: 60, last_tick_timestamp: '2026-03-11T10:00:00Z', reflexes: 1, pending_receipts: 0, pending_proposals: 0, wallets: 1, events: 4 },
  }),
  useTerminalBriefing: () => ({
    data: {
      active_project: 'bizra-data-lake',
      last_mission_summary: 'Last mission quality Ihsan 0.97 SNR 0.92',
      next_action_suggestion: 'Review the permission envelope, then execute your next mission.',
      quality_trend: 'stable',
      near_compile_patterns: ['pattern-a'],
      time_since_last_mission_s: 120,
      wallet_snapshot: { seed: 42.5, bloom: 1.23 },
    },
  }),
  useMissionPlanner: () => ({
    submitMission: vi.fn(),
    receipt: null,
    loading: false,
    error: null,
    clearReceipt: vi.fn(),
    defaultPermissionEnvelope: {
      filesystem: ['workspace/**'],
      applications: ['terminal', 'editor'],
      network: [],
      data_sensitivity: 'standard',
      spend_budget_usd: 3,
      time_budget_seconds: 900,
      escalation: 'ask-on-boundary-cross',
      audit_verbosity: 'standard',
    },
  }),
  useTerminalState: () => ({
    data: { state: 'ready', execution_path: 'SYSTEM_2_NOVEL', mission_id: '' },
  }),
  useModelRoutingSettings: () => ({
    modelRouting: { planner: 'gpt-4.1', executor: 'gpt-4.1-mini' },
    permissionDefaults: {
      filesystem: ['workspace/**'],
      applications: ['terminal', 'editor'],
      network: [],
      data_sensitivity: 'standard',
      spend_budget_usd: 3,
      time_budget_seconds: 900,
      escalation: 'ask-on-boundary-cross',
      audit_verbosity: 'standard',
    },
    authState: 'anonymous-dev',
    runtimeMode: 'development',
    save: vi.fn(),
    saving: false,
    error: null,
  }),
  useNetworkEffect: () => ({
    data: { nodes: 1, skills_available: 8, compute_tflops: 0.25, latency_factor: 0.9 },
  }),
  useNetworkMilestones: () => ({ data: [] }),
  useNodeLifecycle: () => ({
    data: { current_stage: 'Seed', next_stage: 'Sapling', sovereignty_score: 0.78, progress: 0.24, rank: 1 },
  }),
  useNodeValue: () => ({
    data: { composite: 0.65, potential: 0.7, activation: 0.62, quality: 0.88, compounding: 0.54, synergy: 0.51 },
  }),
  useSeedEpisodes: () => ({
    data: terminalPanelMockData.seedEpisodes,
  }),
  useMemoryStats: () => ({
    data: { episodic_count: 12, semantic_count: 45, procedural_count: 3, total_entries: 60, db_size_mb: 2.5 },
  }),
  useMemoryProfile: () => ({
    data: {
      privacy_note: 'All data is local',
      briefing: {
        active_project: 'bizra-data-lake',
        last_mission_summary: 'Last mission quality Ihsan 0.97 SNR 0.92',
        next_action_suggestion: 'Submit one more mission.',
        quality_trend: 'stable',
        near_compile_patterns: ['pattern-a'],
        time_since_last_mission_s: 120,
        wallet_snapshot: { seed: 42.5, bloom: 1.23 },
      },
      semantic_profile: {
        preferred_domains: ['software engineering'],
        active_hours: '08:00-18:00 GST',
        vocabulary_signature: 'local-first constitutional terminal',
        work_window: 'weekdays',
      },
      missions: [],
      active_projects: [],
      work_streak: 4,
      near_compile_patterns: [
        { name: 'terminal_contract_render', count: 2, threshold: 3, avg_ihsan: 0.97 },
      ],
      compiled_reflex_summary: [
        {
          name: 'close_learning_loop',
          avg_ihsan: 0.99,
          execution_count: 3,
          avg_latency_ms: 48,
          compiled_at: '2026-03-11T09:59:00Z',
          last_hit_at: '2026-03-11T10:00:00Z',
        },
      ],
      stats: { episodic_count: 12, semantic_count: 45, procedural_count: 3, total_entries: 60, db_size_mb: 2.5 },
    },
  }),
  useTerminalStream: () => ({
    connected: true,
    events: [
      {
        topic: 'receipt.generated',
        severity: 'info',
        mission_id: 'episode-12',
        receipt_id: 'receipt-hash-001',
        event_hash: 'event-hash-001',
        prev_hash: '',
        timestamp: '2026-03-11T09:58:00Z',
        source: 'mission',
        payload: {
          mission_id: 'episode-12',
          receipt_id: 'receipt-hash-001',
          status: 'COMPLETE',
          ihsan_score: 0.97,
          snr_score: 0.92,
        },
      },
      {
        topic: 'mission.verified',
        severity: 'info',
        mission_id: 'episode-12',
        receipt_id: 'receipt-hash-001',
        event_hash: 'event-hash-001b',
        prev_hash: 'event-hash-001',
        timestamp: '2026-03-11T09:58:30Z',
        source: 'mission',
        payload: {
          mission_id: 'episode-12',
          receipt_id: 'receipt-hash-001',
          proof_receipt_id: 'proof-receipt-001',
          proof_status: 'ACCEPTED',
          verified: true,
          vrg_root: 'vrg-root-abcdef123456',
          branch_count: 4,
          surviving_branches: 3,
        },
      },
      {
        topic: 'reflex.compiled',
        severity: 'notice',
        mission_id: 'episode-12',
        receipt_id: 'receipt-hash-001',
        event_hash: 'event-hash-002',
        prev_hash: 'event-hash-001b',
        timestamp: '2026-03-11T09:59:00Z',
        source: 'mission',
        payload: {
          mission_id: 'episode-12',
          receipt_id: 'receipt-hash-001',
          name: 'close_learning_loop',
          avg_ihsan: 0.99,
          execution_count: 3,
        },
      },
      {
        topic: 'tick.completed',
        severity: 'info',
        mission_id: '',
        receipt_id: '',
        event_hash: 'event-hash-003',
        prev_hash: 'event-hash-002',
        timestamp: '2026-03-11T10:00:00Z',
        source: 'tick',
        payload: {
          scored: 1,
          minted: 1.5,
          reflexes: 1,
        },
      },
      {
        topic: 'auth.boundary.crossed',
        severity: 'warning',
        mission_id: 'episode-12',
        receipt_id: '',
        event_hash: 'event-hash-004',
        prev_hash: 'event-hash-003',
        timestamp: '2026-03-11T10:00:30Z',
        source: 'auth',
        payload: {
          reason: 'authentication_required',
        },
      },
      {
        topic: 'ihsan.breach',
        severity: 'critical',
        mission_id: '',
        receipt_id: '',
        event_hash: 'event-hash-005',
        prev_hash: 'event-hash-004',
        timestamp: '2026-03-11T10:01:00Z',
        source: 'tick',
        payload: {
          rejected_count: 2,
        },
      },
    ],
  }),
  useCriticalEventAcknowledger: () => ({
    acknowledge: acknowledgeCriticalEventMock,
    acknowledgingId: null,
    error: null,
    clearError: vi.fn(),
  }),
  useSignatureInfo: () => ({
    data: {
      node_id: 'node-1234567890',
      public_key: 'pubkey-abcdef1234567890',
      algorithms: {
        signing: 'Ed25519',
        hashing: 'BLAKE3',
        canonicalization: 'RFC 8785',
        audit_chain: 'HMAC-SHA256',
      },
    },
  }),
  useTokenBalance: () => ({
    data: { seed: 42.5, bloom: 1.23, staked: 5.0 },
  }),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

import TerminalDashboard from '../src/components/terminal/terminal-dashboard';
import TerminalMission from '../src/components/terminal/terminal-mission';
import TerminalMemory from '../src/components/terminal/terminal-memory';
import TerminalSettings from '../src/components/terminal/terminal-settings';
import TerminalNetwork from '../src/components/terminal/terminal-network';
import TerminalSkills from '../src/components/terminal/terminal-skills';
import TerminalTimeline from '../src/components/terminal/terminal-timeline';

describe.sequential('Terminal Panels', () => {
  it('Dashboard renders headings and metrics', () => {
    render(<TerminalDashboard />);
    expect(screen.getByText('Dashboard')).toBeTruthy();
    expect(screen.getByText(/node readiness/i)).toBeTruthy();
    expect(screen.getByText('SEED')).toBeTruthy();
    expect(screen.getByText('BLOOM')).toBeTruthy();
    expect(screen.getByText(/Constitutional/i)).toBeTruthy();
  });

  it('Mission renders heading, input, recent missions', () => {
    render(<TerminalMission />);
    expect(screen.getByText('Mission')).toBeTruthy();
    const composer = screen.getByPlaceholderText(/mission/i);
    expect(composer).toBeTruthy();
    fireEvent.change(composer, { target: { value: 'Run constitutional review' } });
    fireEvent.click(screen.getByText(/Review Envelope/i));
    expect(screen.getByText(/Live Boundary Alerts/i)).toBeTruthy();
    expect(screen.getByText(/Auth boundary crossed/i)).toBeTruthy();
  });

  it('Memory renders heading and store', () => {
    render(<TerminalMemory />);
    expect(screen.getByText('Memory')).toBeTruthy();
    expect(screen.getByText(/Memory Store/i)).toBeTruthy();
  });

  it('Settings renders identity, routing, environment', () => {
    render(<TerminalSettings />);
    expect(screen.getByText('Settings')).toBeTruthy();
    expect(screen.getByText(/Node Identity/i)).toBeTruthy();
    expect(screen.getByText(/Model Routing/i)).toBeTruthy();
    expect(screen.getByText(/Environment/i)).toBeTruthy();
  });

  it('Network renders heading and lifecycle', () => {
    render(<TerminalNetwork />);
    expect(screen.getByText(/Network/i)).toBeTruthy();
    expect(screen.getByText(/Lifecycle/i)).toBeTruthy();
  });

  it('Skills renders agents and tier', () => {
    render(<TerminalSkills />);
    expect(screen.getByText('Agents & Skills')).toBeTruthy();
    expect(screen.getByText(/Personal Agent Team/)).toBeTruthy();
    expect(screen.getByText('Current Tier')).toBeTruthy();
    expect(screen.getByText(/file_organization/)).toBeTruthy();
    expect(screen.getByText(/test_generation/)).toBeTruthy();
  });

  it('Timeline renders receipted episode evidence', async () => {
    render(<TerminalTimeline />);
    expect(screen.getByText('Timeline')).toBeTruthy();
    expect(screen.getByText(/Living narrative/i)).toBeTruthy();
    const episode = await screen.findByText(/Episode: Ihsān 0.97, \+1.5 SEED/i);
    fireEvent.click(episode);
    expect(screen.getByText(/receipt-hash-001/i)).toBeTruthy();
    expect(screen.getByText(/All entries from EventBus or ActionBus/i)).toBeTruthy();
  });
});
