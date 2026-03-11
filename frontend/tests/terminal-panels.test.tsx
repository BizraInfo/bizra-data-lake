/**
 * Terminal Panel Tests — Coverage for terminal-* components.
 * One render per component, multiple assertions per render.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';

vi.mock('../src/hooks/use-sovereign-api', () => ({
  useSovereignHealth: () => ({
    data: { status: 'healthy', uptime_s: 3600, version: '3.0.0-GENESIS', ihsan_score: 0.97, snr_score: 0.92 },
  }),
  useSeedPotential: () => ({
    data: { potential: 0.78, tier: 'SPROUT', growth_rate: 0.12 },
  }),
  useTokenBalance: () => ({
    data: { seed: 42.5, bloom: 1.23, staked: 5.0 },
  }),
  useConstitutionalStatus: () => ({
    data: { ihsan: 0.97, snr: 0.92, gini: 0.28, gates_passed: 3, gates_total: 3 },
  }),
  useNetworkEffect: () => ({ data: { node_count: 1, edge_count: 0, density: 0 } }),
  useNetworkMilestones: () => ({ data: [] }),
  useNodeLifecycle: () => ({ data: { phase: 'genesis', age_hours: 24, transitions: 1 } }),
  useNodeValue: () => ({ data: { value: 0.65, rank: 1, percentile: 100 } }),
  useSeedEpisodes: () => ({ data: [] }),
  useTerminalBriefing: () => ({
    data: { summary: 'Good morning', alerts: [], recommendations: ['Keep going'] },
  }),
  useMemoryStats: () => ({
    data: { episodic_count: 12, semantic_count: 45, procedural_count: 3, total_entries: 60, db_size_mb: 2.5 },
  }),
}));

afterEach(() => { cleanup(); });

import TerminalDashboard from '../src/components/terminal/terminal-dashboard';
import TerminalMission from '../src/components/terminal/terminal-mission';
import TerminalMemory from '../src/components/terminal/terminal-memory';
import TerminalSettings from '../src/components/terminal/terminal-settings';
import TerminalNetwork from '../src/components/terminal/terminal-network';
import TerminalSkills from '../src/components/terminal/terminal-skills';

describe('Terminal Panels', () => {
  it('Dashboard renders headings and metrics', () => {
    const { unmount } = render(<TerminalDashboard />);
    expect(screen.getByText('Dashboard')).toBeTruthy();
    expect(screen.getByText(/Sovereign node/i)).toBeTruthy();
    expect(screen.getByText('SEED')).toBeTruthy();
    expect(screen.getByText('BLOOM')).toBeTruthy();
    expect(screen.getByText(/Constitutional/i)).toBeTruthy();
    unmount();
  });

  it('Mission renders heading, input, recent missions', () => {
    const { unmount } = render(<TerminalMission />);
    expect(screen.getByText('Mission')).toBeTruthy();
    expect(screen.getByPlaceholderText(/mission/i)).toBeTruthy();
    expect(screen.getByText(/Recent/i)).toBeTruthy();
    unmount();
  });

  it('Memory renders heading and store', () => {
    const { unmount } = render(<TerminalMemory />);
    expect(screen.getByText('Memory')).toBeTruthy();
    expect(screen.getByText(/Memory Store/i)).toBeTruthy();
    unmount();
  });

  it('Settings renders identity, routing, environment', () => {
    const { unmount } = render(<TerminalSettings />);
    expect(screen.getByText('Settings')).toBeTruthy();
    expect(screen.getByText(/Node Identity/i)).toBeTruthy();
    expect(screen.getByText(/Model Routing/i)).toBeTruthy();
    expect(screen.getByText(/Environment/i)).toBeTruthy();
    unmount();
  });

  it('Network renders heading and lifecycle', () => {
    const { unmount } = render(<TerminalNetwork />);
    expect(screen.getByText(/Network/i)).toBeTruthy();
    expect(screen.getByText(/Lifecycle/i)).toBeTruthy();
    unmount();
  });

  it('Skills renders agents and tier', () => {
    const { unmount } = render(<TerminalSkills />);
    expect(screen.getByText('Agents & Skills')).toBeTruthy();
    expect(screen.getByText(/Personal Agent Team/)).toBeTruthy();
    expect(screen.getByText('Current Tier')).toBeTruthy();
    unmount();
  });
});
