/**
 * Trust-surface chain banner guard — Dema web face reveals live chain truth.
 *
 * Sprint: Node0 Closure — row 6 (trust_surface) binding (2026-04-21).
 *
 * The StatusBar in terminal-shell.tsx renders a live chain indicator
 * sourced from the /v1/chain proxy (which forwards to the Rust
 * cognition-gateway). This guard enforces the "no shadow state" canon
 * on the UI side:
 *
 *  1. When the chain is reachable, the banner shows "CHAIN#<length>
 *     <head-short>" with the authoritative head prefix.
 *  2. When the chain is NOT reachable (fetch fails with non-200 or
 *     network error), the banner shows "CHAIN#0 —" — honestly signaling
 *     absence. It MUST NOT fabricate a head.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';

import TerminalShell from '../src/components/terminal/terminal-shell';

const CANONICAL_HEAD =
  '369319cd83e2419114dc5c3f36467f5665ab7fddac299e01b8d8374302ff676a';

interface MockCall {
  url: string;
  status: number;
  body: unknown;
}

function mockEndpoint(url: string, status: number, body: unknown): void {
  const call: MockCall = { url, status, body };
  const currentMocks = (globalThis as unknown as { __bizraMocks?: MockCall[] })
    .__bizraMocks ?? [];
  currentMocks.push(call);
  (globalThis as unknown as { __bizraMocks: MockCall[] }).__bizraMocks =
    currentMocks;
}

beforeEach(() => {
  (globalThis as unknown as { __bizraMocks: MockCall[] }).__bizraMocks = [];

  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const urlStr = typeof input === 'string' ? input : input.toString();
    const mocks =
      (globalThis as unknown as { __bizraMocks?: MockCall[] }).__bizraMocks ??
      [];

    for (const mock of mocks) {
      if (urlStr.endsWith(mock.url) || urlStr.includes(mock.url)) {
        return {
          ok: mock.status >= 200 && mock.status < 300,
          status: mock.status,
          json: async () => mock.body,
          text: async () =>
            typeof mock.body === 'string'
              ? mock.body
              : JSON.stringify(mock.body),
        } as Response;
      }
    }
    // Default: return an empty 200 for any unmocked endpoint so the
    // StatusBar's other hooks (useSovereignHealth, useSeedPotential,
    // useTokenBalance) don't crash the test on unrelated fetches.
    return {
      ok: true,
      status: 200,
      json: async () => ({}),
      text: async () => '{}',
    } as Response;
  }) as typeof fetch;
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('Trust surface — chain banner', () => {
  it('renders live chain head when /v1/chain proxy succeeds', async () => {
    mockEndpoint('/v1/chain', 200, {
      head: CANONICAL_HEAD,
      length: 9,
      latestTimestamp: 1745180214,
      sovereignEnvelopes: 1,
      sovereignEntries: 5,
    });

    render(<TerminalShell />);

    await waitFor(
      () => {
        const banner = screen.getByTestId('chain-status');
        // CHAIN#9 369319cd
        expect(banner.textContent).toContain('CHAIN#9');
        expect(banner.textContent).toContain(CANONICAL_HEAD.slice(0, 8));
      },
      { timeout: 2000 },
    );

    // Verifiably the truth — no shadow state. Tooltip carries the full head.
    const banner = screen.getByTestId('chain-status');
    expect(banner.getAttribute('title')).toContain(CANONICAL_HEAD);
  });

  it('shows honest absence when /v1/chain returns 503 (gateway down)', async () => {
    mockEndpoint('/v1/chain', 503, {
      status: 'gateway_unreachable',
      gateway_url: 'http://localhost:7421/chain',
      error: 'ConnectError',
    });

    render(<TerminalShell />);

    await waitFor(
      () => {
        const banner = screen.getByTestId('chain-status');
        // On error, useFetch keeps the fallback { head: "", length: 0 }.
        // Banner shows "CHAIN#0 —" — NOT a fabricated head.
        expect(banner.textContent).toContain('CHAIN#0');
        expect(banner.textContent).toContain('—');
      },
      { timeout: 2000 },
    );

    const banner = screen.getByTestId('chain-status');
    // Must NOT display any hex head when gateway is unreachable.
    expect(banner.textContent).not.toContain(CANONICAL_HEAD.slice(0, 8));
    expect(banner.getAttribute('title')).toContain('unavailable');
  });

  it('shows honest absence when /v1/chain returns invalid head (empty string)', async () => {
    mockEndpoint('/v1/chain', 200, {
      head: '',
      length: 0,
      latestTimestamp: null,
    });

    render(<TerminalShell />);

    await waitFor(
      () => {
        const banner = screen.getByTestId('chain-status');
        expect(banner.textContent).toContain('—');
      },
      { timeout: 2000 },
    );
  });
});
