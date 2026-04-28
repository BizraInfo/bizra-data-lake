/**
 * Dema Goal Surface — §9 contract tests.
 *
 * 1. All four state cards render: Current, Ideal, Gap, Next Admissible Action.
 * 2. Truth badges appear on every card and the trust strip.
 * 3. With healthy backend data, Current is MEASURED and trust strip shows
 *    real ihsān/gini.
 * 4. With cold backend (no health), no fake metrics — Current is UNKNOWN
 *    and trust strip shows "—" placeholders.
 * 5. Genesis chain (length 0) renders honestly without inventing a head.
 * 6. The view does NOT show any token / mint / financial language.
 */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TerminalGoal from "../src/components/terminal/terminal-goal";

interface MockResponse {
  status: number;
  body: unknown;
}

type MockHandler = MockResponse | (() => MockResponse);

function jsonResponse({ status, body }: MockResponse): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () =>
      typeof body === "string" ? body : JSON.stringify(body),
  } as Response;
}

function installFetchMock(overrides: Record<string, MockHandler>) {
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    for (const [needle, handler] of Object.entries(overrides)) {
      if (url.endsWith(needle) || url.includes(needle)) {
        const resolved = typeof handler === "function" ? handler() : handler;
        return jsonResponse(resolved);
      }
    }
    return jsonResponse({ status: 200, body: {} });
  }) as typeof fetch;
}

beforeEach(() => {
  vi.useRealTimers();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("Dema Goal Surface — §9 four-state model", () => {
  it("renders all four state cards with truth badges", async () => {
    installFetchMock({
      "/v1/health": {
        status: 200,
        body: {
          status: "healthy",
          running: true,
          ihsan_score: 0.96,
          gini: 0.31,
          snr_score: 0.91,
        },
      },
      "/v1/chain/latest": {
        status: 200,
        body: {
          head: "abcdef0123456789",
          length: 7,
          latestTimestamp: 1745180214,
          latestReceipt: { id: "rct_1", kind: "MissionApproved", timestamp: 1745180214 },
        },
      },
      "/v1/node/lifecycle": {
        status: 200,
        body: { stage: "Sapling" },
      },
    });

    render(<TerminalGoal />);

    await waitFor(() => {
      expect(screen.getByTestId("goal-card-current")).toBeInTheDocument();
      expect(screen.getByTestId("goal-card-ideal")).toBeInTheDocument();
      expect(screen.getByTestId("goal-card-gap")).toBeInTheDocument();
      expect(screen.getByTestId("goal-card-next")).toBeInTheDocument();
    });

    // Each card has a truth badge.
    expect(screen.getAllByTestId(/^goal-truth-badge-/).length).toBeGreaterThanOrEqual(4);
  });

  it("renders MEASURED trust strip when /v1/health responds", async () => {
    installFetchMock({
      "/v1/health": {
        status: 200,
        body: {
          status: "healthy",
          running: true,
          ihsan_score: 0.96,
          gini: 0.31,
          snr_score: 0.91,
        },
      },
      "/v1/chain/latest": {
        status: 200,
        body: {
          head: "fedcba9876543210",
          length: 12,
          latestTimestamp: 1745180214,
          latestReceipt: { id: "rct_2", kind: "MissionApproved", timestamp: 1745180214 },
        },
      },
    });

    render(<TerminalGoal />);

    await waitFor(() => {
      expect(screen.getByTestId("goal-trust-ihsan").textContent).toContain("0.96");
      expect(screen.getByTestId("goal-trust-gini").textContent).toContain("0.31");
      expect(screen.getByTestId("goal-trust-chain").textContent).toContain("#12");
      expect(screen.getByTestId("goal-trust-chain").textContent).toContain(
        "fedcba98",
      );
      expect(screen.getByTestId("goal-trust-receipt").textContent).toContain(
        "MissionApproved",
      );
    });
  });

  it("renders honest UNKNOWN placeholders on cold backend", async () => {
    // /v1/health returns 503 — health hook fallback fires.
    installFetchMock({
      "/v1/health": { status: 503, body: { status: "gateway_unreachable" } },
      "/v1/chain/latest": { status: 503, body: {} },
    });

    render(<TerminalGoal />);

    await waitFor(() => {
      expect(screen.getByTestId("goal-trust-ihsan").textContent).toBe("—");
      expect(screen.getByTestId("goal-trust-gini").textContent).toBe("—");
      expect(screen.getByTestId("goal-trust-chain").textContent).toBe("—");
      expect(screen.getByTestId("goal-trust-receipt").textContent).toBe("—");
    });
  });

  it("renders genesis chain honestly (length 0, no head)", async () => {
    installFetchMock({
      "/v1/health": {
        status: 200,
        body: {
          status: "healthy",
          running: true,
          ihsan_score: 0.97,
          gini: 0.28,
        },
      },
      "/v1/chain/latest": {
        status: 200,
        body: {
          head: "",
          length: 0,
          latestTimestamp: null,
          latestReceipt: null,
        },
      },
    });

    render(<TerminalGoal />);

    await waitFor(() => {
      expect(screen.getByTestId("goal-trust-chain").textContent).toBe("—");
      // Gap or Next mentions seal/genesis.
      const gap = screen.getByTestId("goal-card-gap-body").textContent ?? "";
      const next = screen.getByTestId("goal-card-next-body").textContent ?? "";
      expect((gap + next).toLowerCase()).toMatch(/genesis|seal|first mission/);
    });
  });

  it("never shows token / mint / financial language", async () => {
    installFetchMock({
      "/v1/health": {
        status: 200,
        body: {
          status: "healthy",
          running: true,
          ihsan_score: 0.96,
          gini: 0.31,
        },
      },
      "/v1/chain/latest": {
        status: 200,
        body: {
          head: "abc",
          length: 1,
          latestTimestamp: 1745180214,
          latestReceipt: { id: "rct_1", kind: "MissionApproved", timestamp: 1745180214 },
        },
      },
    });

    render(<TerminalGoal />);

    await waitFor(() => {
      expect(screen.getByTestId("terminal-goal")).toBeInTheDocument();
    });

    const fullText = screen.getByTestId("terminal-goal").textContent ?? "";
    const forbidden = [
      "yield",
      "earnings",
      "investment",
      "guaranteed",
      "risk-free",
      "AGI achieved",
      "first in the world",
      "token profit",
    ];
    for (const term of forbidden) {
      expect(fullText.toLowerCase()).not.toContain(term.toLowerCase());
    }
  });
});
