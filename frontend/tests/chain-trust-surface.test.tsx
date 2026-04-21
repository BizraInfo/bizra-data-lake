/**
 * Trust-surface guard — Dema reveals one authoritative CHAIN/RECEIPT snapshot.
 *
 * Sprint: Node0 Closure — row 6 (trust_surface) binding (2026-04-21).
 *
 * These tests lock the operator-visible truth surface:
 *  1. CHAIN comes from the authoritative /v1/chain/latest snapshot.
 *  2. RECEIPT renders kind/timestamp when available.
 *  3. Genesis and receipt-detail lookup failure are distinct absence states.
 *  4. IHSĀN band and GINI come from authoritative /v1/health data.
 *  5. A polling outage must clear stale chain data instead of simulating liveness.
 */

import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TerminalShell from "../src/components/terminal/terminal-shell";

const CANONICAL_HEAD =
  "369319cd83e2419114dc5c3f36467f5665ab7fddac299e01b8d8374302ff676a";
const RECEIPT_TIMESTAMP = 1745180214;
const RECEIPT_TIME_LABEL = new Date(RECEIPT_TIMESTAMP * 1000).toLocaleTimeString(
  undefined,
  {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  },
);

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

describe("Trust surface — row 6", () => {
  it("renders CHAIN, RECEIPT, IHSĀN band, and GINI from authoritative sources", async () => {
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
          head: CANONICAL_HEAD,
          length: 9,
          latestTimestamp: RECEIPT_TIMESTAMP,
          latestReceipt: {
            id: "rct_369319cd83e24191",
            kind: "MissionApproved",
            timestamp: RECEIPT_TIMESTAMP,
          },
        },
      },
    });

    render(<TerminalShell />);

    await waitFor(() => {
      expect(screen.getByTestId("chain-status").textContent).toContain("CHAIN#9");
      expect(screen.getByTestId("chain-status").textContent).toContain(
        CANONICAL_HEAD.slice(0, 8),
      );
      expect(screen.getByTestId("receipt-status").textContent).toContain(
        "RECEIPT:MissionApproved",
      );
      expect(screen.getByTestId("receipt-status").textContent).toContain(
        RECEIPT_TIME_LABEL,
      );
      expect(screen.getByTestId("ihsan-band").textContent).toContain("IHSĀN:ideal");
      expect(screen.getByTestId("gini-status").textContent).toContain("GINI:0.31");
    });

    expect(screen.getByTestId("chain-status").getAttribute("title")).toContain(
      CANONICAL_HEAD,
    );
    expect(screen.getByTestId("receipt-status").getAttribute("title")).toContain(
      "MissionApproved",
    );
  });

  it("renders honest genesis absence when the chain has no receipts yet", async () => {
    installFetchMock({
      "/v1/health": {
        status: 200,
        body: {
          status: "healthy",
          running: true,
          ihsan_score: 0.94,
          gini: 0.29,
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

    render(<TerminalShell />);

    await waitFor(() => {
      expect(screen.getByTestId("chain-status").textContent).toContain("CHAIN#0");
      expect(screen.getByTestId("chain-status").textContent).toContain("—");
      expect(screen.getByTestId("receipt-status").textContent).toBe("RECEIPT:— —");
    });

    expect(screen.getByTestId("receipt-status").getAttribute("title")).toContain(
      "chain at genesis",
    );
  });

  it("distinguishes receipt lookup failure from genesis", async () => {
    installFetchMock({
      "/v1/health": {
        status: 200,
        body: {
          status: "healthy",
          running: true,
          ihsan_score: 0.92,
          gini: 0.33,
        },
      },
      "/v1/chain/latest": {
        status: 200,
        body: {
          head: CANONICAL_HEAD,
          length: 9,
          latestTimestamp: RECEIPT_TIMESTAMP,
          latestReceipt: null,
          latestReceiptError: {
            upstream_status: 404,
            detail: "receipt not found",
          },
        },
      },
    });

    render(<TerminalShell />);

    await waitFor(() => {
      expect(screen.getByTestId("chain-status").textContent).toContain("CHAIN#9");
      expect(screen.getByTestId("receipt-status").textContent).toBe("RECEIPT:— —");
    });

    const title = screen.getByTestId("receipt-status").getAttribute("title") ?? "";
    expect(title).toContain("upstream lookup failed");
    expect(title).not.toContain("genesis");
  });

  it("clears stale chain state on polling outage instead of showing the old head", async () => {
    let chainLatestCallCount = 0;
    const intervalCallbacks: Array<() => void> = [];

    vi.spyOn(window, "setInterval").mockImplementation(
      (
        callback: (...args: any[]) => void,
        _delay?: number,
        ..._args: any[]
      ) => {
        intervalCallbacks.push(callback);
        return intervalCallbacks.length as unknown as ReturnType<
          typeof window.setInterval
        >;
      },
    );
    vi.spyOn(window, "clearInterval").mockImplementation(() => {});

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
      "/v1/chain/latest": () => {
        chainLatestCallCount += 1;
        if (chainLatestCallCount === 1) {
          return {
            status: 200,
            body: {
              head: CANONICAL_HEAD,
              length: 9,
              latestTimestamp: RECEIPT_TIMESTAMP,
              latestReceipt: {
                id: "rct_369319cd83e24191",
                kind: "MissionApproved",
                timestamp: RECEIPT_TIMESTAMP,
              },
            },
          };
        }
        return {
          status: 503,
          body: {
            status: "gateway_unreachable",
            error: "ConnectError",
          },
        };
      },
    });

    render(<TerminalShell />);

    await act(async () => {
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.getByTestId("chain-status").textContent).toContain("CHAIN#9");
    });

    await act(async () => {
      for (const callback of intervalCallbacks) {
        callback();
      }
      await Promise.resolve();
    });

    await waitFor(() => {
      const chain = screen.getByTestId("chain-status");
      expect(chain.textContent).toContain("CHAIN#0");
      expect(chain.textContent).toContain("—");
      expect(chain.textContent).not.toContain(CANONICAL_HEAD.slice(0, 8));
    });

    expect(screen.getByTestId("receipt-status").getAttribute("title")).toContain(
      "gateway unreachable",
    );
  });

  it("renders honest unavailable health states instead of simulated band values", async () => {
    installFetchMock({
      "/v1/health": {
        status: 503,
        body: {
          status: "gateway_unreachable",
        },
      },
      "/v1/chain/latest": {
        status: 200,
        body: {
          head: CANONICAL_HEAD,
          length: 9,
          latestTimestamp: RECEIPT_TIMESTAMP,
          latestReceipt: {
            id: "rct_369319cd83e24191",
            kind: "MissionApproved",
            timestamp: RECEIPT_TIMESTAMP,
          },
        },
      },
    });

    render(<TerminalShell />);

    await waitFor(() => {
      expect(screen.getByTestId("ihsan-band").textContent).toContain("IHSĀN:—");
      expect(screen.getByTestId("gini-status").textContent).toContain("GINI:—");
    });
  });
});
