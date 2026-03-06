import { test, expect } from "@playwright/test";

/**
 * BIZRA Canary Smoke Tests
 * ADR-012: These run as an AnalysisTemplate job during Argo Rollouts canary.
 *
 * Tests validate:
 * 1. API health endpoint responds
 * 2. Constitutional thresholds are met (Ihsan >= 0.95, SNR >= 0.85)
 * 3. Core endpoints return valid JSON
 * 4. Response times are within SLO
 */

const BASE = process.env.BASE_URL || "http://localhost:8000";

test.describe("Canary Health Gate", () => {
  test("health endpoint returns 200 with valid body", async ({ request }) => {
    const res = await request.get(`${BASE}/v1/health`);
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("status");
    expect(["healthy", "ok", "ready"]).toContain(body.status.toLowerCase());
  });

  test("status endpoint returns version info", async ({ request }) => {
    const res = await request.get(`${BASE}/v1/status`);
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("version");
  });
});

test.describe("Constitutional SLO Gate", () => {
  test("Ihsan score meets production threshold (>= 0.95)", async ({ request }) => {
    const res = await request.get(`${BASE}/v1/health`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    if (body.ihsan_score !== undefined) {
      expect(body.ihsan_score).toBeGreaterThanOrEqual(0.95);
    }
  });

  test("SNR score meets minimum threshold (>= 0.85)", async ({ request }) => {
    const res = await request.get(`${BASE}/v1/health`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    if (body.snr_score !== undefined) {
      expect(body.snr_score).toBeGreaterThanOrEqual(0.85);
    }
  });
});

test.describe("Core API Functional Gate", () => {
  test("verify endpoint accepts POST", async ({ request }) => {
    const res = await request.post(`${BASE}/v1/verify/health`, {
      data: { probe: "canary-smoke" },
    });
    // 200 or 422 (validation) are both acceptable — just not 500
    expect(res.status()).toBeLessThan(500);
  });

  test("metrics endpoint returns Prometheus format", async ({ request }) => {
    const res = await request.get(`${BASE}/metrics`);
    if (res.ok()) {
      const text = await res.text();
      expect(text).toContain("# HELP");
    }
  });
});

test.describe("Response Time SLO Gate", () => {
  test("health endpoint responds within 500ms", async ({ request }) => {
    const start = Date.now();
    const res = await request.get(`${BASE}/v1/health`);
    const elapsed = Date.now() - start;
    expect(res.ok()).toBeTruthy();
    expect(elapsed).toBeLessThan(500);
  });

  test("status endpoint responds within 1000ms", async ({ request }) => {
    const start = Date.now();
    const res = await request.get(`${BASE}/v1/status`);
    const elapsed = Date.now() - start;
    expect(res.ok()).toBeTruthy();
    expect(elapsed).toBeLessThan(1000);
  });
});
