import { defineConfig, devices } from "@playwright/test";

/**
 * BIZRA E2E Configuration
 * ADR-012: Canary gate requires functional + visual validation
 *
 * Usage:
 *   npx playwright test                    # All tests
 *   npx playwright test canary-smoke       # Canary gate only
 *   npx playwright test --update-snapshots # Update visual baselines
 */
export default defineConfig({
  testDir: ".",
  timeout: 30_000,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,

  use: {
    baseURL: process.env.BASE_URL || "http://localhost:8000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  reporter: process.env.CI
    ? [["json", { outputFile: "results/e2e-results.json" }], ["html", { open: "never" }]]
    : [["html", { open: "on-failure" }]],

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
