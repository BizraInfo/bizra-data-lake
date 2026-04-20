/**
 * BIZRA CRYPTOGRAPHIC SMOKE VERIFIER
 * ============================================================================
 * Enforces Manifest §12: Empirical Reproducibility & Truth-Discipline Pipeline
 *
 * Target routes: `/`, `/manifesto`, `/brand`.
 * Actions: headless navigation, DOM integrity check, visual capture, and
 * SHA-256 Merkle root generation per face-validation run.
 *
 * Canon references:
 *   - `docs/design/CANON-TERMS.md` §01 (CLAIM_MUST_BIND)
 *   - ADK North Star V2 · L-001 ("BIZRA reveals truth, never simulates")
 *
 * Output: JSON receipt at `.bizra/receipts/smoke/smoke_receipt_<ts>.json`
 * (gitignored by default; Merkle root + receipt path land in commit body
 * as cryptographic evidence).
 *
 * Run with Bun:   `bun run scripts/smoke-verifier.ts`
 * Prereqs: dev server at http://localhost:3000 + `playwright` dev dep +
 * `npx playwright install chromium` (one-time browser binary download).
 * ============================================================================
 */

import { chromium, Browser, Page } from "playwright";
import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";

// ─── Cryptographic primitives ───────────────────────────────────

const hashData = (data: string | Buffer): string =>
  crypto.createHash("sha256").update(data).digest("hex");

const computeMerkleRoot = (hashes: string[]): string => {
  if (hashes.length === 0) return hashData("");
  if (hashes.length === 1) return hashes[0];

  const nextLevel: string[] = [];
  for (let i = 0; i < hashes.length; i += 2) {
    const left = hashes[i];
    const right = i + 1 < hashes.length ? hashes[i + 1] : left;
    nextLevel.push(hashData(left + right));
  }
  return computeMerkleRoot(nextLevel);
};

// ─── Receipt types ──────────────────────────────────────────────

interface RouteVerdict {
  route: string;
  status: "PASS" | "FAIL";
  dom_hash: string;
  screenshot_hash: string;
  console_errors: number;
  load_time_ms: number;
}

interface SmokePassReceipt {
  receipt_id: string;
  timestamp_iso: string;
  base_url: string;
  routes_verified: number;
  merkle_root: string;
  verdicts: RouteVerdict[];
  signature: string;
}

// ─── Verification engine ────────────────────────────────────────

class SmokeVerifier {
  private browser: Browser | null = null;
  private readonly baseUrl: string;
  private readonly routes: string[];
  private readonly outputDir: string;

  constructor(baseUrl: string = process.env.BIZRA_SMOKE_URL || "http://localhost:3000") {
    this.baseUrl = baseUrl;
    this.routes = ["/", "/manifesto", "/brand"];
    this.outputDir = path.join(process.cwd(), ".bizra", "receipts", "smoke");

    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
    }
  }

  private async verifyRoute(page: Page, route: string): Promise<RouteVerdict> {
    console.log(`\n[IHSAN] Orienting to route: ${route}`);
    const url = `${this.baseUrl}${route}`;
    const startTime = Date.now();
    let consoleErrors = 0;

    const errorMessages: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        consoleErrors++;
        errorMessages.push(`[console] ${msg.text().slice(0, 200)}`);
      }
    });
    page.on("pageerror", (err) => {
      consoleErrors++;
      errorMessages.push(`[pageerror] ${err.message.slice(0, 200)}`);
    });
    // Surface error details to stdout when non-empty (after goto).
    const flushErrors = () => {
      if (errorMessages.length > 0) {
        console.log(`  └─ errors:`);
        errorMessages.forEach((m) => console.log(`     ${m}`));
      }
    };

    try {
      // Next.js dev mode keeps an HMR WebSocket open — "networkidle" never
      // fires. Use "domcontentloaded" (reliable on dev + prod) and follow
      // with a small settle pause so async in-view animations can mount.
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
      await page.waitForLoadState("load", { timeout: 10000 }).catch(() => {
        // load never arriving on dev is acceptable; we already have DOM.
      });
      await page.waitForTimeout(800);
      const loadTimeMs = Date.now() - startTime;

      const domContent = await page.content();
      const domHash = hashData(domContent);

      const screenshotPath = path.join(
        this.outputDir,
        `snapshot_${route.replace("/", "root")}_${Date.now()}.png`,
      );
      const screenshotBuffer = await page.screenshot({
        path: screenshotPath,
        fullPage: true,
      });
      const screenshotHash = hashData(screenshotBuffer);

      const status = consoleErrors === 0 ? "PASS" : "FAIL";
      console.log(
        `[VERDICT] ${route} -> ${status} | Load: ${loadTimeMs}ms | Errors: ${consoleErrors}`,
      );
      flushErrors();

      return {
        route,
        status,
        dom_hash: domHash,
        screenshot_hash: screenshotHash,
        console_errors: consoleErrors,
        load_time_ms: loadTimeMs,
      };
    } catch (error: any) {
      console.error(`[REJECT] Failed to verify ${route}: ${error.message}`);
      return {
        route,
        status: "FAIL",
        dom_hash: hashData("FAILED"),
        screenshot_hash: hashData("FAILED"),
        console_errors: consoleErrors + 1,
        load_time_ms: Date.now() - startTime,
      };
    }
  }

  public async executeRun(): Promise<void> {
    console.log("==================================================");
    console.log(" BIZRA CRYPTOGRAPHIC SMOKE VERIFIER ACTIVATED");
    console.log("==================================================");

    this.browser = await chromium.launch({ headless: true });
    const context = await this.browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 2,
    });

    const verdicts: RouteVerdict[] = [];
    const leafHashes: string[] = [];

    for (const route of this.routes) {
      const page = await context.newPage();
      const verdict = await this.verifyRoute(page, route);
      verdicts.push(verdict);
      leafHashes.push(hashData(verdict.dom_hash + verdict.screenshot_hash));
      await page.close();
    }

    await this.browser.close();

    const allPassed = verdicts.every((v) => v.status === "PASS");
    if (!allPassed) {
      console.error(
        "\n[CRITICAL] Smoke pass failed. SHADOW_STATE detected. Halting receipt generation.",
      );
      process.exit(1);
    }

    const merkleRoot = computeMerkleRoot(leafHashes);
    const timestamp = new Date().toISOString();

    const receipt: SmokePassReceipt = {
      receipt_id: hashData(merkleRoot + timestamp),
      timestamp_iso: timestamp,
      base_url: this.baseUrl,
      routes_verified: this.routes.length,
      merkle_root: merkleRoot,
      verdicts,
      signature: "UNSIGNED_AWAITING_PQC_NODE_KEY",
    };

    const receiptPath = path.join(
      this.outputDir,
      `smoke_receipt_${Date.now()}.json`,
    );
    fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2));

    console.log("\n==================================================");
    console.log(" SMOKE PASS COMPLETE: Canonical Status Achieved");
    console.log("==================================================");
    console.log(` Merkle Root: ${merkleRoot}`);
    console.log(` Receipt Sealed: ${receiptPath}`);
    console.log("==================================================\n");
  }
}

const verifier = new SmokeVerifier();
verifier.executeRun().catch((err) => {
  console.error(err);
  process.exit(1);
});
