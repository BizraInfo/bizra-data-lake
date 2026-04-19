// ═══════════════════════════════════════════════════════════════
// DEMA — Health Check Endpoint
// Operational readiness verification.
// ═══════════════════════════════════════════════════════════════

import { NextResponse } from "next/server";
import { db } from "@/lib/db";

const START_TIME = Date.now();

export async function GET() {
  const startTime = performance.now();

  try {
    const receiptCount = await db.receipt.count();
    const manifestCount = await db.manifest.count();

    const uptime = Math.floor((Date.now() - START_TIME) / 1000);
    const latencyMs = Math.round(performance.now() - startTime);

    return NextResponse.json({
      status: "healthy",
      version: "0.1.0",
      phase: "R1",
      uptime,
      latencyMs,
      checks: {
        database: { status: "ok", receipts: receiptCount, manifests: manifestCount },
        runtime: { status: "ok", node: process.version },
      },
      timestamp: new Date().toISOString(),
    });
  } catch {
    const latencyMs = Math.round(performance.now() - startTime);
    return NextResponse.json(
      { status: "unhealthy", error: "Database connectivity failed", latencyMs, timestamp: new Date().toISOString() },
      { status: 503 },
    );
  }
}
