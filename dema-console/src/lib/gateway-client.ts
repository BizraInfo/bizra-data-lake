// ═══════════════════════════════════════════════════════════════
// BIZRA Cognition Gateway client — typed fetch surface
// ═══════════════════════════════════════════════════════════════
//
// The UI NEVER asserts admissibility itself. Every lawful action
// goes through this client which POSTs to bizra-cognition-gateway.
// The gateway returns the authoritative verdict + receipts; this
// client only transports and types them.
//
// Drift guarantee: every type below comes from `src/bindings/`,
// which is generated from Rust DTOs by ts-rs. A Rust rename fails
// CI (see bizra-cognition-gateway/bindings/README.md).

import type { ActivatePrincipalResponseContract } from "@/bindings/ActivatePrincipalResponseContract";
import type { ErrorResponseContract } from "@/bindings/ErrorResponseContract";
import type { OrganizeResponseContract } from "@/bindings/OrganizeResponseContract";
import type { PoiLedgerResponseContract } from "@/bindings/PoiLedgerResponseContract";
import type { PoiSummaryResponseContract } from "@/bindings/PoiSummaryResponseContract";
import type { ResourceContract } from "@/bindings/ResourceContract";
import type { UrpViewContract } from "@/bindings/UrpViewContract";

// ─── configuration ────────────────────────────────────────────────

const DEFAULT_GATEWAY_URL = "http://127.0.0.1:7421";

function gatewayUrl(): string {
  if (typeof process !== "undefined" && process.env.NEXT_PUBLIC_BIZRA_GATEWAY_URL) {
    return process.env.NEXT_PUBLIC_BIZRA_GATEWAY_URL;
  }
  return DEFAULT_GATEWAY_URL;
}

// ─── outcome type — successful response OR structured gateway error ─

/**
 * Three-way outcome matching the gateway's HTTP status contract.
 *   - "ok": 200 — lawful permit, data is the typed response
 *   - "refused": 400/403/422 — constitutional pre-gate refusal OR
 *     admissibility rejection. `error.code` names the refusal class.
 *     Data is the error envelope with remediation strings.
 *   - "unreachable": network error, gateway down, non-JSON response,
 *     or unexpected status code. Caller must surface an honest
 *     "cannot reach cognition runtime" to the operator.
 */
export type GatewayOutcome<T> =
  | { kind: "ok"; data: T }
  | { kind: "refused"; status: number; error: ErrorResponseContract["error"] }
  | { kind: "unreachable"; reason: string };

// ─── request helpers ──────────────────────────────────────────────

async function get<T>(path: string): Promise<GatewayOutcome<T>> {
  try {
    const res = await fetch(`${gatewayUrl()}${path}`, { method: "GET" });
    return await parseResponse<T>(res);
  } catch (e) {
    return { kind: "unreachable", reason: (e as Error).message };
  }
}

async function post<Req, T>(path: string, body: Req): Promise<GatewayOutcome<T>> {
  try {
    const res = await fetch(`${gatewayUrl()}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    return await parseResponse<T>(res);
  } catch (e) {
    return { kind: "unreachable", reason: (e as Error).message };
  }
}

async function parseResponse<T>(res: Response): Promise<GatewayOutcome<T>> {
  const text = await res.text();
  if (res.ok) {
    try {
      return { kind: "ok", data: JSON.parse(text) as T };
    } catch (e) {
      return { kind: "unreachable", reason: `ok status but non-JSON body: ${(e as Error).message}` };
    }
  }
  try {
    const err = JSON.parse(text) as ErrorResponseContract;
    return { kind: "refused", status: res.status, error: err.error };
  } catch {
    return { kind: "unreachable", reason: `HTTP ${res.status} with non-JSON body` };
  }
}

// ─── principal activation (G2) ────────────────────────────────────

export interface ActivatePrincipalRequest {
  principalName: string;
  declaredRole?: string;
  qualityScore?: number;
  identityAnchorPath?: string;
}

export function activatePrincipal(
  req: ActivatePrincipalRequest,
): Promise<GatewayOutcome<ActivatePrincipalResponseContract>> {
  return post("/principal/activate", req);
}

// ─── resources + URP (G4) ────────────────────────────────────────

export interface RegisterResourceRequest {
  kind: string;
  id: string;
  summary?: string;
  allowlisted?: boolean;
}

export function registerResource(
  req: RegisterResourceRequest,
): Promise<GatewayOutcome<{ outcome: string; resource: ResourceContract }>> {
  return post("/resources/register", req);
}

export function listResources(): Promise<GatewayOutcome<{ resources: ResourceContract[] }>> {
  return get("/resources/list");
}

export function getUrp(): Promise<GatewayOutcome<UrpViewContract>> {
  return get("/resources/urp");
}

// ─── organize mission (G5) ───────────────────────────────────────

export interface OrganizeRequest {
  path: string;
  qualityScore?: number;
}

export function submitOrganize(
  req: OrganizeRequest,
): Promise<GatewayOutcome<OrganizeResponseContract>> {
  return post("/missions/organize", req);
}

// ─── Proof-of-Impact (G6) ────────────────────────────────────────

export function getPoiLedger(): Promise<GatewayOutcome<PoiLedgerResponseContract>> {
  return get("/poi/ledger");
}

export function getPoiSummary(): Promise<GatewayOutcome<PoiSummaryResponseContract>> {
  return get("/poi/summary");
}

// ─── health (trivial liveness) ────────────────────────────────────

export function health(): Promise<GatewayOutcome<{ status: string; domain: string }>> {
  return get("/health");
}
