# BIZRA Dema Console

بسم الله الرحمن الرحيم

The operator face for Node0. Forked from an external Z.ai prototype and stripped to bind exclusively to BIZRA's `bizra-cognition-gateway` HTTP surface.

**Status:** `v0.1.0-alpha.1` — stripped + typed. Not yet wired end-to-end to the 6 surfaces from `cycle-7/surface-catalog-v1.md`.

## Authority

- **Kernel truth:** Rust `bizra-cognition` + Python `sovereign_state/`. Not here.
- **HTTP surface:** Rust `bizra-cognition-gateway` on `:7421`. Source of truth for all mission state.
- **Face:** this package. Consumes the gateway, displays, never writes chain state directly.

The Dema one-face law (manifest §) holds: this console is one of several modalities over the same lawful loop. The terminal face is `dema` (CLI, shipped in Cycle-7). This is the web face.

## Stack (post-strip)

- **Runtime:** Next.js 16 + React 19 + Bun
- **UI:** Tailwind 4 + shadcn/ui (52 Radix primitives) + Framer Motion
- **State:** Zustand (in-session), no Redux
- **Data:** `src/bindings/` — typed contracts from `bizra-cognition-gateway` via ts-rs
- **Forms:** react-hook-form + zod
- **Explicitly NOT installed:** `z-ai-web-dev-sdk`, `next-auth`, Firebase, Gemini SDK

## Wiring to the gateway

All lawful actions go through `src/lib/gateway-client.ts`:

```ts
import { submitOrganize } from "@/lib/gateway-client";

const result = await submitOrganize({ path: "/home/mumo/docs", qualityScore: 0.98 });
if (result.kind === "ok") {
  // result.data is OrganizeResponseContract — fully typed
  console.log(result.data.chainHead === result.data.organizeReceiptId); // ✓ sealed
} else if (result.kind === "refused") {
  // 400/403/422 — name the refusal reason + show remediation
  console.log(result.error.code, result.error.message);
} else {
  // gateway unreachable — honest operator message
}
```

The three-way `GatewayOutcome<T>` maps directly to the gateway's HTTP status contract (see `cycle-7/surface-catalog-v1.md` §3.3).

## Local development (intended)

```bash
# 1. Start BIZRA cognition gateway (another terminal):
cd /data/bizra/repos/bizra-data-lake/bizra-omega
BIZRA_DEMA_CACHE_ROOT=/tmp/dema-console-dev \
  BIZRA_IDENTITY_ANCHOR=/tmp/dema-console-dev/identity/credentials.json \
  cargo run --release -p bizra-cognition-gateway

# 2. Install + run the console:
cd dema-console
bun install
bun run dev  # http://localhost:3000
```

## Bindings sync

The 20 `.ts` contract files in `src/bindings/` are copies of
`bizra-omega/bizra-cognition-gateway/bindings/`. They stay in sync via:

1. Rust DTO changes in `bizra-cognition-gateway/src/contracts.rs`
2. `cargo test -p bizra-cognition-gateway --bin bizra-cognition-gateway`
3. Regenerated `.ts` gets committed on the same PR (CI gate enforces this)
4. A sync step — v0.1 is manual `cp`; v0.2 will be a script

**NEVER edit `src/bindings/*.ts` by hand** — they will be overwritten.

## What this console does NOT do

Per `cycle-7/surface-catalog-v1.md` §6:

- Not a dashboard (no always-on state exposure)
- Not a chat interface (no bubbles, no freeform text walls)
- Not a workflow builder (the lawful loop is the workflow)
- Not a PAT/SAT swarm console (hidden organism stays hidden)
- Not a settings panel (Mission Composer is where lawful intent begins)

## What is NOT yet wired (v0.1 gaps)

- The 6 surfaces (Mission Composer, Gate Ladder, Action Surface, Receipt Reveal, Memory Constellation, Reject Remediation) exist as Z.ai-upstream components under `src/components/` but are **not yet bound** to `gateway-client.ts`. That wiring is the next arc after this session.
- No tests yet. Tests are a follow-up commit on this branch.
- Prisma schema exists but the decision is pending: either delete Prisma and read cache JSONs directly, or make Prisma a read-only view. See ADR §5.12.

## Canon links

- `cycle-7/retrospective.md` — what Node0 can do today
- `cycle-7/surface-catalog-v1.md` — the 6 surfaces
- `cycle-7/prototype-adoption-adr-v1.md` — why this fork exists and what was rejected
- `bizra-omega/bizra-cognition-gateway/bindings/README.md` — how `src/bindings/` is generated

**سُبْحَانَ اللَّهِ**
