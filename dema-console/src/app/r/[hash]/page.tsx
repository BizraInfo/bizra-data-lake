// bizra.ai/r/<hash> — public receipt viewer
//
// بسم الله الرحمن الرحيم
//
// Shareable URL for a sealed receipt. The user pastes a 64-char hex
// hash; this page attempts to fetch it from the local gateway at
// 127.0.0.1:7421 and renders it honestly.
//
// Three outcomes, NO_SHADOW_STATE at the result boundary:
//   - ok: receipt found — render its canonical fields verbatim
//   - not_found: honest "no receipt with this hash on this node"
//   - unreachable: honest "local gateway unreachable; run `dema start`"
//
// Local-first discipline: this page does NOT attempt to fetch from
// a remote gateway. Receipts are by default LOCAL to the operator's
// own node. The public bizra.ai/r/<hash> URL scheme is a way to
// SHARE a hash — the receiver pastes it into THEIR own dema-console
// where THEIR local gateway answers. No cross-node leakage.

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type ReceiptOutcome =
  | { kind: "loading" }
  | { kind: "ok"; data: Record<string, unknown> }
  | { kind: "not_found"; hash: string }
  | { kind: "unreachable"; reason: string }
  | { kind: "invalid_hash"; hash: string };

const GATEWAY_URL =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_BIZRA_GATEWAY_URL
    ? process.env.NEXT_PUBLIC_BIZRA_GATEWAY_URL
    : "http://127.0.0.1:7421";

function isValidHash(hash: string): boolean {
  return /^[0-9a-f]{64}$/i.test(hash);
}

async function fetchReceipt(hash: string): Promise<ReceiptOutcome> {
  if (!isValidHash(hash)) {
    return { kind: "invalid_hash", hash };
  }
  try {
    const resp = await fetch(`${GATEWAY_URL}/chain/${hash}`, {
      cache: "no-store",
    });
    if (resp.status === 200) {
      const data = await resp.json();
      return { kind: "ok", data };
    }
    if (resp.status === 404) {
      return { kind: "not_found", hash };
    }
    return {
      kind: "unreachable",
      reason: `gateway responded ${resp.status}`,
    };
  } catch (err) {
    return {
      kind: "unreachable",
      reason: err instanceof Error ? err.message : "network error",
    };
  }
}

export default function ReceiptPage({
  params,
}: {
  params: { hash: string };
}) {
  const [outcome, setOutcome] = useState<ReceiptOutcome>({ kind: "loading" });

  useEffect(() => {
    fetchReceipt(params.hash).then(setOutcome);
  }, [params.hash]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-2xl mx-auto px-6 py-12 space-y-6">
        <header className="space-y-2">
          <Link
            href="/landing"
            className="text-xs text-muted-foreground hover:text-foreground underline decoration-dotted"
          >
            ← DEMA landing
          </Link>
          <h1 className="text-xl font-semibold">Receipt viewer</h1>
          <code className="block text-[11px] font-mono text-muted-foreground break-all">
            {params.hash}
          </code>
        </header>

        {outcome.kind === "loading" && (
          <Card className="border-border/50 bg-muted/10">
            <CardContent className="p-4 text-sm text-muted-foreground">
              Checking local gateway at <code className="font-mono">{GATEWAY_URL}</code>…
            </CardContent>
          </Card>
        )}

        {outcome.kind === "invalid_hash" && (
          <Card className="border-destructive/50 bg-destructive/5">
            <CardHeader className="pb-2">
              <Badge variant="destructive" className="w-fit">Invalid hash</Badge>
            </CardHeader>
            <CardContent className="p-4 pt-0 text-sm text-muted-foreground space-y-2">
              <p>
                A sealed receipt hash is 64 lowercase hexadecimal characters.
                The path parameter received does not match that shape.
              </p>
              <p className="text-xs">
                No kernel lookup was attempted; NO_SHADOW_STATE at the input
                boundary.
              </p>
            </CardContent>
          </Card>
        )}

        {outcome.kind === "unreachable" && (
          <Card className="border-warning/50 bg-warning/5">
            <CardHeader className="pb-2">
              <Badge variant="outline" className="w-fit text-warning border-warning/50">
                Gateway unreachable
              </Badge>
            </CardHeader>
            <CardContent className="p-4 pt-0 text-sm text-muted-foreground space-y-3">
              <p>
                Receipts are local-first — they live on YOUR node's chain,
                not on a central server. This page looked for the receipt at
                your local gateway and couldn't reach it.
              </p>
              <div className="bg-muted/30 border border-border/50 rounded p-3 space-y-2">
                <p className="text-xs font-medium text-foreground">
                  If DEMA is installed on this machine:
                </p>
                <code className="block font-mono text-xs">
                  dema start   # or wherever you launch the gateway
                </code>
                <p className="text-xs">
                  then reload this page.
                </p>
              </div>
              <div className="bg-muted/30 border border-border/50 rounded p-3 space-y-2">
                <p className="text-xs font-medium text-foreground">
                  If DEMA is installed on a different machine:
                </p>
                <p className="text-xs">
                  Copy the hash and paste it into <code className="font-mono">dema receipt {`{hash}`}</code> on the machine that sealed it.
                </p>
              </div>
              <p className="text-[11px] italic">Reason: {outcome.reason}</p>
            </CardContent>
          </Card>
        )}

        {outcome.kind === "not_found" && (
          <Card className="border-border/50 bg-muted/10">
            <CardHeader className="pb-2">
              <Badge variant="outline" className="w-fit">
                Not on this node
              </Badge>
            </CardHeader>
            <CardContent className="p-4 pt-0 text-sm text-muted-foreground space-y-3">
              <p>
                The gateway at <code className="font-mono">{GATEWAY_URL}</code> was reached, but it has no receipt with this hash.
              </p>
              <p className="text-xs">
                This is an honest answer. The kernel never fabricates a receipt
                it hasn't sealed.
              </p>
              <div className="bg-muted/30 border border-border/50 rounded p-3 text-xs space-y-1">
                <p className="text-foreground font-medium">
                  If you believe this hash was sealed somewhere:
                </p>
                <p>
                  - Check the machine that produced it — each node's chain is sovereign.
                </p>
                <p>
                  - Re-run <code className="font-mono">dema chain</code> to see what IS sealed here.
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {outcome.kind === "ok" && (
          <Card className="border-trust/50 bg-trust/5">
            <CardHeader className="pb-2">
              <Badge variant="outline" className="w-fit text-trust border-trust/50">
                Receipt sealed
              </Badge>
            </CardHeader>
            <CardContent className="p-4 pt-0 space-y-3">
              <p className="text-xs text-muted-foreground">
                Canonical fields returned by the local gateway. Verify by running{" "}
                <code className="font-mono bg-muted/30 px-1 rounded">dema receipt {params.hash}</code>{" "}
                on the sealing node.
              </p>
              <pre className="bg-muted/20 border border-border/50 rounded p-3 text-[11px] font-mono overflow-x-auto">
                {JSON.stringify(outcome.data, null, 2)}
              </pre>
            </CardContent>
          </Card>
        )}

        <footer className="text-center text-[11px] text-muted-foreground/70 italic pt-8 border-t border-border/50">
          <p>Close it. Prove it. Reveal it.</p>
        </footer>
      </div>
    </div>
  );
}
