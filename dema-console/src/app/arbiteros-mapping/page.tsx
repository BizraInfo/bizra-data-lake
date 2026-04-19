import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export const metadata = {
  title: "BIZRA ↔ ArbiterOS mapping",
  description:
    "Academic convergence: arXiv:2510.13857v1 (Xu et al., CUHK, 2025-10-12) theorizes exactly the architecture BIZRA has been implementing in Rust for 3 years. External validation, not source material.",
};

export default function ArbiterOsMappingPage() {
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
          <h1 className="text-xl font-semibold">BIZRA ↔ ArbiterOS mapping</h1>
        </header>

        <Card className="border-border/50 bg-muted/10">
          <CardHeader className="pb-2">
            <Badge variant="outline" className="w-fit">External academic convergence</Badge>
          </CardHeader>
          <CardContent className="p-4 pt-0 space-y-3 text-sm text-muted-foreground">
            <p>
              In October 2025, Xu et al. at the Chinese University of Hong Kong published{" "}
              <em>From Craft to Constitution: A Governance-First Paradigm for
              Principled Agent Engineering</em> (
              <a
                className="underline decoration-dotted hover:text-foreground"
                href="https://arxiv.org/abs/2510.13857"
                target="_blank"
                rel="noreferrer"
              >
                arXiv:2510.13857v1
              </a>
              ), proposing an architecture they call <strong className="text-foreground">ArbiterOS</strong>:
              a neuro-symbolic OS where a <em>Symbolic Governor</em> (the kernel)
              arbitrates a <em>Probabilistic CPU</em> (an LLM) through a non-bypassable
              Arbiter Loop, governed by an Agent Constitution Framework.
            </p>
            <p>
              BIZRA was already three years into building exactly that
              architecture in Rust. The five canonical invariants map directly:
            </p>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardContent className="p-4">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="text-left p-2 font-medium">ArbiterOS paper</th>
                    <th className="text-left p-2 font-medium">BIZRA (shipped, Rust)</th>
                  </tr>
                </thead>
                <tbody className="text-foreground/80">
                  <tr className="border-t border-border/50">
                    <td className="p-2">Kernel-as-Governor (Symbolic Governor)</td>
                    <td className="p-2"><code className="font-mono bg-muted/30 px-1 rounded">bizra-cognition::admissibility_freeze_v1</code></td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="p-2">Probabilistic CPU (LLM)</td>
                    <td className="p-2 italic">Not yet wired (HANDOVER §10 known gap)</td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="p-2">Arbiter Loop (non-bypassable)</td>
                    <td className="p-2"><code className="font-mono bg-muted/30 px-1 rounded">runtime::submit_mission</code></td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="p-2">Agent Constitution Framework (ACF)</td>
                    <td className="p-2">5 invariants: ZANN_ZERO, CLAIM_MUST_BIND, RIBA_ZERO, NO_SHADOW_STATE, IHSAN_FLOOR</td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="p-2">Managed State</td>
                    <td className="p-2"><code className="font-mono bg-muted/30 px-1 rounded">sovereign_state/dema_cache/</code></td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="p-2">Flight Data Recorder</td>
                    <td className="p-2"><code className="font-mono bg-muted/30 px-1 rounded">ReceiptChain</code> (BLAKE3 hash-chain)</td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="p-2">EDLC (4 phases)</td>
                    <td className="p-2">Autopoietic Loop (7 phases)</td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="p-2">Gradient of Verification (3 levels)</td>
                    <td className="p-2">Ihsan 4-tier (0.90 CI / 0.95 Prod / 0.99 Strict / 1.0 Runtime)</td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="p-2">Cognitive IDE (§8.8 vision)</td>
                    <td className="p-2 italic">Horizon — DEMA Desktop Overlay</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50 bg-muted/10">
          <CardContent className="p-4 text-sm text-muted-foreground space-y-3">
            <p>
              The paper is <strong className="text-foreground">independent academic validation</strong>,
              not source material. BIZRA's architecture was frozen before the
              paper existed. That BIZRA independently reached the same
              conclusions as a research lab 6 months later is evidence of the
              architecture's structural correctness, not borrowed authority.
            </p>
            <p className="text-xs italic">
              A fuller written mapping is drafted in Mumo's operator memory
              (<code className="font-mono bg-muted/30 px-1 rounded">reference_arbiteros_paper.md</code>)
              and will be published on this branch once it clears review.
            </p>
          </CardContent>
        </Card>

        <footer className="text-center text-[11px] text-muted-foreground/70 italic pt-8 border-t border-border/50">
          <p>Close it. Prove it. Reveal it.</p>
        </footer>
      </div>
    </div>
  );
}
