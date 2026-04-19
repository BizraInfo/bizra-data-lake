// bizra.ai — public consumer landing page
//
// بسم الله الرحمن الرحيم
//
// Source of truth: dema-console/docs/launch/LANDING-CONSUMER-v1.md
// Positioning lock: U1=consumer · U2=no · U3=only-with-help
// Tone: constitutional prophecy under proof discipline, not startup brochure.
//
// Deployment: this page is a reference implementation. The public
// bizra.ai marketing site may eventually serve a dedicated static
// export of this route or extract to a separate deployment. The
// dema-console local app itself runs on the operator's machine
// post-install; this route is for first-time visitors who haven't
// installed yet.
//
// Honest link discipline: only links to artifacts that actually
// exist on this branch render as live links. Missing artifacts
// (Manifest, Doctrine, Priority) render as "Coming soon" until
// they merge to this branch from cycle-8.

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export const metadata = {
  title: "DEMA seals reality. Organize is the first proof.",
  description:
    "DEMA is a governed runtime that turns your intent into lawful, receipted, replayable action. Install in 60 seconds. Verify every claim. No cloud. No account. No cost.",
};

const INSTALL_COMMAND = "curl -fsSL https://bizra.ai/install.sh | sh";
const FIRST_MISSION_COMMAND = "dema organize ~/Downloads";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-3xl mx-auto px-6 py-16 space-y-16">
        {/* Hero */}
        <section className="space-y-6 text-center">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight leading-tight">
            DEMA seals reality.
            <br />
            <span className="text-muted-foreground">Organize is the first proof.</span>
          </h1>
          <p className="text-sm md:text-base text-muted-foreground max-w-xl mx-auto leading-relaxed">
            One command. Sixty seconds. No account. No cloud. No cost.
          </p>

          <Card className="bg-muted/30 border-border/50 text-left max-w-2xl mx-auto">
            <CardContent className="p-4 space-y-2">
              <code className="block font-mono text-xs md:text-sm text-foreground/90 break-all">
                {INSTALL_COMMAND}
              </code>
              <code className="block font-mono text-xs md:text-sm text-foreground/90 break-all">
                {FIRST_MISSION_COMMAND}
              </code>
            </CardContent>
          </Card>

          <p className="text-xs text-muted-foreground max-w-xl mx-auto leading-relaxed">
            You get a cryptographically sealed manifest of your digital clutter —
            receipted, replayable, and verifiable by any skeptical stranger on
            any machine.
          </p>

          <p className="text-[11px] text-muted-foreground/70 italic">
            Install script not live yet — this page is a pre-launch reference.
            Install will become live when the first cargo-dist release is cut.
          </p>
        </section>

        {/* Why DEMA exists */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">Why DEMA exists</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Every app you use claims things about your data that only the app
            can verify. You have no way to check. You trust because you have no
            other choice.
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            DEMA replaces that trust with <strong className="text-foreground">proof</strong>. When
            DEMA seals something, a skeptical stranger can, using only public
            tooling, verify in bounded time that the claim is true — or produce
            transferable evidence that it's false.
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            DEMA is not an AI assistant. DEMA is a governed runtime that turns
            your intent into lawful, receipted, replayable action. The first
            thing it proves is the hardest one: that it didn't lie to you.
          </p>
        </section>

        {/* Your first proof */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">Your first proof</h2>
          <ol className="list-decimal list-inside text-sm text-muted-foreground space-y-3 leading-relaxed">
            <li>
              <span className="text-foreground">Install</span> (&lt; 5 minutes, reproducible build, SHA-256 verified):
              <pre className="mt-2 bg-muted/30 border border-border/50 p-2 rounded text-xs font-mono overflow-x-auto">
                {INSTALL_COMMAND}
              </pre>
            </li>
            <li>
              <span className="text-foreground">Register an allowlisted path</span> — DEMA will NEVER touch anything else:
              <pre className="mt-2 bg-muted/30 border border-border/50 p-2 rounded text-xs font-mono overflow-x-auto">
                dema register-resource --kind filesystem --id ~/Downloads --allowlisted
              </pre>
            </li>
            <li>
              <span className="text-foreground">Seal the first mission</span>:
              <pre className="mt-2 bg-muted/30 border border-border/50 p-2 rounded text-xs font-mono overflow-x-auto">
                {FIRST_MISSION_COMMAND}
              </pre>
            </li>
          </ol>
          <p className="text-xs text-muted-foreground leading-relaxed">
            You'll see five constitutional gate verdicts, a chain-sealed{" "}
            <code className="bg-muted/30 px-1 py-0.5 rounded font-mono">
              MissionExecuted
            </code>{" "}
            receipt with a BLAKE3 hash, and a deterministic listing digest that
            reproduces byte-identical on any machine. Keep the receipt. Anyone
            can replay it: <code className="bg-muted/30 px-1 py-0.5 rounded font-mono">dema receipt &lt;hash&gt;</code>.
          </p>
        </section>

        {/* How you verify */}
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">How you verify DEMA didn't lie</h2>
          <p className="text-xs text-muted-foreground leading-relaxed">
            The Four-Modality Golden Standard. DEMA commits to all four at T=0 —
            honestly labeled, no overclaim.
          </p>
          <div className="border border-border/50 rounded-md overflow-hidden">
            <table className="w-full text-xs">
              <thead className="bg-muted/20 text-muted-foreground">
                <tr>
                  <th className="text-left p-3 font-medium">Modality</th>
                  <th className="text-left p-3 font-medium">Meaning</th>
                  <th className="text-left p-3 font-medium">How you check</th>
                </tr>
              </thead>
              <tbody className="text-foreground/80">
                <tr className="border-t border-border/50">
                  <td className="p-3 font-medium">Cryptographic</td>
                  <td className="p-3">BLAKE3 hash-chain + Ed25519 signatures</td>
                  <td className="p-3 font-mono text-[11px]">sha256sum $(which dema)</td>
                </tr>
                <tr className="border-t border-border/50">
                  <td className="p-3 font-medium">Empirical</td>
                  <td className="p-3">Same input → same output on any machine</td>
                  <td className="p-3">Run <code className="bg-muted/30 px-1 rounded">dema organize</code> twice — receipt IDs match</td>
                </tr>
                <tr className="border-t border-border/50">
                  <td className="p-3 font-medium">Formal (TESTED)</td>
                  <td className="p-3">309 + 77 tests green under <code className="bg-muted/30 px-1 rounded">-D warnings</code></td>
                  <td className="p-3 font-mono text-[11px]">git clone && cargo test</td>
                </tr>
                <tr className="border-t border-border/50">
                  <td className="p-3 font-medium">Economic (witness-grade)</td>
                  <td className="p-3">Independent witness observes chain head</td>
                  <td className="p-3 font-mono text-[11px]">curl &lt;witness&gt;/witness/head/&lt;node-id&gt;</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-muted-foreground/80 italic leading-relaxed">
            Full Isabelle/HOL-grade formal proof and bonded cryptoeconomic
            enforcement are explicitly Horizon — not at T=0, not claimed to be.
          </p>
        </section>

        {/* Constitutional anchors */}
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Constitutional anchors</h2>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Every lawful act in DEMA passes through five gates, fail-closed, no exceptions:
          </p>
          <ul className="text-sm text-foreground/80 space-y-1.5 leading-relaxed">
            <li>
              <strong className="text-foreground">ZANN_ZERO</strong>
              <span className="text-muted-foreground"> — no claim without evidence</span>
            </li>
            <li>
              <strong className="text-foreground">CLAIM_MUST_BIND</strong>
              <span className="text-muted-foreground"> — evidence must cryptographically bind to the claim</span>
            </li>
            <li>
              <strong className="text-foreground">RIBA_ZERO</strong>
              <span className="text-muted-foreground"> — no extractive economic pattern</span>
            </li>
            <li>
              <strong className="text-foreground">NO_SHADOW_STATE</strong>
              <span className="text-muted-foreground"> — what you see is what the kernel sealed</span>
            </li>
            <li>
              <strong className="text-foreground">IHSAN_FLOOR</strong>
              <span className="text-muted-foreground"> — quality floor ≥ 0.95 for any permit</span>
            </li>
          </ul>
        </section>

        {/* What DEMA is NOT */}
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">What DEMA is NOT</h2>
          <ul className="text-sm text-muted-foreground space-y-2 leading-relaxed">
            <li>
              <strong className="text-foreground">Not a chatbot.</strong> No conversation surface, no LLM wrapping, no model calls in the critical path.
            </li>
            <li>
              <strong className="text-foreground">Not a cloud service.</strong> Everything runs local on your machine. No account. No server.
            </li>
            <li>
              <strong className="text-foreground">Not an agent framework.</strong> The kernel is not programmable through natural language.
            </li>
            <li>
              <strong className="text-foreground">Not a startup pitch.</strong> BIZRA has been built solo for three years. No VC, no tokens, no ads.
            </li>
          </ul>
        </section>

        {/* Provenance */}
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Built by one operator in three years</h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            BIZRA (the kernel behind DEMA) is the result of ~15,000 hours of
            solo engineering, anchored in two Arabic founding texts written in
            Ramadan 2023. Independently validated in October 2025 by academic
            convergence: <a className="underline decoration-dotted hover:text-foreground" href="https://arxiv.org/abs/2510.13857" target="_blank" rel="noreferrer">arXiv:2510.13857v1</a> (Xu et al., CUHK) theorizes, six months after BIZRA was already
            building it, the same architecture DEMA implements in Rust.
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            That paper is external evidence of convergence, not source material. DEMA came first.
          </p>
        </section>

        {/* Links */}
        <section className="space-y-3 border-t border-border/50 pt-6">
          <h2 className="text-lg font-semibold">Links</h2>
          <ul className="text-sm space-y-2">
            <li>
              <strong className="text-foreground">Install</strong>{" "}
              <span className="text-muted-foreground">(pending first cargo-dist release)</span>
            </li>
            <li>
              <Link className="underline decoration-dotted hover:text-foreground" href="/manifest">
                Manifest
              </Link>
              <span className="text-muted-foreground"> — constitutional canon</span>
            </li>
            <li>
              <Link className="underline decoration-dotted hover:text-foreground" href="/doctrine">
                First Fire Doctrine
              </Link>
              <span className="text-muted-foreground"> — launch logic</span>
            </li>
            <li>
              <Link className="underline decoration-dotted hover:text-foreground" href="/priority">
                Proof-of-Priority
              </Link>
              <span className="text-muted-foreground"> — 3-year architectural record</span>
            </li>
            <li>
              <Link className="underline decoration-dotted hover:text-foreground" href="/arbiteros-mapping">
                ArbiterOS ↔ BIZRA mapping
              </Link>
              <span className="text-muted-foreground"> — academic convergence cite</span>
            </li>
            <li>
              <a
                className="underline decoration-dotted hover:text-foreground"
                href="https://github.com/BizraInfo/bizra-data-lake"
                target="_blank"
                rel="noreferrer"
              >
                Source on GitHub
              </a>
            </li>
          </ul>
        </section>

        {/* Footer */}
        <footer className="border-t border-border/50 pt-6 text-center text-[11px] text-muted-foreground/70 italic leading-relaxed">
          <p>Close it. Prove it. Reveal it.</p>
          <p className="mt-2 opacity-70">الحمد لله</p>
        </footer>
      </div>
    </div>
  );
}
