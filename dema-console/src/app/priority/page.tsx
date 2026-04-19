import { PendingArtifact } from "@/components/pending-artifact";

export const metadata = {
  title: "BIZRA Proof-of-Priority",
  description:
    "Cryptographically anchored record that BIZRA's architecture predates its academic convergence (arXiv:2510.13857).",
};

export default function PriorityPage() {
  return (
    <PendingArtifact
      title="Proof-of-Priority manifest (generator)"
      summary="A script that produces a signed JSON manifest binding: BIZRA's earliest repo commits, SHA-256 of the arXiv:2510.13857v1 reference paper, and an Ed25519 signature from the BIZRA witness identity. Any skeptical stranger can recompute the paper hash, check the commit SHAs against GitHub, and verify the signature — independently confirming that BIZRA's architecture preceded the paper. At T=0 the artifact ships unsigned; Ed25519 signing is a follow-up step."
      sourcePath="scripts/generate-proof-of-priority.sh"
      sourceBranch="cycle-8/seal-primitive-days-1-2"
    />
  );
}
