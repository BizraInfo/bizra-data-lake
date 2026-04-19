import { PendingArtifact } from "@/components/pending-artifact";

export const metadata = {
  title: "BIZRA Manifest — North Star Canon",
  description:
    "The constitutional doctrine. 13 sections covering invariants, planes, lawful loop, truth labels, Golden Standard.",
};

export default function ManifestPage() {
  return (
    <PendingArtifact
      title="BIZRA Manifest — North Star Canon (v1 DRAFT)"
      summary="The constitutional doctrine. 13 sections: opening declaration, why BIZRA exists, canonical thesis, governing mandate, the 5 invariants, lawful runtime, surface doctrine, Node0 achievement, hidden organism, ecosystem horizon, truth labels, Golden Standard (witness-grade at T=0), final declaration."
      sourcePath="docs/cycle-8/MANIFEST-NORTH-STAR-v1.md"
      sourceBranch="cycle-8/seal-primitive-days-1-2"
    />
  );
}
