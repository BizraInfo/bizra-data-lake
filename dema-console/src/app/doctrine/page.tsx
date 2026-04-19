import { PendingArtifact } from "@/components/pending-artifact";

export const metadata = {
  title: "BIZRA First Fire Doctrine",
  description:
    "The launch-specific doctrine. Operator reality, one-bullet logic, Four-Modality preflight, T=0 gate list, Rest Clause.",
};

export default function DoctrinePage() {
  return (
    <PendingArtifact
      title="BIZRA First Fire Doctrine (v1 DRAFT)"
      summary="The launch-specific doctrine calibrated to actual operator condition (solo, exhausted, one-shot distribution bullet). Contains: operator reality, one-bullet logic, Four-Modality preflight checklist, bullet-target selection (DEMA seals reality · Organize is the first proof), witness-node closure plan, 12-day fire plan with live status, T=0 go/no-go gate list, 10 explicit non-goals, and the Rest Clause."
      sourcePath="docs/cycle-8/FIRST-FIRE-DOCTRINE-v1.md"
      sourceBranch="cycle-8/seal-primitive-days-1-2"
    />
  );
}
