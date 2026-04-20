"use client";

/**
 * /manifesto — BIZRA brand + architecture landing
 * ================================================
 *
 * Stand-alone route rendering the canonical LandingPage component
 * (ported from Z.ai workspace shell). Presents the Seed of Life
 * hero, Covenant, Architecture pillars, and Threshold CTA.
 *
 * Distinct from the operator console at `/` — this surface is the
 * brand / narrative entry, not the mission workspace.
 *
 * Entry from here goes back to `/` (the operator console).
 */

import { useRouter } from "next/navigation";
import { useState } from "react";
import { LandingPage } from "@/components/bizra/landing-page";
import { BrandTransition } from "@/components/bizra/brand-transition";

export default function ManifestoRoute() {
  const router = useRouter();
  const [transitioning, setTransitioning] = useState(false);

  const handleEnter = () => {
    setTransitioning(true);
  };

  const handleTransitionComplete = () => {
    router.push("/");
  };

  return (
    <>
      <LandingPage onEnter={handleEnter} isReturn={false} />
      <BrandTransition
        isTransitioning={transitioning}
        onComplete={handleTransitionComplete}
      />
    </>
  );
}
