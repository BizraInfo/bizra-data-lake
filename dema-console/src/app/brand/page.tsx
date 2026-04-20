"use client";

/**
 * /brand — BIZRA full brand identity showcase
 * =============================================
 *
 * Renders the canonical BrandIdentityPage component (ported with
 * 7 label reconciliations — see component docstring).
 *
 * Distinct from:
 *   - `/`          — operator console (mission workspace)
 *   - `/manifesto` — consumer landing with architecture pillars
 *   - `/brand`     — this: full identity system showcase
 *                    (typography, palette, Aegis, digital card)
 */

import { useRouter } from "next/navigation";
import { useState } from "react";
import { BrandIdentityPage } from "@/components/bizra/brand-page";
import { BrandTransition } from "@/components/bizra/brand-transition";

export default function BrandRoute() {
  const router = useRouter();
  const [transitioning, setTransitioning] = useState(false);

  const handleEnter = () => setTransitioning(true);
  const handleTransitionComplete = () => router.push("/");

  return (
    <>
      <BrandIdentityPage onEnter={handleEnter} isReturn={false} />
      <BrandTransition
        isTransitioning={transitioning}
        onComplete={handleTransitionComplete}
      />
    </>
  );
}
