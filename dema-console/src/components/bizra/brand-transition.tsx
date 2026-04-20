"use client";

/**
 * BIZRA Brand Transition — full-screen morph between surfaces
 * ===========================================================
 *
 * Ported from the Z.ai workspace shell (2026-04) without modification.
 * Clean timer-driven 4-phase state machine with proper cleanup.
 *
 * Phases (each 400ms, total 1200ms):
 *   0 → 1: Contract (brand page shrinks, background darkens)
 *   1 → 2: Morph (logo center stage, gold ring expands)
 *   2 → 3: Expand (logo compacts, "Entering Sovereign Mode" fades in)
 *   3   : Completion callback fires
 *
 * Depends on `@/components/bizra/brand-logo` (Phase 1 port).
 * Canon: `docs/design/CANON-TERMS.md` §07.
 */

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BrandLogo, BrandLogoCompact } from "@/components/bizra/brand-logo";

interface BrandTransitionProps {
  isTransitioning: boolean;
  onComplete: () => void;
}

export function BrandTransition({ isTransitioning, onComplete }: BrandTransitionProps) {
  const [phase, setPhase] = useState<0 | 1 | 2 | 3>(0);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    if (!isTransitioning) return;

    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];

    timersRef.current.push(setTimeout(() => setPhase(1), 0));
    timersRef.current.push(setTimeout(() => setPhase(2), 400));
    timersRef.current.push(setTimeout(() => setPhase(3), 800));
    timersRef.current.push(setTimeout(() => onComplete(), 1200));

    return () => {
      timersRef.current.forEach(clearTimeout);
      timersRef.current = [];
    };
  }, [isTransitioning, onComplete]);

  return (
    <AnimatePresence>
      {isTransitioning && (
        <motion.div
          className="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.1 }}
        >
          <motion.div
            className="absolute inset-0"
            initial={{ backgroundColor: "#050B14" }}
            animate={{ backgroundColor: "#030810" }}
            transition={{ duration: 1.0, ease: "easeInOut" }}
          />

          <motion.div
            className="absolute inset-0"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            style={{
              background:
                "radial-gradient(circle at center, transparent 0%, rgba(0,0,0,0.4) 100%)",
            }}
          />

          <AnimatePresence>
            {phase >= 2 && (
              <motion.div
                className="absolute"
                initial={{ width: 0, height: 0, borderRadius: "50%", opacity: 0.6 }}
                animate={{ width: 600, height: 600, opacity: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                style={{
                  border: "1px solid rgba(201, 169, 98, 0.3)",
                  top: "50%",
                  left: "50%",
                  x: "-50%",
                  y: "-50%",
                }}
              />
            )}
          </AnimatePresence>

          <AnimatePresence>
            {phase >= 2 && (
              <motion.div
                className="absolute"
                initial={{ width: 0, height: 0, borderRadius: "50%", opacity: 0.3 }}
                animate={{ width: 800, height: 800, opacity: 0 }}
                transition={{ duration: 1.0, ease: "easeOut", delay: 0.1 }}
                style={{
                  border: "1px solid rgba(201, 169, 98, 0.15)",
                  top: "50%",
                  left: "50%",
                  x: "-50%",
                  y: "-50%",
                }}
              />
            )}
          </AnimatePresence>

          <motion.div
            className="relative z-10"
            initial={{
              scale: 0.8,
              opacity: 0,
              filter: "drop-shadow(0 0 0px rgba(201,169,98,0))",
            }}
            animate={
              phase < 2
                ? {
                    scale: 1,
                    opacity: 1,
                    filter: "drop-shadow(0 0 30px rgba(201,169,98,0.5))",
                  }
                : {
                    scale: 0.25,
                    opacity: 1,
                    filter: "drop-shadow(0 0 8px rgba(201,169,98,0.3))",
                  }
            }
            transition={{
              duration: phase < 2 ? 0.4 : 0.4,
              ease: [0.16, 1, 0.3, 1],
            }}
          >
            <motion.div
              initial={{ scale: 1 }}
              animate={{ scale: phase >= 2 ? 0 : 1 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
            >
              {phase < 3 ? (
                <BrandLogo size={200} animate={true} showConstruction={false} />
              ) : (
                <div className="flex items-center justify-center">
                  <BrandLogoCompact size={48} />
                </div>
              )}
            </motion.div>
          </motion.div>

          <AnimatePresence>
            {phase >= 2 && phase < 3 && (
              <motion.div
                className="absolute z-10"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                transition={{ duration: 0.3 }}
                style={{ top: "calc(50% + 80px)" }}
              >
                <span
                  className="text-[10px] font-mono uppercase tracking-[0.3em]"
                  style={{ color: "#C9A962" }}
                >
                  Entering Sovereign Mode
                </span>
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {phase >= 3 && (
              <motion.div
                className="absolute z-10"
                initial={{ opacity: 0, scaleX: 0 }}
                animate={{ opacity: 1, scaleX: 1 }}
                transition={{ duration: 0.3 }}
                style={{
                  width: 120,
                  height: 1,
                  top: "calc(50% + 36px)",
                  left: "50%",
                  x: "-50%",
                  background:
                    "linear-gradient(90deg, transparent, rgba(201,169,98,0.3), transparent)",
                }}
              />
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
