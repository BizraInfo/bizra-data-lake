"use client";

/**
 * BIZRA Brand Logo — Seed of Life sacred geometry mark
 * ====================================================
 *
 * Ported from the DEMA workspace shell (Z.ai lineage, 2026-04).
 * Pure SVG + framer-motion animation; no imperative DOM.
 *
 * Three visual layers (inside → out):
 *   1. Construction circles  — the Seed of Life grid (6 + 1 centre)
 *   2. Manifested flower     — 6 petals rendered with gold gradient
 *   3. Nuqta                 — the central diamond dot (Tawhid)
 *
 * See `docs/design/CANON-TERMS.md` §07 for the brand canon. Colours
 * reference `src/lib/design-tokens.ts::GOLD`.
 */

import { motion } from "framer-motion";

const SEED_CIRCLES = [
  { cx: 0, cy: 0 },
  { cx: 0, cy: -40 },
  { cx: 34.6, cy: -20 },
  { cx: 34.6, cy: 20 },
  { cx: 0, cy: 40 },
  { cx: -34.6, cy: 20 },
  { cx: -34.6, cy: -20 },
];

const PETALS = [
  "M0 -40 Q20 -20 0 0 Q-20 -20 0 -40",
  "M34.6 -20 Q17.3 10 0 0 Q17.3 -10 34.6 -20",
  "M34.6 20 Q17.3 10 0 0 Q17.3 30 34.6 20",
  "M0 40 Q-20 20 0 0 Q20 20 0 40",
  "M-34.6 20 Q-17.3 10 0 0 Q-17.3 30 -34.6 20",
  "M-34.6 -20 Q-17.3 10 0 0 Q-17.3 -10 -34.6 -20",
];

interface BrandLogoProps {
  size?: number;
  className?: string;
  animate?: boolean;
  showConstruction?: boolean;
}

export function BrandLogo({
  size = 200,
  className = "",
  animate = true,
  showConstruction = true,
}: BrandLogoProps) {
  return (
    <svg
      viewBox="0 0 200 200"
      className={className}
      style={{ width: size, height: size }}
      overflow="visible"
    >
      <defs>
        <linearGradient id="goldGrad" x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#8A6B2E" stopOpacity={1} />
          <stop offset="50%" stopColor="#C9A962" stopOpacity={1} />
          <stop offset="100%" stopColor="#F9F1D8" stopOpacity={1} />
        </linearGradient>
        <linearGradient id="goldGradFade" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#C9A962" stopOpacity={0.15} />
          <stop offset="100%" stopColor="#C9A962" stopOpacity={0.02} />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <g transform="translate(100, 100)">
        {showConstruction &&
          SEED_CIRCLES.map((c, i) => (
            <motion.circle
              key={`seed-${i}`}
              cx={c.cx}
              cy={c.cy}
              r={40}
              fill="none"
              stroke={i === 0 ? "#C9A962" : "rgba(201, 169, 98, 0.4)"}
              strokeWidth={i === 0 ? 1.2 : 0.8}
              filter={i === 0 ? "url(#glow)" : undefined}
              initial={
                animate
                  ? { opacity: 0, scale: 0.8 }
                  : { opacity: i === 0 ? 1 : 0.4, scale: 1 }
              }
              animate={{ opacity: i === 0 ? 1 : 0.4, scale: 1 }}
              transition={{
                delay: animate ? 0.1 + i * 0.12 : 0,
                duration: 0.8,
                ease: [0.16, 1, 0.3, 1],
              }}
            />
          ))}

        {showConstruction && (
          <motion.circle
            cx={0}
            cy={0}
            r={80}
            fill="none"
            stroke="rgba(201, 169, 98, 0.08)"
            strokeWidth={0.5}
            strokeDasharray="4 4"
            initial={
              animate ? { opacity: 0, scale: 0.5 } : { opacity: 1, scale: 1 }
            }
            animate={{ opacity: 1, scale: 1 }}
            transition={{
              delay: animate ? 0.9 : 0,
              duration: 1,
              ease: "easeOut",
            }}
          />
        )}

        {PETALS.map((d, i) => (
          <motion.path
            key={`petal-${i}`}
            d={d}
            fill="none"
            stroke="url(#goldGrad)"
            strokeWidth={1.5}
            strokeLinecap="round"
            filter="url(#glow)"
            initial={
              animate
                ? { pathLength: 0, opacity: 0 }
                : { pathLength: 1, opacity: 0.9 }
            }
            animate={{ pathLength: 1, opacity: 0.9 }}
            transition={{
              delay: animate ? 1.2 + i * 0.1 : 0,
              duration: 1,
              ease: [0.16, 1, 0.3, 1],
            }}
          />
        ))}

        <motion.rect
          x={-3}
          y={-3}
          width={6}
          height={6}
          transform="rotate(45)"
          fill="url(#goldGrad)"
          filter="url(#glow)"
          initial={animate ? { opacity: 0, scale: 0 } : { opacity: 1, scale: 1 }}
          animate={
            animate
              ? {
                  opacity: [0, 1, 1],
                  scale: [0, 1.2, 1],
                }
              : { opacity: 1, scale: 1 }
          }
          transition={
            animate
              ? { delay: 1.8, duration: 0.6, ease: "easeOut", times: [0, 0.6, 1] }
              : { duration: 0 }
          }
        />
      </g>
    </svg>
  );
}

/** Compact static logo for nav / card / avatar use (no animation). */
export function BrandLogoCompact({ size = 48 }: { size?: number }) {
  return (
    <svg viewBox="0 0 100 100" style={{ width: size, height: size }}>
      <g stroke="#C9A962" strokeWidth={1.2} fill="none">
        <circle cx="50" cy="50" r="20" />
        <circle cx="50" cy="30" r="20" />
        <circle cx="67.3" cy="40" r="20" />
        <circle cx="67.3" cy="60" r="20" />
        <circle cx="50" cy="70" r="20" />
        <circle cx="32.7" cy="60" r="20" />
        <circle cx="32.7" cy="40" r="20" />
      </g>
    </svg>
  );
}
