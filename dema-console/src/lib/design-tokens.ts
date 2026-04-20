/**
 * BIZRA Design Tokens — Single Source of Truth
 * =============================================
 *
 * Runtime constants for colours, typography, spacing, and visual effects
 * used across DEMA face surfaces. Kept as TypeScript (not CSS-only) so
 * React components can reference them programmatically (e.g. framer-motion
 * animation targets, inline SVG gradients, data visualisations).
 *
 * CSS-level equivalents live in `src/app/globals.css` under the `@theme`
 * block for Tailwind v4 utility generation. Values here MUST match the
 * CSS vars one-to-one — any change requires updating both.
 *
 * Source: BIZRA Brand Identity v2 (Genesis Gold + Celestial Navy).
 * Canon: see `docs/design/CANON-TERMS.md` §07 for brand phrase canon.
 */

// ─────────────────────────────────────────────────────────────
// Colour — Genesis Gold / Celestial Navy / Bizra Emerald
// ─────────────────────────────────────────────────────────────
//
// Gold scale: from deep ember (900) to pale luminance (100). 500 is the
// canonical accent. Used for highlights, actions, and sacred-geometry
// strokes.
//
// Navy scale: from true-void (900) to near-black (800). Used for
// backgrounds and void layering.
//
// Bizra emerald: single semantic hue for "live / active / sovereign"
// state indicators. Sparingly applied; never a brand primary.

export const GOLD = {
  100: "#F9F1D8",
  300: "#E6D5A6",
  400: "#D4B875",
  500: "#C9A962",
  600: "#B08D45",
  900: "#8A6B2E",
} as const;

export const NAVY = {
  800: "#0A1628",
  900: "#050B14",
  void: "#030810",
} as const;

export const BIZRA_EMERALD = "#10b981" as const;

// ─────────────────────────────────────────────────────────────
// Typography families
// ─────────────────────────────────────────────────────────────
//
// Playfair Display: spiritual / editorial headings. Never inline body.
// Inter:            UI, data, body. Invisible good design.
// Amiri:            Arabic wordmarks and quotations. Serifed.
// JetBrains Mono:   code, hashes, receipt identifiers, mono labels.

export const FONTS = {
  serif: "'Playfair Display', Georgia, serif",
  sans: "'Inter', system-ui, sans-serif",
  arabic: "'Amiri', serif",
  mono: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
} as const;

// ─────────────────────────────────────────────────────────────
// Tracking (letter-spacing) — multi-file usage
// ─────────────────────────────────────────────────────────────

export const TRACKING = {
  heading: "0.15em",  // Serif headings
  label: "0.2em",     // Small uppercase labels
  eyebrow: "0.3em",   // Eyebrow labels above hero
  micro: "0.4em",     // Smallest category labels
} as const;

// ─────────────────────────────────────────────────────────────
// Visual effects
// ─────────────────────────────────────────────────────────────

export const EFFECTS = {
  glowGold: "drop-shadow(0 0 30px rgba(201,169,98,0.5))",
  glowGoldSubtle: "drop-shadow(0 0 15px rgba(201,169,98,0.3))",
  glassFill: "rgba(255,255,255,0.03)",
  glassBorder: "rgba(255,255,255,0.05)",
  glassBlur: "blur(10px)",
} as const;

// ─────────────────────────────────────────────────────────────
// Semantic status (maps to CANON-TERMS.md §03 phase taxonomy)
// ─────────────────────────────────────────────────────────────
//
// Use these for status badges in the face. Canonical names only
// (retired legacy labels PROVEN/VALIDATED/DEFERRED must NOT appear).

export const PHASE_TIER = {
  VERIFIED: {
    label: "Verified",
    color: BIZRA_EMERALD,
    description: "Shipped, receipt-backed, physically audited",
  },
  MEASURED: {
    label: "Measured",
    color: "#3b82f6", // blue-500
    description: "Quantified; methodology visible",
  },
  DERIVED: {
    label: "Derived",
    color: GOLD[500],
    description: "Architected; spec or diagram exists",
  },
  PLANNED: {
    label: "Planned",
    color: "#a855f7", // purple-500
    description: "Roadmap; scoped but not built",
  },
} as const;

export type PhaseTier = keyof typeof PHASE_TIER;
