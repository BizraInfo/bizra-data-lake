/**
 * BIZRA DDAGI OS — Astrolabe Console Design Tokens
 *
 * CANONICAL SOURCE for all visual tokens in the frontend.
 * Mirrors core/integration/constants.py for constitutional thresholds.
 *
 * Color Provenance:
 *   Gold  #C9A962 — Genesis Gold. Canonical across all surfaces.
 *   Navy  #0A1628 — Canonical bg per brand system (promoted from #040B14).
 *                    Phase 42 spec used #0A0B0F (deprecated filedfs era).
 *   Raised/Card/Hover derived from canonical navy via lightness steps.
 *
 * Any color change MUST update this file AND the provenance note above.
 */

// ═══ Constitutional Thresholds (synced with constants.py) ═══
export const THRESHOLDS = {
  IHSAN_PRODUCTION: 0.95,
  IHSAN_CI: 0.9,
  IHSAN_STRICT: 0.99,
  IHSAN_RUNTIME: 1.0,
  IHSAN_CONSENSUS: 0.99,
  SNR_MINIMUM: 0.85,
  SNR_T1: 0.95,
  SNR_T0: 0.98,
  ADL_GINI: 0.35,
  ZAKAT_RATE: 0.025,
  SEED_SUPPLY_CAP_PER_YEAR: 1_000_000,
} as const;

// ═══ Color Palette ═══
export const color = {
  // Obsidian Depth — canonical navy #0A1628
  bg: '#0A1628',
  bgRaised: '#0E1E34',
  bgCard: '#122640',
  bgHover: '#162E4C',

  // Living Gold — canonical #C9A962
  gold: '#C9A962',
  goldBright: '#E2CFA0',
  goldDim: '#8B7340',
  goldGlow: 'rgba(201,169,98,.12)',
  goldLine: 'rgba(201,169,98,.15)',

  // Constellation
  emerald: '#34D399',
  sapphire: '#60A5FA',
  ruby: '#F87171',
  amethyst: '#A78BFA',
  amber: '#FBBF24',
  cyan: '#22D3EE',
  flame: '#FB923C',
  rose: '#FB7185',

  // Text
  text: '#E8E4DB',
  muted: 'rgba(232,228,219,.65)',
  dim: 'rgba(232,228,219,.38)',
  ghost: 'rgba(232,228,219,.18)',
  line: 'rgba(255,255,255,.06)',
} as const;

// ═══ Typography ═══
export const font = {
  display: "'Cormorant Garamond', Georgia, serif",
  label: "'Cinzel', serif",
  mono: "'IBM Plex Mono', 'Menlo', monospace",
  arabic: "'Amiri', serif",
} as const;

// ═══ Tier System ═══
export const TIERS = ['Novice', 'Apprentice', 'Adept', 'Expert', 'Master', 'Grandmaster'] as const;
export type Tier = (typeof TIERS)[number];

export const TIER_COLORS: Record<number, string> = {
  0: '#6B7280',
  1: color.sapphire,
  2: color.emerald,
  3: color.amethyst,
  4: color.amber,
  5: color.gold,
};

// ═══ Human Lifecycle Stages ═══
export const STAGES = [
  { name: 'Seed',       low: 0,    high: 0.10, desc: 'Identity created. Potential infinite.' },
  { name: 'Node',       low: 0.10, high: 0.20, desc: 'First mission completed.' },
  { name: 'Apprentice', low: 0.20, high: 0.35, desc: 'Building habits.' },
  { name: 'Builder',    low: 0.35, high: 0.55, desc: 'Compiled first reflex.' },
  { name: 'Verifier',   low: 0.55, high: 0.70, desc: 'Trusted to attest others.' },
  { name: 'Mentor',     low: 0.70, high: 0.85, desc: 'Skills published.' },
  { name: 'Catalyst',   low: 0.85, high: 1.0,  desc: 'Network multiplier.' },
] as const;

export type StageName = (typeof STAGES)[number]['name'];

export function getStage(sovereignty: number) {
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (sovereignty >= STAGES[i].low) return STAGES[i];
  }
  return STAGES[0];
}
