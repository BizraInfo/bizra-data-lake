/**
 * Tokens & Design System Tests — Constitutional Thresholds + Stage System
 * =========================================================================
 * Verifies that the canonical design tokens and constitutional thresholds
 * maintain their contract with core/integration/constants.py.
 */

import { describe, it, expect } from 'vitest';
import { THRESHOLDS, color, font, TIERS, STAGES, getStage, TIER_COLORS } from '../tokens';

// ═══ CONSTITUTIONAL THRESHOLDS ═══

describe('Constitutional Thresholds', () => {
  it('IHSAN_PRODUCTION is 0.95 (synced with constants.py)', () => {
    expect(THRESHOLDS.IHSAN_PRODUCTION).toBe(0.95);
  });

  it('IHSAN_CI is 0.9', () => {
    expect(THRESHOLDS.IHSAN_CI).toBe(0.9);
  });

  it('IHSAN_STRICT is 0.99', () => {
    expect(THRESHOLDS.IHSAN_STRICT).toBe(0.99);
  });

  it('IHSAN_RUNTIME is 1.0', () => {
    expect(THRESHOLDS.IHSAN_RUNTIME).toBe(1.0);
  });

  it('IHSAN_CONSENSUS is 0.99', () => {
    expect(THRESHOLDS.IHSAN_CONSENSUS).toBe(0.99);
  });

  it('SNR_MINIMUM is 0.85', () => {
    expect(THRESHOLDS.SNR_MINIMUM).toBe(0.85);
  });

  it('ZAKAT_RATE is exactly 2.5%', () => {
    expect(THRESHOLDS.ZAKAT_RATE).toBe(0.025);
  });

  it('SEED_SUPPLY_CAP_PER_YEAR is 1,000,000', () => {
    expect(THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR).toBe(1_000_000);
  });

  it('ADL_GINI is 0.35', () => {
    expect(THRESHOLDS.ADL_GINI).toBe(0.35);
  });

  it('threshold ordering: CI < PRODUCTION < STRICT ≤ CONSENSUS ≤ RUNTIME', () => {
    expect(THRESHOLDS.IHSAN_CI).toBeLessThan(THRESHOLDS.IHSAN_PRODUCTION);
    expect(THRESHOLDS.IHSAN_PRODUCTION).toBeLessThan(THRESHOLDS.IHSAN_STRICT);
    expect(THRESHOLDS.IHSAN_STRICT).toBeLessThanOrEqual(THRESHOLDS.IHSAN_CONSENSUS);
    expect(THRESHOLDS.IHSAN_CONSENSUS).toBeLessThanOrEqual(THRESHOLDS.IHSAN_RUNTIME);
  });
});

// ═══ COLOR PALETTE ═══

describe('Color Palette: Canonical Values', () => {
  it('bg is canonical navy #0A1628', () => {
    expect(color.bg).toBe('#0A1628');
  });

  it('gold is canonical Genesis Gold #C9A962', () => {
    expect(color.gold).toBe('#C9A962');
  });

  it('all color tokens are non-empty strings', () => {
    for (const [key, value] of Object.entries(color)) {
      expect(value, `color.${key} should be non-empty`).toBeTruthy();
      expect(typeof value).toBe('string');
    }
  });

  it('hex colors are valid format', () => {
    const hexPattern = /^#[0-9A-Fa-f]{6}$/;
    const hexColors = ['bg', 'bgRaised', 'bgCard', 'bgHover', 'gold', 'goldBright', 'goldDim',
      'emerald', 'sapphire', 'ruby', 'amethyst', 'amber', 'cyan', 'flame', 'rose', 'text'] as const;
    for (const key of hexColors) {
      expect(color[key], `color.${key} should be valid hex`).toMatch(hexPattern);
    }
  });
});

// ═══ TYPOGRAPHY ═══

describe('Typography Tokens', () => {
  it('display font includes serif fallback', () => {
    expect(font.display).toContain('serif');
  });

  it('mono font includes monospace fallback', () => {
    expect(font.mono).toContain('monospace');
  });

  it('arabic font is Amiri', () => {
    expect(font.arabic).toContain('Amiri');
  });
});

// ═══ TIER SYSTEM ═══

describe('Tier System', () => {
  it('has 6 tiers from Novice to Grandmaster', () => {
    expect(TIERS).toHaveLength(6);
    expect(TIERS[0]).toBe('Novice');
    expect(TIERS[5]).toBe('Grandmaster');
  });

  it('every tier has a color mapping', () => {
    for (let i = 0; i < TIERS.length; i++) {
      expect(TIER_COLORS[i]).toBeDefined();
      expect(typeof TIER_COLORS[i]).toBe('string');
    }
  });
});

// ═══ STAGE SYSTEM (getStage) ═══

describe('Stage System: Lifecycle Stages', () => {
  it('has 7 stages', () => {
    expect(STAGES).toHaveLength(7);
  });

  it('stages cover the full [0, 1] range', () => {
    expect(STAGES[0].low).toBe(0);
    expect(STAGES[STAGES.length - 1].high).toBe(1.0);
  });

  it('stages are contiguous (no gaps)', () => {
    for (let i = 1; i < STAGES.length; i++) {
      expect(STAGES[i].low).toBe(STAGES[i - 1].high);
    }
  });

  it('getStage(0) returns Seed', () => {
    expect(getStage(0).name).toBe('Seed');
  });

  it('getStage(0.15) returns Node', () => {
    expect(getStage(0.15).name).toBe('Node');
  });

  it('getStage(0.25) returns Apprentice', () => {
    expect(getStage(0.25).name).toBe('Apprentice');
  });

  it('getStage(0.5) returns Builder', () => {
    expect(getStage(0.5).name).toBe('Builder');
  });

  it('getStage(0.6) returns Verifier', () => {
    expect(getStage(0.6).name).toBe('Verifier');
  });

  it('getStage(0.75) returns Mentor', () => {
    expect(getStage(0.75).name).toBe('Mentor');
  });

  it('getStage(0.9) returns Catalyst', () => {
    expect(getStage(0.9).name).toBe('Catalyst');
  });

  it('getStage at exact boundaries picks correct stage', () => {
    expect(getStage(0.10).name).toBe('Node');       // low boundary
    expect(getStage(0.35).name).toBe('Builder');     // low boundary
    expect(getStage(0.85).name).toBe('Catalyst');    // low boundary
  });

  it('getStage for negative returns Seed', () => {
    expect(getStage(-1).name).toBe('Seed');
  });

  it('every stage has a non-empty description', () => {
    for (const stage of STAGES) {
      expect(stage.desc.length).toBeGreaterThan(0);
    }
  });
});
