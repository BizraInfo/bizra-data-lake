/**
 * Reward Engine Tests — PoI → SEED/BLOOM Pipeline
 * =================================================
 * Tests the spec-faithful reward calculation pipeline:
 *   Gate checks (Ihsān, SNR) → PoI composite → SEED minting → Zakat → BLOOM accrual
 *
 * Mirrors constitutional invariants from:
 *   core/proof_engine/poi_engine.py + core/token/mint.py
 */

import { describe, it, expect } from 'vitest';
import { calculateReward, simulateReward } from '../lib/reward-engine';
import type { PoIInput, SupplyContext } from '../lib/reward-engine';
import { THRESHOLDS } from '../tokens';

// ═══ HELPERS ═══

/** Valid input that passes all gates */
const VALID_INPUT: PoIInput = {
  contribution: 0.9,
  reach: 0.7,
  longevity: 0.8,
  ihsan: 0.97,
  snr: 0.92,
};

const DEFAULT_SUPPLY: SupplyContext = { yearlyMintedSeed: 0 };

// ═══ GATE TESTS ═══

describe('Reward Engine: Constitutional Gates', () => {
  it('rejects when Ihsān below production threshold (0.95)', () => {
    const input: PoIInput = { ...VALID_INPUT, ihsan: 0.94 };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_REJECT_IHSAN_BELOW_THRESHOLD');
    expect(receipt.grossSeed).toBe(0);
    expect(receipt.netSeed).toBe(0);
    expect(receipt.bloom).toBe(0);
  });

  it('accepts when Ihsān exactly at production threshold', () => {
    const input: PoIInput = { ...VALID_INPUT, ihsan: THRESHOLDS.IHSAN_PRODUCTION };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_OK');
    expect(receipt.netSeed).toBeGreaterThan(0);
  });

  it('rejects when SNR below minimum threshold (0.85)', () => {
    const input: PoIInput = { ...VALID_INPUT, snr: 0.84 };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_REJECT_SNR_BELOW_THRESHOLD');
    expect(receipt.grossSeed).toBe(0);
  });

  it('accepts when SNR exactly at minimum threshold', () => {
    const input: PoIInput = { ...VALID_INPUT, snr: THRESHOLDS.SNR_MINIMUM };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_OK');
  });

  it('Ihsān gate checked before SNR gate', () => {
    // Both below threshold — Ihsān should fail first
    const input: PoIInput = { ...VALID_INPUT, ihsan: 0.5, snr: 0.3 };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_REJECT_IHSAN_BELOW_THRESHOLD');
  });
});

// ═══ POI COMPOSITE SCORE ═══

describe('Reward Engine: PoI Composite Score', () => {
  it('computes weighted composite: 0.5α + 0.3β + 0.2γ', () => {
    const input: PoIInput = {
      contribution: 1.0, reach: 1.0, longevity: 1.0,
      ihsan: 0.97, snr: 0.92,
    };
    const receipt = calculateReward(input);
    // Composite = 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
    expect(receipt.poiScore).toBeCloseTo(1.0, 4);
  });

  it('zero contribution yields reduced score', () => {
    const input: PoIInput = {
      contribution: 0, reach: 0.7, longevity: 0.8,
      ihsan: 0.97, snr: 0.92,
    };
    const receipt = calculateReward(input);
    // Composite = 0.5*0 + 0.3*0.7 + 0.2*0.8 = 0.37
    expect(receipt.poiScore).toBeCloseTo(0.37, 4);
  });

  it('all zeros produce zero score but still passes gates', () => {
    const input: PoIInput = {
      contribution: 0, reach: 0, longevity: 0,
      ihsan: 0.97, snr: 0.92,
    };
    const receipt = calculateReward(input);
    expect(receipt.poiScore).toBe(0);
    expect(receipt.grossSeed).toBe(0);
    expect(receipt.reason).toBe('POI_OK');
  });
});

// ═══ SEED MINTING ═══

describe('Reward Engine: SEED Minting', () => {
  it('grossSeed = BASE_SEED_PER_POI × poiScore × ihsān', () => {
    const receipt = calculateReward(VALID_INPUT);
    const expectedPoI = 0.5 * 0.9 + 0.3 * 0.7 + 0.2 * 0.8; // 0.82
    const expectedGross = 1.0 * expectedPoI * 0.97;
    expect(receipt.grossSeed).toBeCloseTo(expectedGross, 4);
  });

  it('higher Ihsān yields more SEED (quality multiplier)', () => {
    const low = calculateReward({ ...VALID_INPUT, ihsan: 0.95 });
    const high = calculateReward({ ...VALID_INPUT, ihsan: 0.99 });
    expect(high.grossSeed).toBeGreaterThan(low.grossSeed);
  });
});

// ═══ ZAKAT INVARIANT ═══

describe('Reward Engine: Zakat (2.5% Constitutional)', () => {
  it('zakat is exactly 2.5% of gross SEED', () => {
    const receipt = calculateReward(VALID_INPUT);
    const expectedZakat = Math.round(receipt.grossSeed * THRESHOLDS.ZAKAT_RATE * 10000) / 10000;
    expect(receipt.zakatSeed).toBeCloseTo(expectedZakat, 4);
  });

  it('netSeed = grossSeed - zakatSeed', () => {
    const receipt = calculateReward(VALID_INPUT);
    expect(receipt.netSeed).toBeCloseTo(receipt.grossSeed - receipt.zakatSeed, 4);
  });

  it('zakat is zero when gross is zero', () => {
    const input: PoIInput = {
      contribution: 0, reach: 0, longevity: 0,
      ihsan: 0.97, snr: 0.92,
    };
    const receipt = calculateReward(input);
    expect(receipt.zakatSeed).toBe(0);
    expect(receipt.netSeed).toBe(0);
  });
});

// ═══ BLOOM ACCRUAL ═══

describe('Reward Engine: BLOOM Accrual', () => {
  it('bloom is 1% of net SEED', () => {
    const receipt = calculateReward(VALID_INPUT);
    const expectedBloom = Math.round(receipt.netSeed * 0.01 * 10000) / 10000;
    expect(receipt.bloom).toBeCloseTo(expectedBloom, 4);
  });

  it('bloom is zero when rejected', () => {
    const input: PoIInput = { ...VALID_INPUT, ihsan: 0.5 };
    const receipt = calculateReward(input);
    expect(receipt.bloom).toBe(0);
  });
});

// ═══ SUPPLY CAP ═══

describe('Reward Engine: Supply Cap Enforcement', () => {
  it('rejects when yearly supply cap is exhausted', () => {
    const supply: SupplyContext = { yearlyMintedSeed: THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR };
    const receipt = calculateReward(VALID_INPUT, supply);
    expect(receipt.reason).toBe('POI_REJECT_SUPPLY_CAP_EXCEEDED');
  });

  it('caps gross SEED to remaining supply', () => {
    // Almost at cap — only 0.001 remaining
    const supply: SupplyContext = { yearlyMintedSeed: THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR - 0.001 };
    const receipt = calculateReward(VALID_INPUT, supply);
    expect(receipt.capHit).toBe(true);
    expect(receipt.grossSeed).toBeLessThanOrEqual(0.001);
  });

  it('no cap hit when supply is plentiful', () => {
    const receipt = calculateReward(VALID_INPUT, DEFAULT_SUPPLY);
    expect(receipt.capHit).toBe(false);
  });
});

// ═══ SIMULATION ═══

describe('Reward Engine: Offline Simulation', () => {
  it('simulateReward produces valid receipt for high Ihsān', () => {
    const receipt = simulateReward(0.97);
    expect(receipt.reason).toBe('POI_OK');
    expect(receipt.netSeed).toBeGreaterThan(0);
    expect(receipt.bloom).toBeGreaterThan(0);
  });

  it('simulateReward rejects low Ihsān', () => {
    const receipt = simulateReward(0.5);
    expect(receipt.reason).toBe('POI_REJECT_IHSAN_BELOW_THRESHOLD');
  });

  it('simulateReward SNR derives from Ihsān', () => {
    // ihsan = 0.95, snr = max(0.85, 0.95 - 0.05) = 0.90 — passes SNR gate
    const receipt = simulateReward(0.95);
    expect(receipt.reason).toBe('POI_OK');
  });

  it('simulateReward borderline SNR (ihsan = 0.89) fails', () => {
    // ihsan = 0.89 — fails Ihsān gate first (0.89 < 0.95)
    const receipt = simulateReward(0.89);
    expect(receipt.reason).toBe('POI_REJECT_IHSAN_BELOW_THRESHOLD');
  });
});

// ═══ INVARIANT: RECEIPT CONSISTENCY ═══

describe('Reward Engine: Receipt Invariants', () => {
  it('all monetary fields are non-negative', () => {
    const receipt = calculateReward(VALID_INPUT);
    expect(receipt.grossSeed).toBeGreaterThanOrEqual(0);
    expect(receipt.zakatSeed).toBeGreaterThanOrEqual(0);
    expect(receipt.netSeed).toBeGreaterThanOrEqual(0);
    expect(receipt.bloom).toBeGreaterThanOrEqual(0);
  });

  it('rejected receipts have all zero monetary fields', () => {
    const receipt = calculateReward({ ...VALID_INPUT, ihsan: 0.3 });
    expect(receipt.grossSeed).toBe(0);
    expect(receipt.zakatSeed).toBe(0);
    expect(receipt.netSeed).toBe(0);
    expect(receipt.bloom).toBe(0);
    expect(receipt.poiScore).toBe(0);
    expect(receipt.capHit).toBe(false);
  });

  it('netSeed ≤ grossSeed always', () => {
    const receipt = calculateReward(VALID_INPUT);
    expect(receipt.netSeed).toBeLessThanOrEqual(receipt.grossSeed);
  });

  it('poiScore in [0, 1]', () => {
    const receipt = calculateReward(VALID_INPUT);
    expect(receipt.poiScore).toBeGreaterThanOrEqual(0);
    expect(receipt.poiScore).toBeLessThanOrEqual(1);
  });
});
