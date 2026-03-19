/**
 * Economic Contract Tests — buildReceipt + Thresholds
 * =====================================================
 * Tests the unified economic contract that governs PoI → SEED/BLOOM minting.
 * Mirrors the canonical pipeline in core/proof_engine/poi_engine.py.
 */

import { describe, it, expect } from 'vitest';
import { ECONOMIC_THRESHOLDS, POI_WEIGHTS, buildReceipt } from '../lib/economic';
import type { PoIFactors } from '../lib/economic';

// ═══ HELPERS ═══

const VALID_FACTORS: PoIFactors = {
  contribution: 0.9,
  reach: 0.7,
  longevity: 0.8,
  ihsan: 0.97,
  snr: 0.92,
};

// ═══ THRESHOLD CONSTANTS ═══

describe('Economic Thresholds', () => {
  it('MISSION_FLOOR is 0.85', () => {
    expect(ECONOMIC_THRESHOLDS.MISSION_FLOOR).toBe(0.85);
  });

  it('BLOOM_ELIGIBILITY is 0.90', () => {
    expect(ECONOMIC_THRESHOLDS.BLOOM_ELIGIBILITY).toBe(0.90);
  });

  it('MINTING_FLOOR is 0.95', () => {
    expect(ECONOMIC_THRESHOLDS.MINTING_FLOOR).toBe(0.95);
  });

  it('EXCELLENCE is 0.97', () => {
    expect(ECONOMIC_THRESHOLDS.EXCELLENCE).toBe(0.97);
  });

  it('ZAKAT_RATE is 2.5%', () => {
    expect(ECONOMIC_THRESHOLDS.ZAKAT_RATE).toBe(0.025);
  });

  it('COMMUNITY_POOL_SPLIT is 50%', () => {
    expect(ECONOMIC_THRESHOLDS.COMMUNITY_POOL_SPLIT).toBe(0.50);
  });

  it('SNR_MINIMUM is 0.85', () => {
    expect(ECONOMIC_THRESHOLDS.SNR_MINIMUM).toBe(0.85);
  });

  it('ordering: MISSION_FLOOR < BLOOM_ELIGIBILITY < MINTING_FLOOR ≤ EXCELLENCE', () => {
    expect(ECONOMIC_THRESHOLDS.MISSION_FLOOR).toBeLessThan(ECONOMIC_THRESHOLDS.BLOOM_ELIGIBILITY);
    expect(ECONOMIC_THRESHOLDS.BLOOM_ELIGIBILITY).toBeLessThan(ECONOMIC_THRESHOLDS.MINTING_FLOOR);
    expect(ECONOMIC_THRESHOLDS.MINTING_FLOOR).toBeLessThanOrEqual(ECONOMIC_THRESHOLDS.EXCELLENCE);
  });
});

// ═══ POI WEIGHTS ═══

describe('PoI Weights', () => {
  it('weights sum to 1.0', () => {
    const sum = POI_WEIGHTS.contribution + POI_WEIGHTS.reach + POI_WEIGHTS.longevity;
    expect(sum).toBeCloseTo(1.0, 10);
  });

  it('contribution has highest weight (0.5)', () => {
    expect(POI_WEIGHTS.contribution).toBe(0.5);
    expect(POI_WEIGHTS.contribution).toBeGreaterThan(POI_WEIGHTS.reach);
    expect(POI_WEIGHTS.contribution).toBeGreaterThan(POI_WEIGHTS.longevity);
  });
});

// ═══ buildReceipt: GATE CHECKS ═══

describe('buildReceipt: Gate Checks', () => {
  it('rejects Ihsān below MINTING_FLOOR (0.95)', () => {
    const factors: PoIFactors = { ...VALID_FACTORS, ihsan: 0.94 };
    const receipt = buildReceipt(factors);
    expect(receipt.reason).toBe('POI_REJECT_IHSAN_BELOW_THRESHOLD');
    expect(receipt.grossSeed).toBe(0);
  });

  it('rejects SNR below minimum (0.85)', () => {
    const factors: PoIFactors = { ...VALID_FACTORS, snr: 0.84 };
    const receipt = buildReceipt(factors);
    expect(receipt.reason).toBe('POI_REJECT_SNR_BELOW_THRESHOLD');
  });

  it('rejects when supply exhausted', () => {
    const receipt = buildReceipt(VALID_FACTORS, 1_000_000, 1_000_000);
    expect(receipt.reason).toBe('POI_REJECT_SUPPLY_CAP_EXCEEDED');
  });

  it('accepts valid factors', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.reason).toBe('POI_OK');
    expect(receipt.netSeed).toBeGreaterThan(0);
  });
});

// ═══ buildReceipt: CALCULATIONS ═══

describe('buildReceipt: SEED Calculations', () => {
  it('computes poiScore = 0.5α + 0.3β + 0.2γ', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    const expected = 0.5 * 0.9 + 0.3 * 0.7 + 0.2 * 0.8; // 0.82
    expect(receipt.poiScore).toBeCloseTo(expected, 4);
  });

  it('grossSeed = poiScore × ihsān × 10 (base scaling)', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    const expectedPoI = 0.5 * 0.9 + 0.3 * 0.7 + 0.2 * 0.8;
    const expectedGross = expectedPoI * 0.97 * 10;
    expect(receipt.grossSeed).toBeCloseTo(expectedGross, 4);
  });

  it('zakatSeed is 2.5% of grossSeed', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.zakatSeed).toBeCloseTo(receipt.grossSeed * 0.025, 4);
  });

  it('netSeed = grossSeed - zakatSeed', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.netSeed).toBeCloseTo(receipt.grossSeed - receipt.zakatSeed, 4);
  });

  it('poolShare is 50% of netSeed', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.poolShare).toBeCloseTo(receipt.netSeed * 0.50, 3);
  });

  it('walletCredit = netSeed - poolShare', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.walletCredit).toBeCloseTo(receipt.netSeed - receipt.poolShare, 4);
  });

  it('bloom is 1% of walletCredit', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.bloom).toBeCloseTo(receipt.walletCredit * 0.01, 4);
  });
});

// ═══ buildReceipt: SUPPLY CAP ═══

describe('buildReceipt: Supply Cap', () => {
  it('caps grossSeed to remaining supply', () => {
    const receipt = buildReceipt(VALID_FACTORS, 999_999.99, 1_000_000);
    expect(receipt.capHit).toBe(true);
    expect(receipt.grossSeed).toBeLessThanOrEqual(0.01);
  });

  it('no cap hit when supply is plentiful', () => {
    const receipt = buildReceipt(VALID_FACTORS, 0, 1_000_000);
    expect(receipt.capHit).toBe(false);
  });
});

// ═══ buildReceipt: RECEIPT STRUCTURE ═══

describe('buildReceipt: Receipt Structure', () => {
  it('generates a unique ID', () => {
    const r1 = buildReceipt(VALID_FACTORS);
    const r2 = buildReceipt(VALID_FACTORS);
    expect(r1.id).not.toBe(r2.id);
    expect(r1.id).toMatch(/^rcpt_/);
  });

  it('includes ISO timestamp', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });

  it('rejected receipt has zero monetary fields', () => {
    const receipt = buildReceipt({ ...VALID_FACTORS, ihsan: 0.5 });
    expect(receipt.grossSeed).toBe(0);
    expect(receipt.zakatSeed).toBe(0);
    expect(receipt.netSeed).toBe(0);
    expect(receipt.poolShare).toBe(0);
    expect(receipt.walletCredit).toBe(0);
    expect(receipt.bloom).toBe(0);
    expect(receipt.branch).toBe(0);
  });

  it('evidenceHash and chainHash start empty (backend-computed)', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.evidenceHash).toBe('');
    expect(receipt.chainHash).toBe('');
  });

  it('branch is always 0 (attestation handled separately)', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.branch).toBe(0);
  });
});

// ═══ INVARIANTS ═══

describe('buildReceipt: Economic Invariants', () => {
  it('all monetary fields are non-negative', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.grossSeed).toBeGreaterThanOrEqual(0);
    expect(receipt.zakatSeed).toBeGreaterThanOrEqual(0);
    expect(receipt.netSeed).toBeGreaterThanOrEqual(0);
    expect(receipt.poolShare).toBeGreaterThanOrEqual(0);
    expect(receipt.walletCredit).toBeGreaterThanOrEqual(0);
    expect(receipt.bloom).toBeGreaterThanOrEqual(0);
  });

  it('netSeed ≤ grossSeed', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.netSeed).toBeLessThanOrEqual(receipt.grossSeed);
  });

  it('walletCredit ≤ netSeed', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.walletCredit).toBeLessThanOrEqual(receipt.netSeed);
  });

  it('zakat + net = gross (conservation of SEED)', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.zakatSeed + receipt.netSeed).toBeCloseTo(receipt.grossSeed, 4);
  });

  it('poolShare + walletCredit = netSeed (conservation of net)', () => {
    const receipt = buildReceipt(VALID_FACTORS);
    expect(receipt.poolShare + receipt.walletCredit).toBeCloseTo(receipt.netSeed, 4);
  });
});
