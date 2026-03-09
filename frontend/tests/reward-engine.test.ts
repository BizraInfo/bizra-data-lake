import { describe, it, expect } from 'vitest';
import { calculateReward, simulateReward } from '../src/lib/reward-engine';
import type { PoIInput, SupplyContext } from '../src/lib/reward-engine';
import { THRESHOLDS } from '../src/tokens';

// Canonical inputs that pass all gates
const VALID_INPUT: PoIInput = {
  contribution: 0.8,
  reach: 0.6,
  longevity: 0.7,
  ihsan: 0.96,
  snr: 0.90,
};

describe('calculateReward', () => {
  // ── Gate Tests ──

  it('rejects when ihsan is below production threshold', () => {
    const input = { ...VALID_INPUT, ihsan: THRESHOLDS.IHSAN_PRODUCTION - 0.01 };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_REJECT_IHSAN_BELOW_THRESHOLD');
    expect(receipt.netSeed).toBe(0);
    expect(receipt.bloom).toBe(0);
  });

  it('rejects when SNR is below minimum threshold', () => {
    const input = { ...VALID_INPUT, snr: THRESHOLDS.SNR_MINIMUM - 0.01 };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_REJECT_SNR_BELOW_THRESHOLD');
    expect(receipt.netSeed).toBe(0);
  });

  it('passes when ihsan is exactly at threshold', () => {
    const input = { ...VALID_INPUT, ihsan: THRESHOLDS.IHSAN_PRODUCTION };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_OK');
  });

  it('passes when snr is exactly at threshold', () => {
    const input = { ...VALID_INPUT, snr: THRESHOLDS.SNR_MINIMUM };
    const receipt = calculateReward(input);
    expect(receipt.reason).toBe('POI_OK');
  });

  // ── PoI Score Tests ──

  it('computes composite PoI as 0.5*contribution + 0.3*reach + 0.2*longevity', () => {
    const receipt = calculateReward(VALID_INPUT);
    const expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.7; // 0.72
    expect(receipt.poiScore).toBeCloseTo(expected, 4);
  });

  it('returns poiScore of 1.0 when all inputs are 1.0', () => {
    const input: PoIInput = {
      contribution: 1.0,
      reach: 1.0,
      longevity: 1.0,
      ihsan: 1.0,
      snr: 1.0,
    };
    const receipt = calculateReward(input);
    expect(receipt.poiScore).toBe(1.0);
  });

  // ── Zakat Tests ──

  it('deducts exactly 2.5% zakat from gross SEED', () => {
    const receipt = calculateReward(VALID_INPUT);
    expect(receipt.grossSeed).toBeGreaterThan(0);
    const expectedZakat = Math.round(receipt.grossSeed * THRESHOLDS.ZAKAT_RATE * 10000) / 10000;
    expect(receipt.zakatSeed).toBeCloseTo(expectedZakat, 4);
  });

  it('netSeed equals grossSeed minus zakatSeed', () => {
    const receipt = calculateReward(VALID_INPUT);
    const expectedNet = Math.round((receipt.grossSeed - receipt.zakatSeed) * 10000) / 10000;
    expect(receipt.netSeed).toBeCloseTo(expectedNet, 4);
  });

  // ── BLOOM Accrual Tests ──

  it('accrues BLOOM at 1% of net SEED', () => {
    const receipt = calculateReward(VALID_INPUT);
    const expectedBloom = Math.round(receipt.netSeed * 0.01 * 10000) / 10000;
    expect(receipt.bloom).toBeCloseTo(expectedBloom, 4);
  });

  // ── Supply Cap Tests ──

  it('rejects when yearly supply cap is fully exhausted', () => {
    const supply: SupplyContext = { yearlyMintedSeed: THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR };
    const receipt = calculateReward(VALID_INPUT, supply);
    expect(receipt.reason).toBe('POI_REJECT_SUPPLY_CAP_EXCEEDED');
    expect(receipt.netSeed).toBe(0);
  });

  it('clamps gross SEED to remaining cap when partially exhausted', () => {
    const remaining = 0.001;
    const supply: SupplyContext = {
      yearlyMintedSeed: THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR - remaining,
    };
    const receipt = calculateReward(VALID_INPUT, supply);
    expect(receipt.reason).toBe('POI_OK');
    expect(receipt.capHit).toBe(true);
    expect(receipt.grossSeed).toBeCloseTo(remaining, 4);
  });

  it('does not set capHit when reward fits within remaining supply', () => {
    const receipt = calculateReward(VALID_INPUT, { yearlyMintedSeed: 0 });
    expect(receipt.capHit).toBe(false);
  });

  // ── Default Supply Context ──

  it('defaults to zero yearlyMintedSeed when supply not provided', () => {
    const receipt = calculateReward(VALID_INPUT);
    expect(receipt.reason).toBe('POI_OK');
    expect(receipt.capHit).toBe(false);
  });

  // ── Rejected Receipt Shape ──

  it('returns all-zero receipt on rejection', () => {
    const receipt = calculateReward({ ...VALID_INPUT, ihsan: 0.5 });
    expect(receipt).toEqual({
      poiScore: 0,
      grossSeed: 0,
      zakatSeed: 0,
      netSeed: 0,
      bloom: 0,
      capHit: false,
      reason: 'POI_REJECT_IHSAN_BELOW_THRESHOLD',
    });
  });
});

describe('simulateReward', () => {
  it('produces POI_OK for high ihsan', () => {
    const receipt = simulateReward(0.97);
    expect(receipt.reason).toBe('POI_OK');
    expect(receipt.netSeed).toBeGreaterThan(0);
  });

  it('rejects for ihsan below constitutional threshold', () => {
    const receipt = simulateReward(0.80);
    expect(receipt.reason).not.toBe('POI_OK');
  });

  it('clamps synthetic contribution to 1.0 max', () => {
    // ihsan 1.0 → contribution = min(1, 1.02) = 1.0
    const receipt = simulateReward(1.0);
    expect(receipt.reason).toBe('POI_OK');
    expect(receipt.poiScore).toBeLessThanOrEqual(1.0);
  });

  it('ensures synthetic SNR meets minimum threshold', () => {
    // ihsan 0.95 → snr = max(0.85, 0.90) = 0.90
    const receipt = simulateReward(0.95);
    expect(receipt.reason).toBe('POI_OK');
  });
});
