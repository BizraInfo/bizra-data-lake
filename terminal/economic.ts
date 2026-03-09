/**
 * Unified Economic Contract — Single Source of Truth
 * ===================================================
 * Drop into: frontend/src/types/economic.ts
 *
 * This file replaces the split between:
 * - MissionReceipt (deprecated in types.ts)
 * - RewardReceipt (from reward-engine.ts)
 * - WalletState (from useWallet.ts)
 * - Display transforms scattered across Dashboard.tsx
 *
 * One type system for all economic surfaces.
 *
 * Standing on Giants:
 *   Al-Ghazali (1111) — Ihsān as constitutional quality gate
 *   Nakamoto (2008) — verifiable supply, transparent ledger
 *   Ostrom (1990) — commons governance (BLOOM as governance token)
 */

// ═══════════════════════════════════════════════════════════════════
// THRESHOLDS (mirrored from tokens.ts, verified against constants.py)
// ═══════════════════════════════════════════════════════════════════

export const ECONOMIC_THRESHOLDS = {
  /** Mission floor — minimum Ihsān for action execution */
  MISSION_FLOOR: 0.85,
  /** BLOOM eligibility — minimum Ihsān for governance rights */
  BLOOM_ELIGIBILITY: 0.90,
  /** Minting floor — minimum Ihsān for SEED reward */
  MINTING_FLOOR: 0.95,
  /** Excellence tier — bonus multiplier threshold */
  EXCELLENCE: 0.97,
  /** Zakat rate — constitutional redistribution */
  ZAKAT_RATE: 0.025,
  /** Community pool split — البذرة page 19, HARDCODED */
  COMMUNITY_POOL_SPLIT: 0.50,
  /** SNR minimum — signal quality floor */
  SNR_MINIMUM: 0.85,
} as const;

// ═══════════════════════════════════════════════════════════════════
// TOKEN TYPES
// ═══════════════════════════════════════════════════════════════════

export type TokenType = 'SEED' | 'BLOOM' | 'BRANCH';

export interface TokenBalance {
  /** SEED — liquid utility token, transferable */
  seed: number;
  /** BLOOM — soulbound governance token, non-transferable, decays */
  bloom: number;
  /** BRANCH — reputation token, earned through attestation */
  branch: number;
  /** SEED locked in staking */
  lockedSeed: number;
}

// ═══════════════════════════════════════════════════════════════════
// PROOF OF IMPACT (PoI)
// ═══════════════════════════════════════════════════════════════════

/** PoI rejection reason codes — aligned between frontend and backend */
export type PoIReasonCode =
  | 'POI_OK'
  | 'POI_REJECT_IHSAN_BELOW_THRESHOLD'
  | 'POI_REJECT_SNR_BELOW_THRESHOLD'
  | 'POI_REJECT_SUPPLY_CAP_EXCEEDED'
  | 'POI_REJECT_DUPLICATE'
  | 'POI_REJECT_EXPIRED';

/** Input factors for PoI scoring */
export interface PoIFactors {
  /** Direct work contribution (0-1) */
  contribution: number;
  /** Network reach / influence (0-1) */
  reach: number;
  /** Historical consistency (0-1) */
  longevity: number;
  /** Constitutional quality score (0-1) */
  ihsan: number;
  /** Signal-to-noise ratio (0-1) */
  snr: number;
}

/** PoI weights — must match backend poi_engine.py */
export const POI_WEIGHTS = {
  contribution: 0.5,
  reach: 0.3,
  longevity: 0.2,
} as const;

// ═══════════════════════════════════════════════════════════════════
// UNIFIED ECONOMIC RECEIPT
// ═══════════════════════════════════════════════════════════════════

/**
 * EconomicReceipt — THE canonical receipt type.
 *
 * Replaces MissionReceipt, RewardReceipt, and ad-hoc display transforms.
 * Every economic event in BIZRA produces one of these.
 */
export interface EconomicReceipt {
  /** Unique receipt ID */
  id: string;
  /** ISO timestamp */
  timestamp: string;

  /** PoI composite score (0-1) */
  poiScore: number;
  /** PoI input factors */
  factors: PoIFactors;

  /** Gross SEED before zakat */
  grossSeed: number;
  /** Zakat deduction (2.5% of gross) */
  zakatSeed: number;
  /** Net SEED to node (gross - zakat) */
  netSeed: number;
  /** Community pool share (50% of net, per البذرة) */
  poolShare: number;
  /** Final SEED credited to wallet (net - poolShare) */
  walletCredit: number;

  /** BLOOM accrued (governance weight) */
  bloom: number;
  /** BRANCH earned (reputation) */
  branch: number;

  /** Whether supply cap was hit */
  capHit: boolean;
  /** Reason code */
  reason: PoIReasonCode;

  /** Evidence hash (BLAKE2b of receipt content) */
  evidenceHash: string;
  /** Chain hash (links to previous receipt) */
  chainHash: string;
}

// ═══════════════════════════════════════════════════════════════════
// RECEIPT BUILDER (used by frontend simulation and backend minter)
// ═══════════════════════════════════════════════════════════════════

/**
 * Build an EconomicReceipt from PoI factors.
 *
 * NOTE: Frontend uses this for SIMULATION ONLY.
 * Authoritative minting happens in the backend (poi_engine.py + mint.py).
 * The frontend receipt is a preview — not a ledger entry.
 */
export function buildReceipt(
  factors: PoIFactors,
  supplyUsed: number = 0,
  supplyCap: number = 1_000_000,
): EconomicReceipt {
  const T = ECONOMIC_THRESHOLDS;

  // Gate checks
  if (factors.ihsan < T.MINTING_FLOOR) {
    return rejectReceipt('POI_REJECT_IHSAN_BELOW_THRESHOLD');
  }
  if (factors.snr < T.SNR_MINIMUM) {
    return rejectReceipt('POI_REJECT_SNR_BELOW_THRESHOLD');
  }
  if (supplyUsed >= supplyCap) {
    return rejectReceipt('POI_REJECT_SUPPLY_CAP_EXCEEDED');
  }

  // PoI composite
  const poiScore =
    POI_WEIGHTS.contribution * factors.contribution +
    POI_WEIGHTS.reach * factors.reach +
    POI_WEIGHTS.longevity * factors.longevity;

  // SEED computation
  let grossSeed = poiScore * factors.ihsan * 10; // Base scaling
  const remaining = supplyCap - supplyUsed;
  const capHit = grossSeed > remaining;
  if (capHit) grossSeed = remaining;

  // Zakat (2.5%)
  const zakatSeed = round4(grossSeed * T.ZAKAT_RATE);
  const netSeed = round4(grossSeed - zakatSeed);

  // Community pool (50% of net — البذرة page 19)
  const poolShare = round4(netSeed * T.COMMUNITY_POOL_SPLIT);
  const walletCredit = round4(netSeed - poolShare);

  // BLOOM (1% of wallet credit)
  const bloom = round4(walletCredit * 0.01);

  // BRANCH (0 unless attestation work — handled separately)
  const branch = 0;

  return {
    id: `rcpt_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    timestamp: new Date().toISOString(),
    poiScore: round4(poiScore),
    factors,
    grossSeed: round4(grossSeed),
    zakatSeed,
    netSeed,
    poolShare,
    walletCredit,
    bloom,
    branch,
    capHit,
    reason: 'POI_OK',
    evidenceHash: '',  // Computed by backend
    chainHash: '',     // Computed by backend
  };
}

function rejectReceipt(reason: PoIReasonCode): EconomicReceipt {
  return {
    id: `rcpt_${Date.now()}_rejected`,
    timestamp: new Date().toISOString(),
    poiScore: 0,
    factors: { contribution: 0, reach: 0, longevity: 0, ihsan: 0, snr: 0 },
    grossSeed: 0,
    zakatSeed: 0,
    netSeed: 0,
    poolShare: 0,
    walletCredit: 0,
    bloom: 0,
    branch: 0,
    capHit: false,
    reason,
    evidenceHash: '',
    chainHash: '',
  };
}

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}
