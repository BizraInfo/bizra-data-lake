/**
 * BIZRA Reward Engine — Spec-Faithful PoI → SEED/BLOOM Calculation
 *
 * Mirrors the backend canonical pipeline:
 *   core/proof_engine/poi_engine.py — 4-stage PoI scoring
 *   core/token/mint.py              — SEED minting with supply cap + zakat
 *   core/token/types.py             — Constants (ZAKAT_RATE, SEED_SUPPLY_CAP_PER_YEAR)
 *   00_CONSTITUTION/DECLARATION.md  — SEED from verified action, BLOOM soulbound
 *
 * This module replaces the "demo math" DropRarity (LEGENDARY/EPIC/RARE) with
 * the spec-based pipeline: PoI inputs → composite score → SEED/BLOOM amounts + audit.
 *
 * IMPORTANT: This is an OFFLINE SIMULATION for UI preview — not an authoritative
 * mint. The backend (core/token/mint.py) uses proportional pool distribution
 * across all contributors per epoch, while this engine uses direct per-contributor
 * scaling. BLOOM accrual here approximates at 1% of net SEED; the backend mints
 * BLOOM from SEED staking (a separate operation). Gini rebalancing (SAT solver)
 * is backend-only. Authoritative token balances always come from the backend ledger.
 *
 * Standing on Giants:
 *   Nakamoto (2008) — Proof-of-Work as verifiable contribution
 *   Page & Brin (1998) — PageRank for citation-based authority
 *   Shannon (1948) — Signal-to-noise as quality metric
 *   Al-Ghazali (1058-1111) — Proportional justice in distribution
 */

import { THRESHOLDS } from '../tokens';

// ═══ Constants (synced with core/token/types.py) ═══

const ZAKAT_RATE = THRESHOLDS.ZAKAT_RATE;
const SEED_SUPPLY_CAP_PER_YEAR = THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR;

/** PoI composite weights — mirrors core/proof_engine/poi_engine.py defaults. */
const POI_WEIGHTS = {
  contribution: 0.5,  // α — signature + SNR quality
  reach: 0.3,         // β — PageRank centrality
  longevity: 0.2,     // γ — temporal relevance with decay
} as const;

/** Base SEED reward per PoI unit (tunable by governance). */
const BASE_SEED_PER_POI = 1.0;

/** BLOOM accumulation rate per SEED minted (constitutional minimum). */
const BLOOM_ACCRUAL_RATE = 0.01;

// ═══ Reason Codes (mirrors core/proof_engine/poi_engine.py PoIReasonCode) ═══

export type PoIReasonCode =
  | 'POI_OK'
  | 'POI_QUARANTINE_MISSING_EVIDENCE'
  | 'POI_REJECT_BAD_SIGNATURE'
  | 'POI_REJECT_DUPLICATE_ARTIFACT'
  | 'POI_REJECT_SNR_BELOW_THRESHOLD'
  | 'POI_REJECT_EPOCH_MISMATCH'
  | 'POI_REJECT_IHSAN_BELOW_THRESHOLD'
  | 'POI_REJECT_SUPPLY_CAP_EXCEEDED'
  | 'POI_PENALTY_RING_DETECTED'
  | 'POI_PENALTY_RECIPROCAL_FARM'
  | 'POI_INTERNAL_INVARIANT_FAIL';

// ═══ Input / Output Types ═══

/** Raw inputs for PoI scoring — matches the 4-stage pipeline. */
export interface PoIInput {
  /** Contribution quality  (0–1). Stage 1: signature validity + SNR score. */
  contribution: number;
  /** Network reach factor  (0–1). Stage 2: PageRank centrality. */
  reach: number;
  /** Temporal longevity    (0–1). Stage 3: decay function applied. */
  longevity: number;
  /** Ihsan quality score   (0–1). Constitutional gate. */
  ihsan: number;
  /** SNR of the artifact   (0–1). Hard gate at THRESHOLDS.SNR_MINIMUM. */
  snr: number;
}

/** Supply context for cap enforcement. */
export interface SupplyContext {
  /** SEED already minted this epoch year. */
  yearlyMintedSeed: number;
}

/** Reward receipt — output of the reward engine. */
export interface RewardReceipt {
  /** Composite PoI score (0–1). */
  poiScore: number;
  /** Gross SEED before zakat. */
  grossSeed: number;
  /** Zakat deducted to community fund. */
  zakatSeed: number;
  /** Net SEED credited to contributor. */
  netSeed: number;
  /** BLOOM accrued (soulbound governance token). */
  bloom: number;
  /** Whether the yearly supply cap was hit. */
  capHit: boolean;
  /** Audit reason code. */
  reason: PoIReasonCode;
}

// ═══ Engine ═══

/**
 * Compute the 4-stage composite PoI score.
 * Same formula as core/proof_engine/poi_engine.py Stage 4:
 *   composite = α * contribution + β * reach + γ * longevity
 */
function computePoIScore(input: PoIInput): number {
  return (
    POI_WEIGHTS.contribution * input.contribution +
    POI_WEIGHTS.reach * input.reach +
    POI_WEIGHTS.longevity * input.longevity
  );
}

/**
 * Calculate a spec-faithful reward receipt from PoI inputs.
 *
 * Gates:
 *   1. Ihsan must meet production threshold (0.95)
 *   2. SNR must meet minimum threshold (0.85)
 *
 * Pipeline:
 *   PoI composite → Base SEED × PoI × Ihsan multiplier
 *   → supply cap check → zakat deduction → net SEED + BLOOM accrual
 */
export function calculateReward(
  input: PoIInput,
  supply: SupplyContext = { yearlyMintedSeed: 0 },
): RewardReceipt {
  // Gate 1: Ihsan floor
  if (input.ihsan < THRESHOLDS.IHSAN_PRODUCTION) {
    return rejected('POI_REJECT_IHSAN_BELOW_THRESHOLD');
  }

  // Gate 2: SNR floor
  if (input.snr < THRESHOLDS.SNR_MINIMUM) {
    return rejected('POI_REJECT_SNR_BELOW_THRESHOLD');
  }

  const poiScore = computePoIScore(input);

  // SEED calculation: base × composite × ihsan quality multiplier
  let grossSeed = BASE_SEED_PER_POI * poiScore * input.ihsan;

  // Supply cap enforcement
  const remaining = SEED_SUPPLY_CAP_PER_YEAR - supply.yearlyMintedSeed;
  if (remaining <= 0) {
    return rejected('POI_REJECT_SUPPLY_CAP_EXCEEDED');
  }
  const capHit = grossSeed > remaining;
  if (capHit) {
    grossSeed = remaining;
  }

  // Zakat: 2.5% to community fund
  const zakatSeed = round(grossSeed * ZAKAT_RATE);
  const netSeed = round(grossSeed - zakatSeed);

  // BLOOM: soulbound governance accrual
  const bloom = round(netSeed * BLOOM_ACCRUAL_RATE);

  return {
    poiScore: round(poiScore),
    grossSeed: round(grossSeed),
    zakatSeed,
    netSeed,
    bloom,
    capHit,
    reason: 'POI_OK',
  };
}

// ═══ Offline Simulation ═══

/**
 * Simulate reward when the backend is unreachable.
 * Produces a plausible receipt using the same pipeline with synthetic inputs.
 * Clearly marked as simulation — not a minting event.
 */
export function simulateReward(ihsan: number): RewardReceipt {
  const input: PoIInput = {
    contribution: Math.min(1, ihsan * 1.02),
    reach: 0.5,         // median network position for solo node
    longevity: 0.8,     // recent contribution, mild decay
    ihsan,
    snr: Math.max(THRESHOLDS.SNR_MINIMUM, ihsan - 0.05),
  };
  return calculateReward(input);
}

// ═══ Helpers ═══

function rejected(reason: PoIReasonCode): RewardReceipt {
  return {
    poiScore: 0,
    grossSeed: 0,
    zakatSeed: 0,
    netSeed: 0,
    bloom: 0,
    capHit: false,
    reason,
  };
}

function round(n: number, decimals = 4): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}
