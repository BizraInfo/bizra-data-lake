/**
 * Mission Contract - deterministic pre-execution authority and consent gate.
 *
 * This is the minimum executable slice of bizra.dema.mission_contract.v0.1.
 * It binds a mission to exact human consent, an explicit authority ceiling,
 * and a deterministic SHA-256 content digest before FATE permits execution.
 */

import { createHash } from 'crypto';

export const MISSION_CONTRACT_SCHEMA = 'bizra.dema.mission_contract.v0.1' as const;

export interface MissionBoundary {
  executionAllowed: boolean;
  fileMutationAllowed: boolean;
  networkAllowed: boolean;
  modelInvocationAllowed: boolean;
  maxAuthorityDelta: number;
}

export interface MissionContract {
  schema: typeof MISSION_CONTRACT_SCHEMA;
  missionId: string;
  requiredConsentPhraseHash: string;
  authorityDeltaCeiling: number;
  boundaryDeclared: MissionBoundary;
  contentHash: string;
}

export interface MissionContractValidation {
  passed: boolean;
  gateName: 'MISSION_CONTRACT';
  score: number;
  reason?: string;
}

function canonicalize(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) {
      throw new Error('Mission contract contains a non-canonical value');
    }
    return encoded;
  }

  if (Array.isArray(value)) {
    return `[${value.map(canonicalize).join(',')}]`;
  }

  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalize(record[key])}`).join(',')}}`;
}

export function sha256Hex(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

export function hashConsentPhrase(consentPhrase: string): string {
  return sha256Hex(consentPhrase);
}

export function getMissionContractCanonicalBytes(
  contract: Omit<MissionContract, 'contentHash'> | MissionContract
): Buffer {
  const body: Record<string, unknown> = {
    ...(contract as unknown as Record<string, unknown>),
  };
  delete body.contentHash;
  return Buffer.from(canonicalize(body), 'utf8');
}

export function computeMissionContractHash(
  contract: Omit<MissionContract, 'contentHash'> | MissionContract
): string {
  return createHash('sha256').update(getMissionContractCanonicalBytes(contract)).digest('hex');
}

export function createMissionContract(params: {
  missionId: string;
  consentPhrase: string;
  boundaryDeclared: MissionBoundary;
  authorityDeltaCeiling?: number;
}): MissionContract {
  const authorityDeltaCeiling = params.authorityDeltaCeiling ?? 0;
  const withoutHash: Omit<MissionContract, 'contentHash'> = {
    schema: MISSION_CONTRACT_SCHEMA,
    missionId: params.missionId,
    requiredConsentPhraseHash: hashConsentPhrase(params.consentPhrase),
    authorityDeltaCeiling,
    boundaryDeclared: {
      ...params.boundaryDeclared,
      maxAuthorityDelta: Math.min(
        params.boundaryDeclared.maxAuthorityDelta,
        authorityDeltaCeiling,
      ),
    },
  };

  return {
    ...withoutHash,
    contentHash: computeMissionContractHash(withoutHash),
  };
}

/**
 * Fail-closed pre-execution validation.
 *
 * This gate proves only four things:
 * 1) schema identity,
 * 2) deterministic content integrity,
 * 3) exact consent binding,
 * 4) non-escalating authority.
 *
 * It does not claim that downstream execution, empirical proof, or receipt
 * sealing has occurred.
 */
export function validateMissionContract(
  contract: MissionContract,
  providedConsentPhrase: string,
): MissionContractValidation {
  if (contract.schema !== MISSION_CONTRACT_SCHEMA) {
    return {
      passed: false,
      gateName: 'MISSION_CONTRACT',
      score: 0,
      reason: `Unsupported mission contract schema: ${contract.schema}`,
    };
  }

  let expectedHash: string;
  try {
    expectedHash = computeMissionContractHash(contract);
  } catch (error) {
    return {
      passed: false,
      gateName: 'MISSION_CONTRACT',
      score: 0,
      reason: `Mission contract canonicalization failed: ${String(error)}`,
    };
  }

  if (contract.contentHash !== expectedHash) {
    return {
      passed: false,
      gateName: 'MISSION_CONTRACT',
      score: 0,
      reason: 'Mission contract content hash mismatch',
    };
  }

  if (hashConsentPhrase(providedConsentPhrase) !== contract.requiredConsentPhraseHash) {
    return {
      passed: false,
      gateName: 'MISSION_CONTRACT',
      score: 0,
      reason: 'Exact consent phrase mismatch',
    };
  }

  if (
    contract.authorityDeltaCeiling > 0 ||
    contract.boundaryDeclared.maxAuthorityDelta > contract.authorityDeltaCeiling
  ) {
    return {
      passed: false,
      gateName: 'MISSION_CONTRACT',
      score: 0,
      reason: 'Mission contract violates monotonic authority ceiling',
    };
  }

  return {
    passed: true,
    gateName: 'MISSION_CONTRACT',
    score: 1,
  };
}
