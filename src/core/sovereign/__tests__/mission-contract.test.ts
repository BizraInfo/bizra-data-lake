/**
 * Tests for the deterministic Mission Contract pre-execution gate.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
  MISSION_CONTRACT_SCHEMA,
  computeMissionContractHash,
  createMissionContract,
  validateMissionContract,
  type MissionContract,
} from '../mission-contract';

const boundary = {
  executionAllowed: true,
  fileMutationAllowed: false,
  networkAllowed: false,
  modelInvocationAllowed: true,
  maxAuthorityDelta: 0,
};

describe('Mission Contract pre-execution gate', () => {
  it('accepts an intact contract with exact consent and zero authority delta', () => {
    const contract = createMissionContract({
      missionId: 'mission-001',
      consentPhrase: 'execute mission-001',
      boundaryDeclared: boundary,
    });

    assert.equal(contract.schema, MISSION_CONTRACT_SCHEMA);
    assert.equal(contract.authorityDeltaCeiling, 0);
    assert.equal(contract.contentHash, computeMissionContractHash(contract));

    const result = validateMissionContract(contract, 'execute mission-001');
    assert.equal(result.passed, true);
    assert.equal(result.gateName, 'MISSION_CONTRACT');
    assert.equal(result.score, 1);
  });

  it('fails closed when the consent phrase is not exact', () => {
    const contract = createMissionContract({
      missionId: 'mission-002',
      consentPhrase: 'approve mission-002 exactly',
      boundaryDeclared: boundary,
    });

    const result = validateMissionContract(contract, 'approve mission-002');
    assert.equal(result.passed, false);
    assert.match(result.reason ?? '', /consent/i);
  });

  it('fails closed when contract content is changed after hashing', () => {
    const contract = createMissionContract({
      missionId: 'mission-003',
      consentPhrase: 'execute mission-003',
      boundaryDeclared: boundary,
    });

    const tampered: MissionContract = {
      ...contract,
      missionId: 'mission-003-tampered',
    };

    const result = validateMissionContract(tampered, 'execute mission-003');
    assert.equal(result.passed, false);
    assert.match(result.reason ?? '', /hash/i);
  });

  it('rejects authority escalation even when the escalated contract is re-hashed', () => {
    const contract = createMissionContract({
      missionId: 'mission-004',
      consentPhrase: 'execute mission-004',
      boundaryDeclared: boundary,
    });

    const escalatedBody = {
      ...contract,
      boundaryDeclared: {
        ...contract.boundaryDeclared,
        maxAuthorityDelta: 1,
      },
    };

    const escalated: MissionContract = {
      ...escalatedBody,
      contentHash: computeMissionContractHash(escalatedBody),
    };

    const result = validateMissionContract(escalated, 'execute mission-004');
    assert.equal(result.passed, false);
    assert.match(result.reason ?? '', /authority/i);
  });
});
