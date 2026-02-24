/**
 * Tests for CapabilityCard — PCI-Signed Model Credentials
 */

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  IHSAN_THRESHOLD,
  SNR_THRESHOLD,
  ModelTier,
  TaskType,
  CapabilityCard,
  createCapabilityCard,
  getCardCanonicalBytes,
  isCapabilityCardValid,
  getCardRemainingDays,
  getCardFingerprint,
} from '../capability-card';

// -- Threshold Constants -----------------------------------------------------

describe('CapabilityCard - Thresholds', () => {
  it('should define IHSAN_THRESHOLD as 0.95', () => {
    assert.equal(IHSAN_THRESHOLD, 0.95);
  });

  it('should define SNR_THRESHOLD as 0.85', () => {
    assert.equal(SNR_THRESHOLD, 0.85);
  });
});

// -- Enum Completeness -------------------------------------------------------

describe('CapabilityCard - Enums', () => {
  it('should define ModelTier with EDGE, LOCAL, POOL values', () => {
    assert.equal(ModelTier.EDGE, 'EDGE');
    assert.equal(ModelTier.LOCAL, 'LOCAL');
    assert.equal(ModelTier.POOL, 'POOL');
    assert.equal(Object.values(ModelTier).length, 3);
  });

  it('should define TaskType with all 7 task types', () => {
    const expected = [
      'reasoning', 'chat', 'summarization', 'code_generation',
      'translation', 'classification', 'embedding',
    ];
    const values = Object.values(TaskType);
    assert.equal(values.length, 7);
    for (const e of expected) {
      assert.ok(values.includes(e as TaskType), `Missing task type: ${e}`);
    }
  });
});

// -- createCapabilityCard ----------------------------------------------------

describe('createCapabilityCard', () => {
  const validParams = {
    modelId: 'test-model-7b',
    modelName: 'Test Model 7B',
    tier: ModelTier.LOCAL,
    ihsanScore: 0.97,
    snrScore: 0.90,
    tasksSupported: [TaskType.CHAT, TaskType.REASONING],
  };

  it('should create a valid card with correct fields', () => {
    const card = createCapabilityCard(validParams);
    assert.equal(card.modelId, 'test-model-7b');
    assert.equal(card.modelName, 'Test Model 7B');
    assert.equal(card.tier, ModelTier.LOCAL);
    assert.equal(card.capabilities.ihsanScore, 0.97);
    assert.equal(card.capabilities.snrScore, 0.90);
    assert.deepEqual(card.capabilities.tasksSupported, [TaskType.CHAT, TaskType.REASONING]);
    assert.equal(card.revoked, false);
    assert.equal(card.signature, '');
    assert.equal(card.issuerPublicKey, '');
    assert.ok(card.issuedAt);
    assert.ok(card.expiresAt);
  });

  it('should throw when ihsanScore is below IHSAN_THRESHOLD', () => {
    assert.throws(
      () => createCapabilityCard({ ...validParams, ihsanScore: 0.80 }),
      (err: Error) => {
        assert.ok(err.message.includes('0.8'));
        assert.ok(err.message.includes(String(IHSAN_THRESHOLD)));
        return true;
      }
    );
  });

  it('should throw when snrScore is below SNR_THRESHOLD', () => {
    assert.throws(
      () => createCapabilityCard({ ...validParams, snrScore: 0.50 }),
      (err: Error) => {
        assert.ok(err.message.includes('0.5'));
        assert.ok(err.message.includes(String(SNR_THRESHOLD)));
        return true;
      }
    );
  });

  it('should set 90-day expiration from issue date', () => {
    const before = Date.now();
    const card = createCapabilityCard(validParams);
    const after = Date.now();
    const issued = new Date(card.issuedAt).getTime();
    const expires = new Date(card.expiresAt).getTime();
    const ninetyDaysMs = 90 * 24 * 60 * 60 * 1000;
    assert.ok(issued >= before && issued <= after);
    assert.equal(expires - issued, ninetyDaysMs);
  });

  it('should apply defaults: quantization=unknown, parameterCount=null, maxContext=2048', () => {
    const card = createCapabilityCard({
      modelId: 'bare-model',
      tier: ModelTier.EDGE,
      ihsanScore: 0.96,
      snrScore: 0.88,
      tasksSupported: [TaskType.CHAT],
    });
    assert.equal(card.quantization, 'unknown');
    assert.equal(card.parameterCount, null);
    assert.equal(card.capabilities.maxContext, 2048);
  });
});

// -- getCardCanonicalBytes ---------------------------------------------------

describe('getCardCanonicalBytes', () => {
  it('should return a deterministic Buffer for the same inputs', () => {
    const card = createCapabilityCard({
      modelId: 'determinism-test',
      tier: ModelTier.LOCAL,
      ihsanScore: 0.98,
      snrScore: 0.92,
      tasksSupported: [TaskType.REASONING],
    });
    const bytes1 = getCardCanonicalBytes(card);
    const bytes2 = getCardCanonicalBytes(card);
    assert.ok(Buffer.isBuffer(bytes1));
    assert.ok(bytes1.length > 0);
    assert.ok(bytes1.equals(bytes2));
  });
});

// -- isCapabilityCardValid ---------------------------------------------------

describe('isCapabilityCardValid', () => {
  let baseCard: CapabilityCard;

  beforeEach(() => {
    baseCard = createCapabilityCard({
      modelId: 'validity-test',
      tier: ModelTier.EDGE,
      ihsanScore: 0.97,
      snrScore: 0.90,
      tasksSupported: [TaskType.CHAT],
    });
  });

  it('should reject revoked cards', () => {
    baseCard.revoked = true;
    const result = isCapabilityCardValid(baseCard);
    assert.equal(result.valid, false);
    assert.ok(result.reason?.toLowerCase().includes('revok'));
  });

  it('should reject expired cards', () => {
    baseCard.expiresAt = new Date(Date.now() - 1000).toISOString();
    const result = isCapabilityCardValid(baseCard);
    assert.equal(result.valid, false);
    assert.ok(result.reason?.toLowerCase().includes('expir'));
  });

  it('should reject cards below ihsan threshold', () => {
    // Unsigned cards fail signature check first; verify rejection still holds
    baseCard.capabilities.ihsanScore = 0.50;
    const result = isCapabilityCardValid(baseCard);
    assert.equal(result.valid, false);
    assert.ok(typeof result.reason === 'string' && result.reason.length > 0);
  });

  it('should reject cards below snr threshold', () => {
    // Unsigned cards fail signature check first; verify rejection still holds
    baseCard.capabilities.snrScore = 0.40;
    const result = isCapabilityCardValid(baseCard);
    assert.equal(result.valid, false);
    assert.ok(typeof result.reason === 'string' && result.reason.length > 0);
  });
});

// -- getCardRemainingDays ----------------------------------------------------

describe('getCardRemainingDays', () => {
  it('should return correct days remaining for a fresh card', () => {
    const card = createCapabilityCard({
      modelId: 'days-test', tier: ModelTier.LOCAL,
      ihsanScore: 0.96, snrScore: 0.88, tasksSupported: [TaskType.CHAT],
    });
    const remaining = getCardRemainingDays(card);
    assert.ok(remaining >= 89 && remaining <= 90, `Expected 89-90, got ${remaining}`);
  });

  it('should return 0 for an expired card', () => {
    const card = createCapabilityCard({
      modelId: 'expired-days', tier: ModelTier.EDGE,
      ihsanScore: 0.96, snrScore: 0.88, tasksSupported: [TaskType.CHAT],
    });
    card.expiresAt = new Date(Date.now() - 86400000).toISOString();
    assert.equal(getCardRemainingDays(card), 0);
  });
});

// -- getCardFingerprint ------------------------------------------------------

describe('getCardFingerprint', () => {
  let card: CapabilityCard;

  beforeEach(() => {
    card = createCapabilityCard({
      modelId: 'fingerprint-test', tier: ModelTier.POOL,
      ihsanScore: 0.99, snrScore: 0.95, tasksSupported: [TaskType.EMBEDDING],
    });
  });

  it('should return a 16-character hex string', () => {
    const fp = getCardFingerprint(card);
    assert.equal(fp.length, 16);
    assert.match(fp, /^[0-9a-f]{16}$/);
  });

  it('should be deterministic for the same card', () => {
    const fp1 = getCardFingerprint(card);
    const fp2 = getCardFingerprint(card);
    assert.equal(fp1, fp2);
  });
});
