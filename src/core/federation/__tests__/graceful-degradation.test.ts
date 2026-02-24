/**
 * Tests for GracefulDegradation — failure recording, mode degradation,
 * recovery, executeWithFallback, and forceMode.
 */

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';

import {
  GracefulDegradation,
  FallbackConfig,
} from '../graceful-degradation';
import { ModelRegistry } from '../../sovereign/model-registry';
import { NetworkMode } from '../../sovereign/network-mode';

// -- Helpers ---------------------------------------------------------------

function createDegradation(
  targetMode: NetworkMode = NetworkMode.FEDERATED,
  config: Partial<FallbackConfig> = {}
): GracefulDegradation {
  const registry = new ModelRegistry();
  return new GracefulDegradation(registry, targetMode, {
    maxConsecutiveFailures: 3,
    recoveryCheckIntervalMs: 600_000, // Very long to prevent auto-triggers
    fallbackTimeoutMs: 100,
    autoRecovery: false, // Disable auto-recovery in tests
    ...config,
  });
}

// -- Constructor -----------------------------------------------------------

describe('GracefulDegradation - Constructor', () => {
  it('should start in non-degraded state', () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    assert.equal(gd.isDegraded(), false);
    assert.equal(gd.getEffectiveMode(), NetworkMode.HYBRID);
  });

  it('state should reflect target mode', () => {
    const gd = createDegradation(NetworkMode.OFFLINE);
    const state = gd.getState();
    assert.equal(state.currentMode, NetworkMode.OFFLINE);
    assert.equal(state.targetMode, NetworkMode.OFFLINE);
    assert.equal(state.isDegraded, false);
  });
});

// -- recordSuccess / recordFailure -----------------------------------------

describe('GracefulDegradation - Success/Failure Recording', () => {
  let gd: GracefulDegradation;

  beforeEach(() => {
    gd = createDegradation(NetworkMode.FEDERATED);
  });

  it('recordSuccess() resets consecutive failures', () => {
    gd.recordFailure('err1');
    gd.recordFailure('err2');
    gd.recordSuccess();
    assert.equal(gd.getState().consecutiveFailures, 0);
  });

  it('recordSuccess() updates lastSuccessAt', () => {
    gd.recordSuccess();
    assert.ok(gd.getState().lastSuccessAt instanceof Date);
  });

  it('recordFailure() increments consecutive failures', () => {
    gd.recordFailure('fail1');
    assert.equal(gd.getState().consecutiveFailures, 1);
    gd.recordFailure('fail2');
    assert.equal(gd.getState().consecutiveFailures, 2);
  });

  it('recordFailure() updates lastFailureAt', () => {
    gd.recordFailure('fail');
    assert.ok(gd.getState().lastFailureAt instanceof Date);
  });
});

// -- Degradation -----------------------------------------------------------

describe('GracefulDegradation - Degradation Trigger', () => {
  it('degrades after maxConsecutiveFailures', () => {
    const gd = createDegradation(NetworkMode.FEDERATED);

    gd.recordFailure('f1');
    gd.recordFailure('f2');
    assert.equal(gd.isDegraded(), false);

    gd.recordFailure('f3'); // Triggers degradation (maxConsecutiveFailures=3)
    assert.equal(gd.isDegraded(), true);
  });

  it('FEDERATED degrades to LOCAL_ONLY', () => {
    const gd = createDegradation(NetworkMode.FEDERATED);
    for (let i = 0; i < 3; i++) gd.recordFailure('fail');
    assert.equal(gd.getEffectiveMode(), NetworkMode.LOCAL_ONLY);
  });

  it('HYBRID degrades to LOCAL_ONLY', () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    for (let i = 0; i < 3; i++) gd.recordFailure('fail');
    assert.equal(gd.getEffectiveMode(), NetworkMode.LOCAL_ONLY);
  });

  it('LOCAL_ONLY degrades to OFFLINE', () => {
    const gd = createDegradation(NetworkMode.LOCAL_ONLY);
    for (let i = 0; i < 3; i++) gd.recordFailure('fail');
    assert.equal(gd.getEffectiveMode(), NetworkMode.OFFLINE);
  });

  it('OFFLINE cannot degrade further', () => {
    const gd = createDegradation(NetworkMode.OFFLINE);
    for (let i = 0; i < 5; i++) gd.recordFailure('fail');
    assert.equal(gd.getEffectiveMode(), NetworkMode.OFFLINE);
    // Still not considered degraded since target == current
    assert.equal(gd.isDegraded(), false);
  });

  it('emits degraded event with from/to/reason', () => {
    const gd = createDegradation(NetworkMode.FEDERATED);
    let emittedData: any = null;
    gd.on('degraded', (data) => { emittedData = data; });

    for (let i = 0; i < 3; i++) gd.recordFailure('network down');

    assert.ok(emittedData);
    assert.equal(emittedData.from, NetworkMode.FEDERATED);
    assert.equal(emittedData.to, NetworkMode.LOCAL_ONLY);
    assert.equal(emittedData.reason, 'network down');
  });
});

// -- Recovery --------------------------------------------------------------

describe('GracefulDegradation - Recovery', () => {
  it('attemptRecovery() returns true when not degraded', async () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    const recovered = await gd.attemptRecovery();
    assert.equal(recovered, true);
  });

  it('attemptRecovery() upgrades OFFLINE to LOCAL_ONLY', async () => {
    const gd = createDegradation(NetworkMode.FEDERATED);

    // Degrade to LOCAL_ONLY first
    for (let i = 0; i < 3; i++) gd.recordFailure('fail');
    assert.equal(gd.getEffectiveMode(), NetworkMode.LOCAL_ONLY);

    // Degrade again to OFFLINE
    for (let i = 0; i < 3; i++) gd.recordFailure('fail');
    assert.equal(gd.getEffectiveMode(), NetworkMode.OFFLINE);

    // Recover one step
    const recovered = await gd.attemptRecovery();
    assert.equal(gd.getEffectiveMode(), NetworkMode.LOCAL_ONLY);
    // Not fully recovered yet (target is FEDERATED)
    assert.equal(recovered, false);
    assert.equal(gd.isDegraded(), true);
  });

  it('emits recovered event', async () => {
    const gd = createDegradation(NetworkMode.FEDERATED);

    // Degrade FEDERATED → LOCAL_ONLY
    for (let i = 0; i < 3; i++) gd.recordFailure('fail');
    assert.equal(gd.getEffectiveMode(), NetworkMode.LOCAL_ONLY);

    // Degrade LOCAL_ONLY → OFFLINE
    for (let i = 0; i < 3; i++) gd.recordFailure('fail');
    assert.equal(gd.getEffectiveMode(), NetworkMode.OFFLINE);

    let emittedData: any = null;
    gd.on('recovered', (data) => { emittedData = data; });

    // Recover OFFLINE → LOCAL_ONLY (always succeeds)
    await gd.attemptRecovery();
    assert.ok(emittedData);
    assert.equal(emittedData.from, NetworkMode.OFFLINE);
    assert.equal(emittedData.to, NetworkMode.LOCAL_ONLY);
  });
});

// -- forceMode -------------------------------------------------------------

describe('GracefulDegradation - forceMode', () => {
  it('overrides current mode', () => {
    const gd = createDegradation(NetworkMode.FEDERATED);
    gd.forceMode(NetworkMode.OFFLINE);
    assert.equal(gd.getEffectiveMode(), NetworkMode.OFFLINE);
    assert.equal(gd.isDegraded(), true);
  });

  it('resets consecutive failures', () => {
    const gd = createDegradation(NetworkMode.FEDERATED);
    gd.recordFailure('f1');
    gd.recordFailure('f2');
    gd.forceMode(NetworkMode.FEDERATED);
    assert.equal(gd.getState().consecutiveFailures, 0);
  });

  it('not degraded when forced to target mode', () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    gd.forceMode(NetworkMode.HYBRID);
    assert.equal(gd.isDegraded(), false);
  });

  it('emits mode-forced event', () => {
    const gd = createDegradation(NetworkMode.FEDERATED);
    let emitted = false;
    gd.on('mode-forced', () => { emitted = true; });
    gd.forceMode(NetworkMode.OFFLINE);
    assert.equal(emitted, true);
  });
});

// -- executeWithFallback ---------------------------------------------------

describe('GracefulDegradation - executeWithFallback', () => {
  it('returns primary result on success', async () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    const { result, usedFallback } = await gd.executeWithFallback(
      async () => 'primary',
      async () => 'fallback',
    );
    assert.equal(result, 'primary');
    assert.equal(usedFallback, false);
  });

  it('returns fallback result on primary failure', async () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    const { result, usedFallback } = await gd.executeWithFallback(
      async () => { throw new Error('boom'); },
      async () => 'fallback',
    );
    assert.equal(result, 'fallback');
    assert.equal(usedFallback, true);
  });

  it('records failure when primary fails', async () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    await gd.executeWithFallback(
      async () => { throw new Error('fail'); },
      async () => 'ok',
    );
    assert.equal(gd.getState().consecutiveFailures, 1);
  });

  it('throws when both primary and fallback fail', async () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    await assert.rejects(
      () => gd.executeWithFallback(
        async () => { throw new Error('primary-fail'); },
        async () => { throw new Error('fallback-fail'); },
      ),
      (err: Error) => {
        assert.ok(err.message.includes('Primary and fallback both failed'));
        return true;
      },
    );
  });

  it('times out slow primary and uses fallback', async () => {
    const gd = createDegradation(NetworkMode.HYBRID, { fallbackTimeoutMs: 50 });
    const { result, usedFallback } = await gd.executeWithFallback(
      () => new Promise((resolve) => setTimeout(() => resolve('slow'), 500)),
      async () => 'fast-fallback',
      { timeoutMs: 50 },
    );
    assert.equal(result, 'fast-fallback');
    assert.equal(usedFallback, true);
  });
});

// -- start / stop (timer management) ---------------------------------------

describe('GracefulDegradation - Lifecycle', () => {
  it('start() and stop() do not throw', () => {
    const gd = createDegradation(NetworkMode.HYBRID, { autoRecovery: true, recoveryCheckIntervalMs: 600_000 });
    gd.start();
    gd.stop();
  });

  it('stop() is idempotent', () => {
    const gd = createDegradation(NetworkMode.HYBRID);
    gd.stop();
    gd.stop(); // Should not throw
  });
});
