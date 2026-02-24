/**
 * Tests for SovereignRuntime — Unified entry point for the Sovereign LLM ecosystem.
 *
 * Validates constructor defaults, start/stop lifecycle, status shape,
 * registry access, inference guards, and banner output.
 *
 * Note: start() tests use NetworkMode.OFFLINE to avoid federation init,
 * and the runtime gracefully handles sandbox/fate-binding unavailability.
 */

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';

import {
  SovereignRuntime,
  DEFAULT_RUNTIME_CONFIG,
  printBanner,
} from '../runtime';
import { ModelRegistry } from '../model-registry';
import { NetworkMode } from '../network-mode';

// -- DEFAULT_RUNTIME_CONFIG --------------------------------------------------

describe('DEFAULT_RUNTIME_CONFIG', () => {
  it('should set networkMode to HYBRID', () => {
    assert.equal(DEFAULT_RUNTIME_CONFIG.networkMode, NetworkMode.HYBRID);
  });

  it('should set poolQuorum to 0.67', () => {
    assert.equal(DEFAULT_RUNTIME_CONFIG.poolQuorum, 0.67);
  });

  it('should enable pool by default', () => {
    assert.equal(DEFAULT_RUNTIME_CONFIG.enablePool, true);
  });

  it('should set discoveryTimeoutMs to 5000', () => {
    assert.equal(DEFAULT_RUNTIME_CONFIG.discoveryTimeoutMs, 5000);
  });

  it('should set modelStorePath to ~/.bizra/models', () => {
    assert.equal(DEFAULT_RUNTIME_CONFIG.modelStorePath, '~/.bizra/models');
  });
});

// -- Constructor -------------------------------------------------------------

describe('SovereignRuntime - Constructor', () => {
  it('should create runtime with default config', () => {
    const runtime = new SovereignRuntime();
    const status = runtime.getStatus();
    assert.equal(status.networkMode, NetworkMode.HYBRID);
  });

  it('should merge partial config with defaults', () => {
    const runtime = new SovereignRuntime({
      networkMode: NetworkMode.OFFLINE,
      poolQuorum: 0.51,
    });
    const status = runtime.getStatus();
    assert.equal(status.networkMode, NetworkMode.OFFLINE);
    assert.equal(status.thresholds.ihsan, 0.95);
  });
});

// -- start / stop lifecycle --------------------------------------------------

describe('SovereignRuntime - Lifecycle', () => {
  let runtime: SovereignRuntime;

  beforeEach(() => {
    // OFFLINE mode avoids federation init; sandbox failure is handled gracefully
    runtime = new SovereignRuntime({ networkMode: NetworkMode.OFFLINE });
  });

  afterEach(async () => {
    try {
      await runtime.stop();
    } catch {
      /* already stopped */
    }
  });

  it('start() should set started to true', async () => {
    // Suppress sandbox/fate console warnings during test
    const origWarn = console.warn;
    const origLog = console.log;
    console.warn = () => {};
    console.log = () => {};
    try {
      await runtime.start();
      assert.equal(runtime.getStatus().started, true);
    } finally {
      console.warn = origWarn;
      console.log = origLog;
    }
  });

  it('start() should be idempotent', async () => {
    const origWarn = console.warn;
    const origLog = console.log;
    console.warn = () => {};
    console.log = () => {};
    try {
      await runtime.start();
      await runtime.start();
      assert.equal(runtime.getStatus().started, true);
    } finally {
      console.warn = origWarn;
      console.log = origLog;
    }
  });

  it('stop() should set started to false', async () => {
    const origWarn = console.warn;
    const origLog = console.log;
    console.warn = () => {};
    console.log = () => {};
    try {
      await runtime.start();
      await runtime.stop();
      assert.equal(runtime.getStatus().started, false);
    } finally {
      console.warn = origWarn;
      console.log = origLog;
    }
  });

  it('stop() should be idempotent', async () => {
    const origWarn = console.warn;
    const origLog = console.log;
    console.warn = () => {};
    console.log = () => {};
    try {
      await runtime.start();
      await runtime.stop();
      await runtime.stop();
      assert.equal(runtime.getStatus().started, false);
    } finally {
      console.warn = origWarn;
      console.log = origLog;
    }
  });
});

// -- getStatus ---------------------------------------------------------------

describe('SovereignRuntime - getStatus', () => {
  it('should return correct RuntimeStatus shape', () => {
    const runtime = new SovereignRuntime();
    const status = runtime.getStatus();
    assert.equal(typeof status.started, 'boolean');
    assert.equal(typeof status.networkMode, 'string');
    assert.equal(typeof status.effectiveMode, 'string');
    assert.equal(typeof status.registeredModels, 'number');
    assert.equal(typeof status.validModels, 'number');
    assert.equal(typeof status.poolAvailable, 'boolean');
    assert.equal(typeof status.sandboxHealthy, 'boolean');
    assert.ok(status.thresholds);
    assert.equal(typeof status.thresholds.ihsan, 'number');
    assert.equal(typeof status.thresholds.snr, 'number');
  });

  it('should report started=false before start() is called', () => {
    const runtime = new SovereignRuntime();
    assert.equal(runtime.getStatus().started, false);
  });

  it('should report 0 registered models on fresh runtime', () => {
    const runtime = new SovereignRuntime();
    assert.equal(runtime.getStatus().registeredModels, 0);
    assert.equal(runtime.getStatus().validModels, 0);
  });
});

// -- getRegistry -------------------------------------------------------------

describe('SovereignRuntime - getRegistry', () => {
  it('should return a ModelRegistry instance', () => {
    const runtime = new SovereignRuntime();
    const registry = runtime.getRegistry();
    assert.ok(registry instanceof ModelRegistry);
  });
});

// -- setInferenceFunction ----------------------------------------------------

describe('SovereignRuntime - setInferenceFunction', () => {
  it('should store the inference function without throwing', () => {
    const runtime = new SovereignRuntime();
    const fn = async (_modelId: string, _prompt: string) => 'response';
    runtime.setInferenceFunction(fn);
  });
});

// -- infer guard -------------------------------------------------------------

describe('SovereignRuntime - infer', () => {
  it('should throw when runtime is not started', async () => {
    const runtime = new SovereignRuntime();
    await assert.rejects(
      () => runtime.infer({ prompt: 'hello' }),
      (err: Error) => {
        assert.ok(err.message.includes('not started'));
        return true;
      },
    );
  });
});

// -- challengeModel guard ----------------------------------------------------

describe('SovereignRuntime - challengeModel', () => {
  it('should throw when inference function is not set', async () => {
    const runtime = new SovereignRuntime();
    await assert.rejects(
      () => runtime.challengeModel('model-1', '/path/to/model'),
      (err: Error) => {
        assert.ok(err.message.includes('Inference function not set'));
        return true;
      },
    );
  });
});

// -- printBanner -------------------------------------------------------------

describe('printBanner', () => {
  it('should execute without throwing', () => {
    const originalLog = console.log;
    console.log = () => {};
    try {
      assert.doesNotThrow(() => printBanner());
    } finally {
      console.log = originalLog;
    }
  });
});
