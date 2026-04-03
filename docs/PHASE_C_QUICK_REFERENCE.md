# Phase C Quick Reference: Self-Healing & Resilience

## Circuit Breaker Events

### Drain Events
```rust
use crate::apex::circuit_breaker::CircuitBreakerManager;

let manager = CircuitBreakerManager::new();
let events = manager.drain_events(); // Retrieve and clear

for event in events {
    match event {
        CircuitBreakerEvent::Tripped { circuit_name, reason, trip_count } => {
            tracing::error!("Circuit {} tripped: {} (count: {})", circuit_name, reason, trip_count);
        }
        CircuitBreakerEvent::RecoveryStarted { circuit_name } => {
            tracing::info!("Circuit {} attempting recovery", circuit_name);
        }
        CircuitBreakerEvent::Recovered { circuit_name, recovery_duration_ms } => {
            tracing::info!("Circuit {} recovered in {}ms", circuit_name, recovery_duration_ms);
        }
        CircuitBreakerEvent::RecoveryFailed { circuit_name, reason } => {
            tracing::warn!("Circuit {} recovery failed: {}", circuit_name, reason);
        }
    }
}
```

---

## FATE Escalation WAL

### Startup Recovery
```rust
use crate::fate::FATECoordinator;

// On system startup
let mut fate = FATECoordinator::new();
let recovered = fate.recover_pending_escalations();

tracing::info!("Recovered {} pending escalations", recovered.len());

for escalation in recovered {
    // Re-persist to Redis if available
    fate.persist_to_synapse(&escalation).await?;
}
```

### WAL Location
- **Path:** `docs/evidence/receipts/fate/wal.jsonl`
- **Format:** JSONL (newline-delimited JSON)
- **Auto-created:** Directory created on first write

---

## Identity Monitor

### Enable Monitoring
```rust
use crate::sovereign_runtime_omega::SovereignKernel;

let mut kernel = SovereignKernel::new();

// Enable with 5-minute check interval
kernel.enable_identity_monitor(300);
```

### Record Baseline
```rust
// After hardware binding, record fingerprints
kernel.record_identity_baseline(
    "cpu_gpu_mobo_fingerprint",
    "ram_storage_mac_fingerprint",
    "os_bios_wsl_fingerprint"
);
```

### Periodic Drift Check
```rust
// In monitoring loop
if let Some(drift) = kernel.check_identity_drift(
    current_tier1,
    current_tier2,
    current_tier3
) {
    match drift.action.as_str() {
        "halt" => {
            // Tier 1: Critical - halt system
            tracing::error!("TIER 1 DRIFT: Hardware identity mismatch");
            std::process::exit(1);
        }
        "warn" => {
            // Tier 2: Warning - require attestation
            tracing::warn!("TIER 2 DRIFT: Hardware component changed");
            // Trigger attestation flow
        }
        "log" => {
            // Tier 3: Info - expected change
            tracing::info!("TIER 3 DRIFT: OS/BIOS updated");
        }
        _ => {}
    }
}
```

### Access Drift Events
```rust
if let Some(monitor) = kernel.identity_monitor() {
    let events = monitor.drift_events();
    for event in events {
        tracing::info!(
            "Drift detected - Tier {}, Action: {}, Expected: {}..., Got: {}...",
            event.tier,
            event.action,
            &event.expected_hash[..16],
            &event.actual_hash[..16]
        );
    }
}
```

---

## Testing Examples

### Test Circuit Breaker Events
```rust
#[test]
fn test_circuit_breaker_event_emission() {
    let manager = CircuitBreakerManager::with_config(CircuitBreakerConfig {
        failure_threshold: 2,
        ..Default::default()
    });

    manager.record_failure("test_agent").unwrap();
    manager.record_failure("test_agent").unwrap();

    let events = manager.drain_events();
    assert_eq!(events.len(), 1);

    match &events[0] {
        CircuitBreakerEvent::Tripped { circuit_name, trip_count, .. } => {
            assert_eq!(circuit_name, "test_agent");
            assert_eq!(*trip_count, 1);
        }
        _ => panic!("Expected Tripped event"),
    }
}
```

### Test FATE WAL Recovery
```rust
#[test]
fn test_wal_persistence() {
    let mut fate = FATECoordinator::new();

    let codes = vec![RejectionCode::SecurityThreat("test".to_string())];
    let escalation = fate.escalate_rejection(&codes, "test", &HashMap::new());

    // Simulate restart
    let mut fate2 = FATECoordinator::new();
    let recovered = fate2.recover_pending_escalations();

    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].id, escalation.id);
}
```

### Test Identity Drift Detection
```rust
#[test]
fn test_identity_drift_tiers() {
    let mut kernel = SovereignKernel::new();
    kernel.enable_identity_monitor(300);
    kernel.record_identity_baseline("t1", "t2", "t3");

    // No drift
    assert!(kernel.check_identity_drift("t1", "t2", "t3").is_none());

    // Tier 3 drift (log)
    let drift = kernel.check_identity_drift("t1", "t2", "t3_new").unwrap();
    assert_eq!(drift.tier, 3);
    assert_eq!(drift.action, "log");

    // Tier 2 drift (warn)
    let drift = kernel.check_identity_drift("t1", "t2_new", "t3").unwrap();
    assert_eq!(drift.tier, 2);
    assert_eq!(drift.action, "warn");

    // Tier 1 drift (halt)
    let drift = kernel.check_identity_drift("t1_new", "t2", "t3").unwrap();
    assert_eq!(drift.tier, 1);
    assert_eq!(drift.action, "halt");
}
```

---

## Troubleshooting

### WAL File Permissions
```bash
# If WAL writes fail, check directory permissions
ls -la docs/evidence/receipts/fate/
chmod 755 docs/evidence/receipts/fate/
```

### Circuit Breaker Not Emitting Events
- Verify `drain_events()` is called after state changes
- Events are cleared on drain - ensure proper collection timing

### Identity Monitor Not Detecting Drift
- Confirm baseline was recorded with `record_identity_baseline()`
- Check that `enable_identity_monitor()` was called
- Verify fingerprint generation matches tier structure

---

## Performance Notes

| Feature | Overhead | Recommended Frequency |
|---------|----------|----------------------|
| Circuit Breaker Event Drain | < 1μs | Per request |
| FATE WAL Write | < 5ms | Per escalation (automatic) |
| Identity Drift Check | < 10ms | Every 5 minutes |

---

## See Also
- Full Implementation: `docs/PHASE_C_IMPLEMENTATION_SUMMARY.md`
- Circuit Breaker: `src/apex/circuit_breaker.rs`
- FATE: `src/fate.rs`
- Sovereign Runtime: `src/sovereign_runtime_omega.rs`
