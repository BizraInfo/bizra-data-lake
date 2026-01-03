# BIZRA Resilience Mesh - Chaos Testing Runbook

## Overview

This runbook provides operational procedures for conducting chaos engineering tests on the BIZRA Resilience Mesh. Chaos testing validates the system's ability to withstand and recover from various failure scenarios while maintaining the guaranteed MTTR ≤30 seconds.

## Chaos Scenarios

### 1. Severed Link Scenario (Primary)

**Description**: Simulates network partition between federation nodes.

**Objective**: Validate automatic failover and self-healing capabilities.

**MTTR Target**: ≤30 seconds

#### Execution Steps

```python
from bizra_kernel.federation_manager import create_federation_node

# Initialize federation
manager = await create_federation_node("node-01", ["node-02", "node-03"])

# Trigger Severed Link scenario
event_id = await manager.trigger_severed_link_scenario(
    affected_nodes=["node-02"],
    isolated_from=["node-01", "node-03"],
    duration_seconds=30
)

print(f"Chaos event triggered: {event_id}")
```

#### Monitoring

```python
# Monitor chaos status
chaos_status = manager.get_chaos_status()
print(f"Active partitions: {chaos_status['active_partitions']}")

# Check MTTR progress
mttr_report = manager.get_mttr_report()
print(f"Current MTTR: {mttr_report['average_mttr']}s")
```

#### Success Criteria

- [ ] Network partition created successfully
- [ ] Automatic failover detected within 5 seconds
- [ ] New leader elected
- [ ] MTTR ≤30 seconds
- [ ] Full recovery achieved
- [ ] No data loss

#### Rollback Procedure

The scenario automatically self-heals after the specified duration. Manual intervention only required if:

```python
# Force immediate recovery
await manager.chaos_engine._remove_network_partition(partition_config)
```

### 2. Node Crash Scenario

**Description**: Simulates complete node failure.

**Objective**: Test node failure recovery and shard rebalancing.

#### Execution Steps

```python
# Simulate node crash (external process termination)
# Monitor federation response
status = manager.get_federation_status()
failed_nodes = [node for node in status['federation_status']['active_nodes']
                if node not in ['node-01', 'node-02', 'node-03']]

if failed_nodes:
    print(f"Detected failed nodes: {failed_nodes}")
    # Automatic recovery initiated
```

#### Verification

```bash
# Check node recovery
curl http://localhost:8888/health

# Verify shard rebalancing
status = manager.get_federation_status()
print(f"Shard distribution: {status['sharding_stats']['load_distribution']}")
```

### 3. Leader Isolation Scenario

**Description**: Isolates the current leader node from the federation.

**Objective**: Validate leader election under stress.

#### Execution Steps

```python
# Identify current leader
status = manager.get_federation_status()
leader = status['federation_status']['leader_node']

# Isolate leader
event_id = await manager.trigger_severed_link_scenario(
    affected_nodes=[leader],
    isolated_from=[node for node in ['node-01', 'node-02', 'node-03'] if node != leader],
    duration_seconds=45  # Slightly longer for leader election
)
```

## Automated Chaos Testing

### Continuous Chaos Monitoring

```python
# Start continuous chaos monitoring
await manager.chaos_engine.start_chaos_monitoring()

# Monitor for 24 hours
import asyncio
await asyncio.sleep(86400)  # 24 hours

# Generate MTTR report
mttr_report = manager.get_mttr_report()
print(f"24h MTTR Report: {mttr_report}")
```

### Scheduled Chaos Tests

```bash
# Daily chaos test schedule
0 2 * * * /path/to/chaos_test.py --scenario severed_link --duration 30
0 14 * * * /path/to/chaos_test.py --scenario node_crash --duration 60
```

## Performance Benchmarking

### MTTR Measurement

```python
from bizra_kernel.performance_suite import ChaosEngineeringBenchmark

chaos_bench = ChaosEngineeringBenchmark()

# Benchmark MTTR for all scenarios
for scenario in chaos_bench.failure_scenarios:
    result = chaos_bench.benchmark_mttr(scenario, iterations=10)
    print(f"{scenario}: {result['avg_mttr_ms']:.2f}ms MTTR")
```

### Chaos Resilience Testing

```python
# Run continuous chaos for 5 minutes
result = chaos_bench.benchmark_chaos_resilience(duration_sec=300)
print(f"Availability: {result['availability_percent']:.2f}%")
print(f"Total incidents: {result['total_incidents']}")
```

## Alerting and Monitoring

### Chaos Event Alerts

```python
async def chaos_alert_handler(alert_type, alert_data):
    """Handle chaos engineering alerts."""
    if alert_type == "CHAOS_TRIGGERED":
        print(f"🚨 Chaos event: {alert_data['data']['event_id']}")
    elif alert_type == "MTTR_TARGET_EXCEEDED":
        print(f"❌ MTTR violation: {alert_data['data']['mttr_seconds']}s > 30s")
        # Escalate to on-call engineer
    elif alert_type == "FAILOVER_DETECTED":
        print(f"✅ Automatic failover: {alert_data['data']['new_leader']}")

# Register alert handler
manager.chaos_engine.register_alert_callback(chaos_alert_handler)
```

### Key Metrics to Monitor

- **MTTR**: Mean Time To Recovery (target: ≤30s)
- **Failover Time**: Time to elect new leader
- **Recovery Success Rate**: Percentage of successful recoveries
- **Health Score Degradation**: Maximum health score drop during chaos
- **Consensus Rounds**: Number of consensus rounds during recovery

## Incident Response

### MTTR Target Violation

**Trigger**: MTTR > 30 seconds

**Response**:

1. **Immediate Assessment**

   ```python
   # Get detailed MTTR report
   mttr_report = manager.get_mttr_report()
   print(f"Violations: {mttr_report['target_violations']}")
   ```

2. **Root Cause Analysis**

   ```python
   # Check federation status during violation
   status = manager.get_federation_status()
   print(f"Health score: {status['federation_status']['health_score']}")
   ```

3. **Mitigation**
   - Review network configuration
   - Check system resources
   - Validate consensus parameters
   - Update MTTR targets if necessary

### Recovery Failure

**Trigger**: Chaos scenario fails to recover automatically

**Response**:

1. **Manual Recovery**

   ```python
   # Force consensus reset
   await manager.consensus_engine.reset_consensus_state()

   # Trigger manual rebalancing
   await manager._failure_recovery_loop()
   ```

2. **System Restart**

   ```bash
   # Graceful restart of affected nodes
   systemctl restart bizra-federation
   ```

## Best Practices

### Test Environment Setup

- **Isolation**: Run chaos tests in staging environment first
- **Gradual Rollout**: Start with short-duration, low-impact scenarios
- **Monitoring**: Ensure comprehensive monitoring before chaos tests
- **Backup**: Maintain recent backups before destructive tests

### Production Chaos Testing

- **Schedule**: Run during low-traffic maintenance windows
- **Communication**: Notify stakeholders before production chaos tests
- **Rollback Plan**: Have immediate rollback procedures ready
- **Success Criteria**: Define clear success metrics for each test

### Continuous Improvement

- **Trend Analysis**: Track MTTR trends over time
- **Scenario Expansion**: Add new failure scenarios as system evolves
- **Automation**: Increase automation of chaos testing procedures
- **Documentation**: Update runbooks based on lessons learned

## Compliance and Reporting

### Regulatory Requirements

- **Audit Trail**: All chaos events logged with timestamps
- **Impact Assessment**: Document potential business impact of failures
- **Recovery Testing**: Regular validation of recovery procedures
- **Performance Guarantees**: Maintain MTTR ≤30s SLA

### Reporting

```python
# Generate chaos testing report
def generate_chaos_report():
    mttr_report = manager.get_mttr_report()
    chaos_status = manager.get_chaos_status()

    report = {
        "period": "Last 30 days",
        "total_events": len(manager.chaos_engine.chaos_events),
        "mttr_compliance": mttr_report['target_compliance_rate'],
        "average_mttr": mttr_report['average_mttr'],
        "successful_recoveries": mttr_report['total_events'] - mttr_report['target_violations']
    }

    return report
```

## Emergency Contacts

- **Chaos Engineering Team**: <chaos@bizra.ai>
- **SRE On-Call**: <sre-oncall@bizra.ai>
- **Security Incident Response**: <security@bizra.ai>

## References

- [Resilience Mesh Deployment Guide](README.md)
- [Performance SLAs and Guarantees](performance-slas.md)
- [Federation Architecture Documentation](../../architecture/federation.md)

---

*This runbook is maintained for the BIZRA Resilience Mesh Phase 9. Regular updates required for new chaos scenarios and procedures.*
