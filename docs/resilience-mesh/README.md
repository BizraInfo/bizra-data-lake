# BIZRA Resilience Mesh - Production Deployment Guide

## Overview

The BIZRA Resilience Mesh is a production-ready, 3-node local federation system designed for high-availability distributed AI operations. This guide provides comprehensive instructions for deploying and operating the Resilience Mesh in production environments.

## Architecture

The Resilience Mesh consists of:

- **Federation Manager**: Orchestrates the complete federation lifecycle
- **Consensus Engine**: Ensures distributed agreement across nodes
- **Knowledge Graph Sharding**: Distributes knowledge across federation nodes
- **Chaos Engine**: Implements proactive failure testing and self-healing
- **Graph Reasoning Federation**: Enables distributed Graph-of-Thoughts reasoning

## Prerequisites

### System Requirements

- **Nodes**: Minimum 3 nodes for production deployment
- **OS**: Linux/Windows with Python 3.8+
- **Network**: Reliable inter-node communication (TCP/UDP)
- **Storage**: 100GB+ per node for knowledge graph persistence
- **Memory**: 16GB+ RAM per node
- **CPU**: 4+ cores per node

### Software Dependencies

```bash
pip install -r requirements-kernel.txt
```

### Network Configuration

- **Ports**: Default 8888 (configurable)
- **Firewall**: Allow inter-node communication on federation ports
- **DNS**: Nodes must be able to resolve peer hostnames

## Deployment Steps

### 1. Node Preparation

For each federation node:

```bash
# Clone the repository
git clone <repository-url>
cd bizra-resilience-mesh

# Install dependencies
pip install -r requirements-kernel.txt

# Configure node identity
export BIZRA_NODE_ID="node-01"
export BIZRA_PEER_NODES="node-02,node-03"
export BIZRA_PORT="8888"
```

### 2. Configuration

Create `federation_config.json` for each node:

```json
{
  "node_id": "node-01",
  "peer_nodes": ["node-02", "node-03"],
  "port": 8888,
  "consensus_quorum": 2,
  "heartbeat_interval": 1.0,
  "election_timeout": 3.0,
  "shard_replication": 2,
  "reasoning_timeout": 30.0,
  "chaos_enabled": true,
  "mttr_target_seconds": 30.0
}
```

### 3. Initial Deployment

Start each node in sequence:

```bash
# Node 01
python -m bizra_kernel.federation_manager node-01 node-02 node-03

# Node 02
python -m bizra_kernel.federation_manager node-02 node-01 node-03

# Node 03
python -m bizra_kernel.federation_manager node-03 node-01 node-02
```

### 4. Health Verification

Verify federation health:

```bash
curl http://localhost:8888/health
# Expected: {"status": "healthy", "federation_active": true}
```

## Operations

### Monitoring

The Resilience Mesh provides comprehensive monitoring:

```python
from bizra_kernel.federation_manager import create_federation_node

manager = await create_federation_node("node-01", ["node-02", "node-03"])
status = manager.get_federation_status()

print(f"Health Score: {status['federation_status']['health_score']}")
print(f"Active Nodes: {status['federation_status']['active_nodes']}")
print(f"Leader: {status['federation_status']['leader_node']}")
```

### Scaling Operations

#### Adding Nodes

```python
# Submit consensus request to add new node
await manager.submit_federated_request("add_node", {
    "new_node_id": "node-04",
    "node_config": {...}
})
```

#### Shard Rebalancing

The system automatically rebalances shards on node failures or additions.

### Knowledge Management

#### Adding Knowledge Entities

```python
result = await manager.add_knowledge_entity("entity-001", {
    "entity": "AI Ethics",
    "fact": "Ethical AI requires transparency and accountability",
    "rels": {"domain": "ethics", "importance": "high"}
})
```

#### Distributed Reasoning

```python
session_id = await manager.initiate_distributed_reasoning(
    "What are the ethical implications of autonomous AI systems?"
)
```

## High Availability

### Automatic Failover

The Resilience Mesh implements automatic failover with:

- **MTTR Target**: ≤30 seconds
- **Consensus Quorum**: 2 out of 3 nodes
- **Leader Election**: Automatic on failure detection
- **Shard Replication**: 2x replication factor

### Chaos Engineering

Built-in chaos testing ensures resilience:

```python
# Trigger network partition test
event_id = await manager.trigger_severed_link_scenario(
    affected_nodes=["node-02"],
    isolated_from=["node-01", "node-03"],
    duration_seconds=30
)

# Monitor MTTR
mttr_report = manager.get_mttr_report()
print(f"Average MTTR: {mttr_report['average_mttr']}s")
```

## Troubleshooting

### Common Issues

#### Node Connectivity

```bash
# Check network connectivity
telnet <peer-node> 8888

# Verify federation status
curl http://localhost:8888/federation/status
```

#### Consensus Failures

```bash
# Check consensus logs
tail -f federation_consensus.log

# Manual leader election trigger
await manager.consensus_engine.trigger_election()
```

#### Performance Degradation

```bash
# Run performance diagnostics
python -m bizra_kernel.performance_suite --federation --nodes 3

# Check shard distribution
status = manager.get_federation_status()
print(status['sharding_stats'])
```

## Security Considerations

### Network Security

- Use TLS for inter-node communication in production
- Implement network segmentation
- Regular security audits of federation traffic

### Access Control

- Node authentication via certificates
- Request signing for consensus operations
- Audit logging of all federation activities

## Performance Optimization

### Tuning Parameters

```json
{
  "heartbeat_interval": 0.5,      // Faster failure detection
  "election_timeout": 2.0,        // Quicker leader election
  "shard_replication": 3,         // Higher redundancy
  "reasoning_timeout": 15.0       // Faster reasoning sessions
}
```

### Monitoring Dashboards

Set up monitoring for:

- Federation health score
- Consensus round latency
- Shard distribution balance
- MTTR compliance
- Node resource utilization

## Backup and Recovery

### State Persistence

Federation state is automatically persisted:

```python
# Manual state backup
manager._save_federation_state()

# State restoration on restart
# Automatic on federation start
```

### Disaster Recovery

1. **Node Loss**: Automatic rebalancing within MTTR target
2. **Network Partition**: Self-healing partition recovery
3. **Full Federation Loss**: Restore from state backups

## Compliance and Certification

The Resilience Mesh is designed for production use with:

- **Uptime SLA**: 99.99% availability
- **MTTR Guarantee**: ≤30 seconds
- **Data Consistency**: Strong consistency via consensus
- **Audit Trail**: Complete operation logging

## Support and Maintenance

### Regular Maintenance

- **Weekly**: Review MTTR reports and chaos test results
- **Monthly**: Performance benchmarking and optimization
- **Quarterly**: Security audits and dependency updates

### Emergency Contacts

- **Technical Support**: [support@bizra.ai](mailto:support@bizra.ai)
- **Security Issues**: [security@bizra.ai](mailto:security@bizra.ai)
- **Documentation**: [docs.bizra.ai/resilience-mesh](https://docs.bizra.ai/resilience-mesh)

---

*This deployment guide is for the BIZRA Resilience Mesh Phase 9 implementation. For the latest updates, refer to the official documentation.*
