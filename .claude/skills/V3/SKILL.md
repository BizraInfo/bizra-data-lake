---
name: "V3"
description: "Claude-Flow V3 implementation hub. Routes to the appropriate v3 specialist skill for architecture, security, memory, performance, integration, and swarm coordination."
version: "1.0.0"
category: "development"
tags: ["v3", "claude-flow", "architecture", "hub"]
---

# Claude-Flow V3 Hub

Central routing skill for all V3 implementation work. Use this when you want V3 help without remembering the specific sub-skill name.

## V3 Sub-Skills

| Skill | Domain | What It Does |
|-------|--------|-------------|
| `v3-swarm-coordination` | Orchestration | 15-agent hierarchical mesh, phase management |
| `v3-security-overhaul` | Security | CVE remediation, threat modeling, secure-by-default |
| `v3-core-implementation` | Core | DDD domains, clean architecture, dependency injection |
| `v3-memory-unification` | Memory | AgentDB, HNSW indexing, 150x-12,500x search improvement |
| `v3-integration-deep` | Integration | agentic-flow@alpha integration, eliminate 10K+ duplicate lines |
| `v3-performance-optimization` | Performance | 2.49x-7.47x Flash Attention, benchmarking suite |
| `v3-ddd-architecture` | Architecture | Bounded contexts, microkernel pattern, clean separation |
| `v3-mcp-optimization` | MCP | Connection pooling, load balancing, sub-100ms responses |
| `v3-cli-modernization` | CLI | Interactive prompts, hooks integration, workflow automation |

## Quick Start

### Full Swarm (all 15 agents)
```bash
# Initialize complete V3 implementation swarm
Task("V3 swarm init", "Initialize 15-agent hierarchical mesh for v3", "v3-queen-coordinator")
```

### By Domain
```bash
# Security first (Phase 1)
Task("V3 security", "CVE remediation and threat modeling", "v3-security-architect")

# Core systems (Phase 2)
Task("V3 core", "DDD architecture implementation", "v3-core-implementation")
Task("V3 memory", "AgentDB unification with HNSW", "v3-memory-specialist")

# Integration (Phase 3)
Task("V3 integration", "Deep agentic-flow@alpha integration", "v3-integration-architect")

# Performance (Phase 4)
Task("V3 performance", "Benchmark validation and optimization", "v3-performance-engineer")
```

## 14-Week Timeline

| Phase | Weeks | Agents | Focus |
|-------|-------|--------|-------|
| Foundation | 1-2 | #1-6 | Security + Architecture |
| Core Systems | 3-6 | #1, #5-9, #13 | Memory + Swarm + MCP |
| Integration | 7-10 | #1, #10-14 | Integration + CLI + Performance |
| Release | 11-14 | All 15 | Optimization + Testing + Release |

## 10 ADRs

1. **ADR-001**: Deep agentic-flow@alpha integration (eliminate duplicates)
2. **ADR-002**: Security-first architecture
3. **ADR-003**: DDD modular architecture
4. **ADR-004**: Unified swarm coordination engine
5. **ADR-005**: MCP server optimization
6. **ADR-006**: Unified memory service (AgentDB)
7. **ADR-007**: CLI modernization with hooks
8. **ADR-008**: SONA learning integration
9. **ADR-009**: Hybrid memory backend (HNSW)
10. **ADR-010**: Performance benchmarking framework

## Success Targets

- **Performance**: 2.49x-7.47x Flash Attention speedup
- **Search**: 150x-12,500x AgentDB improvement
- **Code Size**: <5,000 lines (from 15,000+)
- **Security Score**: 90/100
- **Test Coverage**: >80%
- **Memory**: 50-75% reduction
