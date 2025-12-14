# Example API Usage

## Starting the Server

```bash
RUST_LOG=info cargo run --release
```

The server will start on port 8080.

## Example Requests

### 1. Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-02T17:46:31.951Z"
}
```

### 2. System Information

```bash
curl http://localhost:8080/
```

Response:
```json
{
  "name": "BIZRA META ALPHA ELITE - Complete Unified System",
  "version": "2.0.0",
  "architecture": "PAT(7) + SAT(5) + Full Arsenal",
  "capabilities": [
    "MCP Integration",
    "A2A Protocol",
    "Multi-Reasoning (CoT, ToT, GoT, ReAct, Reflexion)",
    "Sub-Agent Spawning",
    "Swarm Intelligence",
    "Hook System",
    "Slash Commands"
  ],
  "status": "PRODUCTION",
  "ihsan": "إحسان"
}
```

### 3. Basic Dual-Agentic Execution

```bash
curl -X POST http://localhost:8080/dual/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "task": "Design a microservices architecture for e-commerce platform",
    "requirements": [
      "scalable",
      "secure",
      "fault-tolerant",
      "high-performance"
    ],
    "target": "architecture_design",
    "priority": "High"
  }'
```

Response:
```json
{
  "pat_contributions": [
    "[Strategic] Long-term vision for 'Design a microservices architecture...': Establish foundation for sustainable growth",
    "[Innovation] Novel approach for 'Design a microservices architecture...': Apply cutting-edge methodologies",
    "[Analysis] Data-driven insights for 'Design a microservices architecture...': Optimize for 95% efficiency",
    "[Implementation] Practical execution plan for 'Design a microservices architecture...': 5-phase delivery",
    "[Quality] Excellence standards for 'Design a microservices architecture...': إحسان score target 0.95+",
    "[UX] User-centric design for 'Design a microservices architecture...': Optimize for user satisfaction",
    "[Coordination] Harmonized approach for 'Design a microservices architecture...': Ensure seamless integration"
  ],
  "sat_contributions": [
    "[Security] No security issues detected in 7 PAT contributions",
    "[Ethics] All 7 PAT contributions ethically aligned",
    "[Performance] Average execution time: 150ns",
    "[Consistency] Logical coherence validated across 7 contributions",
    "[Resources] Optimal resource utilization: 87% efficiency"
  ],
  "synergy_score": 0.925,
  "ihsan_score": 0.952,
  "latency": 47,
  "meta": {
    "pat_agents": 7,
    "sat_agents": 5,
    "validation_time_ms": 0
  }
}
```

### 4. Enhanced Execution with Multi-Reasoning

```bash
curl -X POST http://localhost:8080/enhanced/execute \
  -H "Content-Type: application/json" \
  -d '{
    "base": {
      "user_id": "user_002",
      "task": "Create strategic roadmap for BIZRA ecosystem expansion",
      "requirements": ["innovation", "sustainability", "scalability"],
      "target": "strategic_roadmap",
      "priority": "Critical"
    },
    "reasoning_preference": "GraphOfThought",
    "enable_sub_agents": false
  }'
```

Response includes reasoning steps in metadata:
```json
{
  "pat_contributions": [...],
  "sat_contributions": [],
  "synergy_score": 0.92,
  "ihsan_score": 0.95,
  "latency": 50,
  "meta": {
    "reasoning_method": "GraphOfThought",
    "mcp_tools_used": 4,
    "sub_agents_spawned": 0
  }
}
```

### 5. Slash Command: List Tools

```bash
curl -X POST http://localhost:8080/enhanced/execute \
  -H "Content-Type: application/json" \
  -d '{
    "base": {
      "user_id": "user_003",
      "task": "Find available search tools",
      "requirements": [],
      "target": "tool_discovery"
    },
    "slash_command": {
      "type": "Tools",
      "filter": "search"
    }
  }'
```

Response:
```json
{
  "pat_contributions": [
    "web_search: Search the web"
  ],
  "sat_contributions": [],
  "synergy_score": 1.0,
  "ihsan_score": 1.0,
  "latency": 10,
  "meta": {
    "slash_command": "tools",
    "filter": "search",
    "count": 1
  }
}
```

### 6. Slash Command: Spawn Sub-Agent

```bash
curl -X POST http://localhost:8080/enhanced/execute \
  -H "Content-Type: application/json" \
  -d '{
    "base": {
      "user_id": "user_004",
      "task": "Complex research project requiring specialization",
      "requirements": ["depth", "breadth"],
      "target": "research_output"
    },
    "enable_sub_agents": true,
    "slash_command": {
      "type": "Spawn",
      "role": "Market Research Specialist",
      "task": "Analyze competitive landscape for BIZRA ecosystem"
    }
  }'
```

Response:
```json
{
  "pat_contributions": [
    "Spawned sub-agent 'Market Research Specialist' for task: Analyze competitive landscape for BIZRA ecosystem"
  ],
  "sat_contributions": [],
  "synergy_score": 0.95,
  "ihsan_score": 0.93,
  "latency": 50,
  "meta": {
    "slash_command": "spawn",
    "sub_agent_role": "Market Research Specialist",
    "total_sub_agents": 1
  }
}
```

### 7. Slash Command: Force Reasoning Method

```bash
curl -X POST http://localhost:8080/enhanced/execute \
  -H "Content-Type: application/json" \
  -d '{
    "base": {
      "user_id": "user_005",
      "task": "Improve code quality through self-reflection",
      "requirements": ["excellence", "maintainability"],
      "target": "code_improvement"
    },
    "slash_command": {
      "type": "Reason",
      "method": "Reflexion"
    }
  }'
```

Response includes reflexion steps:
```json
{
  "pat_contributions": [
    "Reflexive improvement completed: High-quality solution for 'Improve code quality through self-reflection'"
  ],
  "sat_contributions": [],
  "synergy_score": 0.93,
  "ihsan_score": 0.92,
  "latency": 100,
  "meta": {
    "slash_command": "reason",
    "method": "Reflexion",
    "steps": [
      "Iteration 1: Initial solution for 'Improve code quality through self-reflection'",
      "Self-Critique: Solution lacks depth in area X",
      "Iteration 2: Enhanced solution addressing critique",
      "Self-Critique: Edge case Y not covered",
      "Iteration 3: Comprehensive solution covering all cases",
      "Self-Critique: Solution meets all quality standards",
      "Final: Refined solution after 3 reflexion iterations for 'Improve code quality through self-reflection'"
    ]
  }
}
```

## Statistics Endpoint

```bash
curl http://localhost:8080/stats
```

Response:
```json
{
  "pat_agents": 7,
  "sat_agents": 5,
  "total_agents": 12,
  "reasoning_methods": 5,
  "mcp_tools": 4,
  "uptime": "operational"
}
```

## Performance Testing

### Benchmarking with Apache Bench

```bash
# 1000 requests, 10 concurrent
ab -n 1000 -c 10 -p request.json -T application/json http://localhost:8080/dual/execute
```

### Load Testing with wrk

```bash
wrk -t4 -c100 -d30s --latency http://localhost:8080/health
```

Expected results:
- P50 latency: < 30ms
- P99 latency: < 100ms
- Throughput: 1000+ req/sec

## Integration Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function executeTask() {
  const response = await axios.post('http://localhost:8080/dual/execute', {
    user_id: 'js_user',
    task: 'Optimize frontend performance',
    requirements: ['speed', 'UX'],
    target: 'optimization',
    priority: 'High'
  });
  
  console.log('Synergy Score:', response.data.synergy_score);
  console.log('إحسان Score:', response.data.ihsan_score);
}
```

### Python

```python
import requests

def execute_task():
    response = requests.post('http://localhost:8080/dual/execute', json={
        'user_id': 'py_user',
        'task': 'Design machine learning pipeline',
        'requirements': ['scalable', 'reproducible'],
        'target': 'ml_pipeline',
        'priority': 'High'
    })
    
    data = response.json()
    print(f"Synergy Score: {data['synergy_score']}")
    print(f"إحسان Score: {data['ihsan_score']}")
```

### cURL Scripts

```bash
#!/bin/bash
# batch_requests.sh

for i in {1..10}; do
  echo "Request $i"
  curl -s -X POST http://localhost:8080/dual/execute \
    -H "Content-Type: application/json" \
    -d "{
      \"user_id\": \"batch_user_$i\",
      \"task\": \"Process batch item $i\",
      \"requirements\": [\"speed\"],
      \"target\": \"batch_result\"
    }" | jq '.synergy_score'
done
```

## Environment Variables

```bash
# Logging level
export RUST_LOG=info  # trace, debug, info, warn, error

# Server port (default: 8080)
export SERVER_PORT=8080

# Maximum sub-agents (default: 100)
export MAX_SUB_AGENTS=100
```

## Troubleshooting

### Server won't start
- Check if port 8080 is available: `lsof -i :8080`
- Check logs: `RUST_LOG=debug cargo run --release`

### High latency
- Check system resources: `top`, `htop`
- Monitor with: `curl http://localhost:8080/stats`

### Connection refused
- Ensure server is running: `ps aux | grep meta_alpha`
- Check firewall rules: `sudo ufw status`
