---
allowed-tools: Bash(python*:*), Read, Grep, Glob, Write
description: Graph of Thoughts - Non-linear reasoning with node exploration
argument-hint: [problem-to-analyze]
---

# Graph of Thoughts (GoT) Reasoning

## Overview

Graph of Thoughts enables **non-linear reasoning** by modeling thoughts as a directed graph, allowing exploration of multiple reasoning paths, backtracking, and synthesis of parallel insights.

**Key Difference from Chain-of-Thought**:
- CoT: Linear `A -> B -> C -> Conclusion`
- GoT: Graph `A -> [B, C] -> [D, E] -> Synthesis -> Conclusion`

## Graph Structure

### Node Types

| Node Type | Symbol | Purpose | Example |
|-----------|--------|---------|---------|
| **Question** | `[Q]` | Initial problem/query | "How to optimize?" |
| **Analysis** | `[A]` | Examination of aspects | "Performance analysis" |
| **Evidence** | `[E]` | Facts, data, observations | "Latency = 200ms" |
| **Synthesis** | `[S]` | Combination of insights | "Combining A1 + A2" |
| **Decision** | `[D]` | Conclusion/choice | "Use caching" |

### Edge Types

| Edge Type | Symbol | Meaning |
|-----------|--------|---------|
| **supports** | `-->` | Evidence supports conclusion |
| **contradicts** | `--X` | Evidence contradicts |
| **elaborates** | `~~~>` | Provides more detail |
| **derives** | `==>` | Logical derivation |
| **depends** | `..>` | Prerequisite relationship |

## Current System Status

- GoT Orchestrator: !`ls -lh bizra_kernel/got_orchestrator.py 2>/dev/null || echo "Not found"`
- Knowledge Graph Nodes: !`curl -s http://localhost:7474/db/neo4j/tx/commit -u neo4j:${NEO4J_PASSWORD:-bizra} -H "Content-Type: application/json" -d '{"statements":[{"statement":"MATCH (n) RETURN count(n)"}]}' 2>/dev/null | jq -r '.results[0].data[0].row[0] // "N/A"' || echo "Neo4j not accessible"`
- Reasoning Engine: !`grep -l "ReasoningMethod" src/*.rs 2>/dev/null | head -1 || echo "Not found"`

## Your Task

### Phase 1: Problem Decomposition

Transform the problem into a root question node:

```
[Q] Root Question: {user's problem}
    |
    +-- [A1] Aspect 1: {first angle of analysis}
    |
    +-- [A2] Aspect 2: {second angle of analysis}
    |
    +-- [A3] Aspect 3: {third angle of analysis}
```

### Phase 2: Evidence Gathering

For each analysis node, gather evidence:

```
[A1] Aspect 1
    |
    +-- [E1.1] Evidence: {fact or observation}
    |       |
    |       +-- [E1.1.1] Sub-evidence: {supporting detail}
    |
    +-- [E1.2] Evidence: {another fact}
```

### Phase 3: Graph Traversal Strategy

Choose traversal based on problem type:

**BFS (Breadth-First)** - Use when:
- Need comprehensive coverage
- All aspects equally important
- Exploring solution space

**DFS (Depth-First)** - Use when:
- Following a promising lead
- Need to reach conclusions quickly
- Deep domain expertise needed

**Hybrid** - Use when:
- Complex multi-faceted problems
- Some paths more promising
- Need both depth and breadth

### Phase 4: Synthesis & Decision

Combine insights from multiple paths:

```
        [E1.1]    [E2.1]    [E3.1]
           \        |        /
            \       |       /
             v      v      v
        [S1] Synthesis Node: {combined insight}
                   |
                   v
        [D] Decision: {final conclusion}
```

## GoT Template

### Problem: [User's Problem]

---

#### Graph Construction

```
[Q] {Root Question}
    |
    +-- [A1] {Aspect 1}
    |   |
    |   +-- [E1.1] {Evidence}
    |   |       --> supports [A1]
    |   |
    |   +-- [E1.2] {Evidence}
    |           --> supports [A1]
    |
    +-- [A2] {Aspect 2}
    |   |
    |   +-- [E2.1] {Evidence}
    |   |       --> supports [A2]
    |   |
    |   +-- [E2.2] {Counter-evidence}
    |           --X contradicts [A2]
    |
    +-- [A3] {Aspect 3}
        |
        +-- [E3.1] {Evidence}
                ==> derives [S1]
```

#### Traversal Log

| Step | Node | Action | Insight |
|------|------|--------|---------|
| 1 | [Q] | Start | Problem framed |
| 2 | [A1] | BFS | First aspect explored |
| 3 | [E1.1] | DFS | Evidence gathered |
| ... | ... | ... | ... |

#### Synthesis Path

```
[E1.1] + [E2.1] --> [S1] Partial synthesis
[S1] + [E3.1] --> [S2] Combined synthesis
[S2] ==> [D] Final decision
```

#### Decision

**Conclusion**: {Final answer/decision}

**Confidence**: {High/Medium/Low}

**Supporting Nodes**: [E1.1, E2.1, E3.1]

**Contradicting Nodes**: [E2.2] (addressed by: {resolution})

---

## Graph Operations

### Node Creation
```python
def create_node(node_type, content, parent=None):
    """Create a new thought node"""
    node = {
        "id": generate_node_id(),
        "type": node_type,  # Q, A, E, S, D
        "content": content,
        "parent": parent,
        "children": [],
        "edges": []
    }
    return node
```

### Edge Creation
```python
def create_edge(source, target, edge_type):
    """Create an edge between nodes"""
    edge = {
        "source": source.id,
        "target": target.id,
        "type": edge_type,  # supports, contradicts, etc.
        "weight": calculate_weight(source, target, edge_type)
    }
    return edge
```

### Path Finding
```python
def find_strongest_path(root, goal):
    """Find the path with strongest evidence chain"""
    paths = bfs_all_paths(root, goal)
    scored_paths = [(path, score_path(path)) for path in paths]
    return max(scored_paths, key=lambda x: x[1])
```

## Validation Checks

### Graph Validity

- [ ] Root question node exists
- [ ] All analysis nodes connected to root
- [ ] Evidence nodes have edge types
- [ ] Synthesis nodes have multiple inputs
- [ ] Decision node has synthesis support

### Reasoning Quality

- [ ] No orphan nodes (disconnected)
- [ ] Contradictions addressed
- [ ] Evidence sufficiently deep
- [ ] Synthesis logically sound
- [ ] Decision confidence justified

## Evidence Generation

Generate GoT reasoning receipt:

```json
{
  "receipt_id": "got-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "problem": "[root question]",
  "graph": {
    "nodes": {
      "questions": 1,
      "analyses": 3,
      "evidence": 6,
      "syntheses": 2,
      "decisions": 1
    },
    "edges": {
      "supports": 5,
      "contradicts": 1,
      "elaborates": 2,
      "derives": 3
    },
    "depth": 4,
    "width": 3
  },
  "traversal": {
    "strategy": "hybrid",
    "steps": 15,
    "backtracks": 2
  },
  "decision": {
    "conclusion": "",
    "confidence": 0.85,
    "supporting_nodes": [],
    "contradictions_resolved": 1
  },
  "integrity_hash": ""
}
```

## Report Format

```
## Graph of Thoughts Analysis

**Problem**: [root question]
**Traversal Strategy**: [BFS/DFS/Hybrid]
**Timestamp**: [ISO timestamp]

### Graph Structure

```
[Visualization of the graph]
```

### Node Summary

| Type | Count | Key Insights |
|------|-------|--------------|
| Questions | 1 | [root problem] |
| Analyses | 3 | [aspects covered] |
| Evidence | 6 | [facts gathered] |
| Syntheses | 2 | [combinations made] |
| Decisions | 1 | [final conclusion] |

### Traversal Path

[Step-by-step traversal log]

### Synthesis

[How evidence was combined]

### Decision

**Conclusion**: [answer]
**Confidence**: [level]
**Key Evidence**: [supporting nodes]

### Receipt
- ID: got-[timestamp]
- Location: docs/evidence/receipts/
```

---

**GoT Philosophy**: "Linear thinking misses connections. Graph thinking reveals the hidden structure of problems. Explore widely, synthesize deeply."
