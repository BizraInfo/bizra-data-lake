---
allowed-tools: Bash(python*:*), Bash(cargo:*), Read, Write, Grep, Glob
description: Execute DAG-based workflows with dependency management
argument-hint: [workflow-name | --define]
---

# Flow - Claude-Flow Pipeline Orchestration

## Overview

The Flow command enables **DAG-based workflow execution** where tasks are organized as a directed acyclic graph with explicit dependencies. Inspired by claude-flow patterns for complex multi-step operations.

## DAG Workflow Architecture

```
     ┌────────────────────────────────────────────────────┐
     │                  WORKFLOW DAG                       │
     ├────────────────────────────────────────────────────┤
     │                                                     │
     │     [Start]                                         │
     │        │                                            │
     │        ▼                                            │
     │     [T1: Research]                                  │
     │        │                                            │
     │        ├──────────────┬───────────────┐            │
     │        ▼              ▼               ▼            │
     │     [T2: Design]  [T3: Analyze]  [T4: Review]     │
     │        │              │               │            │
     │        └──────────────┼───────────────┘            │
     │                       ▼                            │
     │                [T5: Implement]                     │
     │                       │                            │
     │              ┌────────┴────────┐                   │
     │              ▼                 ▼                   │
     │        [T6: Test]      [T7: Document]             │
     │              │                 │                   │
     │              └────────┬────────┘                   │
     │                       ▼                            │
     │                 [T8: Deploy]                       │
     │                       │                            │
     │                       ▼                            │
     │                    [End]                           │
     │                                                     │
     └────────────────────────────────────────────────────┘
```

## Flow Definition Schema

```yaml
# .claude/workflows/example-flow.yaml
name: example-flow
description: Example workflow demonstrating DAG patterns
version: 1.0

# Global settings
settings:
  timeout_ms: 300000
  retry_count: 3
  parallel_limit: 4

# Task definitions
tasks:
  - id: research
    name: Research Phase
    agent: MasterReasoner
    action: research
    inputs:
      query: "${workflow.input.topic}"
    outputs:
      - findings

  - id: design
    name: Design Phase
    agent: CodeArchitect
    action: design
    depends_on: [research]
    inputs:
      requirements: "${research.outputs.findings}"
    outputs:
      - architecture

  - id: implement
    name: Implementation
    agent: RustExpert  # or PythonExpert
    action: code
    depends_on: [design]
    inputs:
      spec: "${design.outputs.architecture}"
    outputs:
      - code_changes

  - id: test
    name: Testing
    agent: TestRunner
    action: test
    depends_on: [implement]
    inputs:
      changes: "${implement.outputs.code_changes}"
    outputs:
      - test_results

  - id: validate
    name: Validation
    agent: IhsanValidator
    action: validate
    depends_on: [test]
    inputs:
      results: "${test.outputs.test_results}"
    outputs:
      - validation_report
```

## Current System Status

- Flow Definitions: !`ls .claude/workflows/*.yaml 2>/dev/null | wc -l || echo "0"`
- Active Workflows: !`ls /tmp/bizra-flows/*.json 2>/dev/null | wc -l || echo "0"`
- Agent Factory: !`ls -lh core/agent_factory.py 2>/dev/null || echo "Not found"`

## Your Task

### Phase 1: Define Workflow DAG

**Identify Tasks**:
1. List all discrete steps required
2. Identify dependencies between steps
3. Determine which steps can run in parallel

**Create DAG**:
```python
from collections import defaultdict

class WorkflowDAG:
    def __init__(self):
        self.tasks = {}
        self.dependencies = defaultdict(list)
        self.reverse_deps = defaultdict(list)

    def add_task(self, task_id, config):
        self.tasks[task_id] = config
        for dep in config.get('depends_on', []):
            self.dependencies[task_id].append(dep)
            self.reverse_deps[dep].append(task_id)

    def get_ready_tasks(self, completed):
        """Get tasks whose dependencies are all completed"""
        ready = []
        for task_id, task in self.tasks.items():
            if task_id in completed:
                continue
            deps = self.dependencies[task_id]
            if all(d in completed for d in deps):
                ready.append(task_id)
        return ready

    def topological_sort(self):
        """Return execution order respecting dependencies"""
        # Kahn's algorithm
        ...
```

### Phase 2: Validate DAG

**Cycle Detection**:
```python
def detect_cycles(dag):
    """Ensure DAG has no cycles"""
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in dag.dependencies[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for task in dag.tasks:
        if task not in visited:
            if dfs(task):
                raise ValueError(f"Cycle detected involving {task}")

    return False  # No cycles
```

**Dependency Validation**:
```python
def validate_dependencies(dag):
    """Ensure all dependencies exist"""
    for task_id, deps in dag.dependencies.items():
        for dep in deps:
            if dep not in dag.tasks:
                raise ValueError(f"Task {task_id} depends on unknown task {dep}")
```

### Phase 3: Execute Flow

**Execution Engine**:
```python
class FlowExecutor:
    def __init__(self, dag, agent_factory):
        self.dag = dag
        self.factory = agent_factory
        self.completed = set()
        self.results = {}

    async def execute(self, inputs):
        """Execute workflow with given inputs"""
        self.context = {"workflow": {"input": inputs}}

        while len(self.completed) < len(self.dag.tasks):
            # Get tasks ready to run
            ready = self.dag.get_ready_tasks(self.completed)

            if not ready:
                # Deadlock detection
                if len(self.completed) < len(self.dag.tasks):
                    raise RuntimeError("Workflow deadlocked")
                break

            # Execute ready tasks in parallel
            results = await asyncio.gather(*[
                self.execute_task(task_id)
                for task_id in ready
            ])

            # Update completed and results
            for task_id, result in zip(ready, results):
                self.completed.add(task_id)
                self.results[task_id] = result
                self.context[task_id] = {"outputs": result}

        return self.results

    async def execute_task(self, task_id):
        """Execute single task"""
        task = self.dag.tasks[task_id]

        # Resolve inputs from context
        inputs = self.resolve_inputs(task.get('inputs', {}))

        # Spawn agent
        agent = await self.factory.spawn_pat(task['agent'])

        # Execute
        result = await agent.execute(
            action=task['action'],
            inputs=inputs
        )

        return result

    def resolve_inputs(self, inputs):
        """Resolve variable references in inputs"""
        resolved = {}
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("${"):
                # Parse reference like ${task.outputs.field}
                path = value[2:-1].split('.')
                resolved[key] = self.get_from_context(path)
            else:
                resolved[key] = value
        return resolved
```

### Phase 4: Monitor Progress

**Progress Tracking**:
```
Workflow: example-flow
Status: RUNNING
Progress: 3/8 tasks (37.5%)

┌──────────────────────────────────────────────────────────┐
│ Task           │ Status    │ Duration │ Agent            │
├──────────────────────────────────────────────────────────┤
│ research       │ COMPLETE  │ 2.3s     │ MasterReasoner   │
│ design         │ COMPLETE  │ 1.8s     │ CodeArchitect    │
│ analyze        │ COMPLETE  │ 1.5s     │ DataAnalyzer     │
│ implement      │ RUNNING   │ -        │ RustExpert       │
│ test           │ WAITING   │ -        │ TestRunner       │
│ validate       │ WAITING   │ -        │ IhsanValidator   │
└──────────────────────────────────────────────────────────┘
```

## Flow Template

### Workflow: [Workflow Name]

---

#### DAG Definition

```yaml
name: [workflow-name]
description: [description]

tasks:
  - id: [task1]
    name: [Task 1 Name]
    agent: [AgentType]
    action: [action]
    outputs: [outputs]

  - id: [task2]
    name: [Task 2 Name]
    agent: [AgentType]
    action: [action]
    depends_on: [task1]
    inputs:
      data: "${task1.outputs.result}"
    outputs: [outputs]
```

#### Visual DAG

```
[Start]
   │
   ▼
[Task 1]
   │
   ├──────┬──────┐
   ▼      ▼      ▼
[T2]   [T3]   [T4]
   │      │      │
   └──────┴──────┘
          │
          ▼
      [Task 5]
          │
          ▼
       [End]
```

#### Execution Plan

| Order | Task | Dependencies | Agent | Est. Duration |
|-------|------|--------------|-------|---------------|
| 1 | task1 | - | Agent1 | Xs |
| 2 | task2, task3, task4 | task1 | Agents | Xs |
| 3 | task5 | task2,3,4 | Agent5 | Xs |

#### Validation

- [ ] No cycles in DAG
- [ ] All dependencies exist
- [ ] All agents available
- [ ] Inputs properly mapped

---

## Predefined Workflows

### feature-implementation
```yaml
tasks: research → design → implement → test → validate → document
```

### bug-fix
```yaml
tasks: diagnose → reproduce → fix → test → validate
```

### refactor
```yaml
tasks: analyze → plan → refactor → test → review
```

### security-audit
```yaml
tasks: scan → analyze → report → remediate → verify
```

## Validation Checks

### DAG Validity

- [ ] No cycles detected
- [ ] All dependencies resolvable
- [ ] No orphan tasks
- [ ] Start node exists
- [ ] End node(s) reachable

### Execution Validity

- [ ] All agents spawnable
- [ ] Input mappings valid
- [ ] Timeout reasonable
- [ ] Parallel limit respected

## Evidence Generation

Generate Flow execution receipt:

```json
{
  "receipt_id": "flow-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "workflow": {
    "name": "[workflow-name]",
    "version": "1.0",
    "task_count": 8
  },
  "execution": {
    "status": "COMPLETE|FAILED",
    "tasks_completed": 8,
    "tasks_failed": 0,
    "total_duration_ms": 15000,
    "parallel_max": 3
  },
  "task_results": [
    {
      "task_id": "research",
      "status": "COMPLETE",
      "duration_ms": 2300,
      "agent": "MasterReasoner"
    }
  ],
  "outputs": {},
  "integrity_hash": ""
}
```

## Report Format

```
## Flow Execution Report

**Workflow**: [name]
**Status**: COMPLETE | FAILED
**Timestamp**: [ISO timestamp]

### DAG Summary

```
[Visual DAG representation]
```

### Execution Timeline

```
T+0ms     : Workflow started
T+100ms   : Task 'research' started (MasterReasoner)
T+2400ms  : Task 'research' completed
T+2500ms  : Tasks 'design', 'analyze' started (parallel)
T+4300ms  : Task 'design' completed
T+4000ms  : Task 'analyze' completed
...
T+15000ms : Workflow completed
```

### Task Results

| Task | Status | Duration | Agent | Output |
|------|--------|----------|-------|--------|
| research | COMPLETE | 2.3s | MasterReasoner | findings |
| design | COMPLETE | 1.8s | CodeArchitect | architecture |
| ... | ... | ... | ... | ... |

### Workflow Outputs

```json
{
  "final_result": "...",
  "artifacts": [...]
}
```

### Metrics
- Total Duration: Xs
- Parallelization: X tasks max concurrent
- Critical Path: [task1] → [task3] → [task5]

### Receipt
- ID: flow-[timestamp]
- Location: docs/evidence/receipts/
```

---

**Flow Philosophy**: "Complex tasks have structure. Express that structure as a DAG. Execute with maximum parallelism while respecting dependencies."
