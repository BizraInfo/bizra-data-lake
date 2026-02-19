# Phase 45.3 — Compute Pool Layer

> **Version:** 0.1.0 | **Status:** Specification + Pseudocode
> **Standing on Giants:** SETI@home (Anderson, 2002) · BOINC (distributed computing) · Kubernetes (container orchestration) · Harberger (self-assessed taxation, 1962)

## 3.1 Purpose

Enable voluntary compute sharing across the BIZRA mesh. Every node
contributes what it can — from a Raspberry Pi to an RTX 4090 —
and receives fair compensation through the SEED token economy.

This is not mandatory. Nodes can participate at cognition-only level.
But nodes that share compute earn SEED, increase reputation, and get
priority access to network resources when they need them.

## 3.2 Design Principles

```
PRINCIPLES:
  voluntary:        "No node is forced to share compute"
  privacy:          "Tasks execute in sandboxed containers"
  fair_exchange:    "SEED tokens per compute-hour, Harberger-taxed"
  anti_plutocracy:  "ADL Gini gate prevents compute hoarding"
  local_first:      "Node always serves its own human first"
  graceful_exit:    "Any node can leave without breaking the mesh"
```

## 3.3 Resource Advertiser — Pseudocode

```
MODULE: core.compute_pool.advertiser

IMPORT: ComputeProfile from core.node_identity.compute_profile
IMPORT: SEED_COMPUTE_HOUR_PEG from core.integration.constants

CLASS ResourceAdvertiser:
  """
  Broadcasts what compute this node offers to the mesh.

  Like a market stall: "Here's what I have, here's my price."
  Prices are bounded by SEED_COMPUTE_HOUR_PEG (1 SEED = 1 hour).
  Harberger tax applies: if you overvalue, you pay more tax.
  If you undervalue, others can claim your resources.
  """

  FIELDS:
    profile: ComputeProfile
    available_fraction: float       -- 0.0 to 1.0 (how much to share)
    price_per_hour: float           -- in SEED tokens
    current_load: float             -- 0.0 to 1.0
    reservation_queue: list[Reservation]

  CONSTRUCTOR(profile: ComputeProfile, share_fraction: float = 0.25):
    self.profile = profile
    self.available_fraction = share_fraction  -- default: share 25%
    self.price_per_hour = SEED_COMPUTE_HOUR_PEG  -- 1 SEED = 1 hour
    self.current_load = 0.0
    self.reservation_queue = []

  METHOD advertise() -> ResourceAdvertisement:
    """Generate current resource advertisement for gossip."""
    available = self.compute_available()
    RETURN ResourceAdvertisement(
      node_id = self_identity.node_id,
      cpu_cores_available = int(self.profile.cpu_cores * self.available_fraction),
      gpu_vram_available_mb = int(self.profile.gpu_vram_mb * self.available_fraction),
      ram_available_mb = int(self.profile.ram_total_mb * self.available_fraction),
      current_load = self.current_load,
      price_per_hour = self.price_per_hour,
      compute_score = self.profile.compute_power_score() * self.available_fraction,
      available_until = self._next_unavailable_time(),
    )

  METHOD compute_available() -> bool:
    """Can this node accept new tasks right now?"""
    RETURN (
      self.current_load < 0.90  -- reserve 10% headroom
      AND self.available_fraction > 0.0
      AND self.profile.is_available_now()
    )

  METHOD reserve(task: TaskDescriptor) -> Reservation | None:
    """Try to reserve compute for a task. Returns None if no capacity."""
    IF NOT self.compute_available():
      RETURN None

    estimated_load = estimate_load(task)
    IF self.current_load + estimated_load > 0.95:
      RETURN None

    reservation = Reservation(
      task_id = task.task_id,
      start = utc_now(),
      estimated_duration = task.timeout_seconds,
      estimated_load = estimated_load,
    )
    self.reservation_queue.append(reservation)
    self.current_load += estimated_load
    RETURN reservation

  METHOD release(task_id: str) -> None:
    """Release compute after task completion."""
    FOR i, res IN enumerate(self.reservation_queue):
      IF res.task_id == task_id:
        self.current_load -= res.estimated_load
        self.reservation_queue.pop(i)
        RETURN
```

## 3.4 Task Sharding — Pseudocode

```
MODULE: core.compute_pool.sharder

CLASS TaskSharder:
  """
  Splits large tasks into smaller shards that can be distributed.

  Standing on Giants:
    MapReduce (Dean & Ghemawat, 2004) — parallel data processing
    BSP (Valiant, 1990) — bulk synchronous parallel model
  """

  METHOD shard(task: TaskDescriptor, available_nodes: list[ResourceAdvertisement])
      -> list[TaskShard]:
    """Split task into shards matched to available compute."""

    IF task.task_type == "reasoning":
      -- Reasoning tasks: split into parallel hypotheses (GoT-style)
      RETURN self._shard_reasoning(task, available_nodes)

    ELIF task.task_type == "computation":
      -- Data tasks: split by data partition
      RETURN self._shard_data(task, available_nodes)

    ELIF task.task_type == "validation":
      -- Validation: same task to multiple nodes (redundant)
      RETURN self._shard_redundant(task, available_nodes)

  METHOD _shard_reasoning(task, nodes) -> list[TaskShard]:
    """
    Graph-of-Thoughts style: each node explores a different hypothesis.
    Results are merged via weighted consensus.
    """
    hypotheses = generate_hypotheses(task.description, k=len(nodes))
    shards = []
    FOR i, (hypothesis, node) IN enumerate(zip(hypotheses, nodes)):
      shards.append(TaskShard(
        shard_id = f"{task.task_id}-h{i}",
        parent_task_id = task.task_id,
        assigned_node = node.node_id,
        shard_type = "hypothesis",
        payload = {"hypothesis": hypothesis, "context": task.description},
        merge_strategy = "weighted_consensus",
      ))
    RETURN shards

  METHOD _shard_data(task, nodes) -> list[TaskShard]:
    """Split data processing across nodes by partition."""
    data = task.payload["data"]
    partitions = split_into_partitions(data, len(nodes))
    shards = []
    FOR i, (partition, node) IN enumerate(zip(partitions, nodes)):
      shards.append(TaskShard(
        shard_id = f"{task.task_id}-d{i}",
        parent_task_id = task.task_id,
        assigned_node = node.node_id,
        shard_type = "data_partition",
        payload = {"partition": partition},
        merge_strategy = "concatenate",
      ))
    RETURN shards

  METHOD _shard_redundant(task, nodes) -> list[TaskShard]:
    """Same task to N nodes. Majority vote on result."""
    shards = []
    FOR i, node IN enumerate(nodes):
      shards.append(TaskShard(
        shard_id = f"{task.task_id}-v{i}",
        parent_task_id = task.task_id,
        assigned_node = node.node_id,
        shard_type = "validation",
        payload = task.payload,
        merge_strategy = "majority_vote",
      ))
    RETURN shards

DATACLASS TaskShard:
  shard_id: str
  parent_task_id: str
  assigned_node: str
  shard_type: str
  payload: dict
  merge_strategy: str
  status: "pending" | "running" | "completed" | "failed" = "pending"
  result: dict | None = None
```

## 3.5 Distributed Inference — Pseudocode

```
MODULE: core.compute_pool.distributed_inference

CLASS DistributedInference:
  """
  Coordinate LLM inference across multiple nodes.

  Use cases:
  1. Parallel hypothesis generation (different models/prompts)
  2. Ensemble: same prompt to multiple models, merge outputs
  3. Pipeline: Node A embeds, Node B reasons, Node C validates

  Standing on Giants:
    MoE (Shazeer et al., 2017) — Mixture of Experts
    Speculative decoding (Leviathan et al., 2023)
  """

  METHOD parallel_hypotheses(prompt: str, nodes: list[NodePublicCard],
                              k: int = 3) -> list[InferenceResult]:
    """
    Send the same prompt to K nodes for diverse hypotheses.
    Each node uses its local LLM (may differ in model/size).
    """
    tasks = []
    FOR node IN nodes[:k]:
      task = TaskDescriptor(
        task_id = generate_task_id(),
        task_type = "reasoning",
        description = prompt,
        required_domains = [],
        min_compute_score = 0.1,
        min_reputation = 0.2,
        max_nodes = 1,
        timeout_seconds = 120,
        reward_seed = 0.5,
        privacy_level = ShareLevel.TASK_RESULTS,
      )
      tasks.append((node, task))

    -- Send all in parallel
    results = parallel_send_and_collect(tasks)

    -- Filter: only keep results above Ihsan floor
    valid = [r for r in results if r.snr_score >= UNIFIED_IHSAN_THRESHOLD]

    RETURN valid

  METHOD ensemble_merge(results: list[InferenceResult]) -> dict:
    """
    Merge results from multiple nodes into a single answer.

    Strategy:
    1. Weight by SNR score * node reputation
    2. Extract agreement points (high confidence)
    3. Flag disagreement points (needs human arbitration)
    """
    IF len(results) == 0:
      RAISE NoValidResultsError()

    IF len(results) == 1:
      RETURN results[0].output

    -- Weight each result
    weighted = []
    FOR r IN results:
      weight = r.snr_score * r.node_reputation
      weighted.append((r.output, weight))

    -- Find consensus via semantic similarity
    agreement = find_semantic_agreement(weighted)
    disagreements = find_disagreements(weighted)

    RETURN {
      "consensus": agreement,
      "confidence": mean([w for _, w in weighted]),
      "disagreements": disagreements,
      "node_count": len(results),
      "merge_strategy": "weighted_ensemble",
    }
```

## 3.6 Compute Credit Accounting — Pseudocode

```
MODULE: core.compute_pool.credit

IMPORT: SEED_COMPUTE_HOUR_PEG, ADL_HARBERGER_TAX_RATE from core.integration.constants
IMPORT: TokenLedger from core.token.ledger

CLASS ComputeCredit:
  """
  Track compute contributions and rewards.

  Every compute contribution generates a PCI receipt.
  Receipts are the basis for SEED token minting.
  Harberger tax applies to declared resource value.

  Standing on Giants:
    Harberger (1962) — self-assessed taxation
    RIBA_ZERO invariant — no exploitation in exchange
  """

  FIELDS:
    ledger: TokenLedger
    contributions: dict[str, list[ComputeContribution]]  -- node_id -> history

  METHOD record_contribution(node_id: str, task_id: str,
                              compute_seconds: float, receipt: PCIReceipt):
    """Record verified compute contribution and mint SEED reward."""
    -- Verify the receipt is valid (ZANN: no unverified claims)
    IF NOT receipt.is_valid():
      RAISE InvalidReceiptError("Cannot credit unverified compute")

    -- Calculate SEED reward
    compute_hours = compute_seconds / 3600.0
    seed_reward = compute_hours * SEED_COMPUTE_HOUR_PEG

    -- Mint and transfer
    self.ledger.mint_seed(node_id, seed_reward, receipt_id=receipt.id)

    -- Track contribution
    contribution = ComputeContribution(
      node_id = node_id,
      task_id = task_id,
      compute_seconds = compute_seconds,
      seed_earned = seed_reward,
      receipt_digest = receipt.digest(),
      timestamp = utc_now(),
    )
    self.contributions.setdefault(node_id, []).append(contribution)

  METHOD apply_harberger_tax(node_id: str, declared_value: float) -> float:
    """
    Apply Harberger tax on declared resource value.

    If you overvalue your resources: you pay more tax.
    If you undervalue: others can claim your compute slots.
    Tax flows to Universal Basic Compute (UBC) pool.
    """
    daily_rate = ADL_HARBERGER_TAX_RATE / 365.0
    tax = declared_value * daily_rate
    self.ledger.transfer(node_id, "UBC_POOL", tax, reason="harberger_tax")
    RETURN tax

  METHOD total_contributed(node_id: str) -> float:
    """Total compute hours contributed by a node."""
    contribs = self.contributions.get(node_id, [])
    RETURN sum(c.compute_seconds / 3600.0 for c in contribs)

DATACLASS ComputeContribution:
  node_id: str
  task_id: str
  compute_seconds: float
  seed_earned: float
  receipt_digest: str
  timestamp: datetime
```

## 3.7 Sandbox Execution — Pseudocode

```
MODULE: core.compute_pool.sandbox

CLASS TaskSandbox:
  """
  Isolated execution environment for mesh tasks.

  Standing on Giants: Agent Zero (Docker isolation)

  Tasks from other nodes execute in sandboxed containers.
  No access to host filesystem, network restricted,
  resource limits enforced.
  """

  METHOD execute(shard: TaskShard) -> TaskResult:
    """Run a task shard in isolation."""

    -- Create sandbox (priority: WASM > microVM > Docker)
    sandbox = create_sandbox(
      memory_limit_mb = shard.resource_limit.ram_mb,
      cpu_limit = shard.resource_limit.cpu_fraction,
      timeout_seconds = shard.timeout,
      network = "none",  -- no network access by default
    )

    TRY:
      -- Execute inside sandbox
      start = monotonic()
      output = sandbox.run(shard.payload)
      elapsed = monotonic() - start

      -- Create PCI receipt for the work
      receipt = create_receipt(
        task_id = shard.shard_id,
        input_digest = blake3_digest(canonical_bytes(shard.payload)),
        output_digest = blake3_digest(canonical_bytes(output)),
        compute_seconds = elapsed,
      )

      RETURN TaskResult(
        task_id = shard.shard_id,
        node_id = self_identity.node_id,
        result = output,
        snr_score = assess_quality(output),
        compute_time_seconds = elapsed,
        receipt_digest = receipt.digest(),
      )

    FINALLY:
      sandbox.destroy()  -- always cleanup
```

## 3.8 TDD Anchors

```
TEST_SUITE: tests/core/compute_pool/

  test_advertiser:
    - default share fraction is 0.25
    - compute_available() false when load > 0.90
    - reserve() decreases available capacity
    - release() restores capacity
    - price bounded by SEED_COMPUTE_HOUR_PEG

  test_sharder:
    - reasoning task produces hypothesis shards
    - data task produces partition shards
    - validation task produces redundant shards
    - shard count matches available nodes

  test_distributed_inference:
    - parallel_hypotheses sends to K nodes
    - results below Ihsan floor filtered
    - ensemble_merge produces weighted consensus
    - disagreements are flagged

  test_credit:
    - valid receipt earns SEED tokens
    - invalid receipt rejected (ZANN)
    - Harberger tax calculated correctly
    - tax flows to UBC pool
    - total_contributed accumulates

  test_sandbox:
    - task executes in isolation
    - resource limits enforced
    - PCI receipt generated for work
    - sandbox destroyed after execution
```
