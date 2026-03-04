# Phase 64.04 — Floor Constraint (Universality Gate)

## Purpose

Enforce that the constitutional pipeline runs on minimum viable hardware.
The 29.4s heartbeat on phi3:mini/CPU is not a baseline to improve — it
IS the specification. If the pipeline doesn't work on a $200 phone,
BIZRA's 8-billion-node thesis is falsified.

Standing on Giants: Shannon (channel capacity floor) · Al-Ghazali (Daughter Test — deploy for her)

## Key Insight

```
In a system designed for universal access,
the floor is more important than the ceiling.

The ceiling rises automatically with network growth (reverse scaling).
The floor must be PROVEN to exist.

A system that requires a 4090 addresses ~50M users.
A system that runs on a $200 phone addresses 8B users.
The floor is a 160x market expansion.
```

## Data Model

```pseudocode
@dataclass(frozen=True)
class FloorProfile:
    """Minimum viable hardware profile for a BIZRA node.

    This is the grandmother's $200 phone in Cairo.
    This is the student's 5-year-old laptop in Jakarta.
    If the pipeline doesn't work HERE, it doesn't work.
    """
    min_ram_gb: float = 2.0          # 2 GB RAM minimum
    min_storage_gb: float = 4.0      # 4 GB free storage minimum
    min_cpu_cores: int = 2           # 2 cores minimum
    min_cpu_mhz: int = 1000          # 1 GHz minimum
    gpu_required: bool = False       # GPU is NEVER required
    network_required: bool = False   # offline-first always
    min_model_params_b: float = 0.5  # smallest viable model (500M params)
    max_pipeline_time_s: float = 60.0  # constitutional pipeline must complete in 60s
    max_memory_usage_mb: float = 512   # pipeline peak memory under 512MB


@dataclass
class FloorCheckResult:
    """Result of checking a node against the floor constraint."""
    compliant: bool
    node_body: "NodeBody"
    floor_profile: FloorProfile
    violations: List[str]            # which constraints are violated
    margin: Dict[str, float]         # how much above/below each threshold
    pipeline_time_s: Optional[float] # measured pipeline execution time
    pipeline_memory_mb: Optional[float]  # measured peak memory
    timestamp: str


@dataclass(frozen=True)
class FloorBenchmark:
    """Recorded benchmark of the constitutional pipeline on this hardware."""
    benchmark_id: str
    hardware_summary: str            # "2-core ARM, 2GB RAM, no GPU"
    pipeline_time_s: float           # actual execution time
    pipeline_memory_mb: float        # actual peak memory
    model_used: str                  # e.g., "phi3:mini", "qwen2.5-0.5b"
    ihsan_score: float               # quality of output
    constitutional_gates_passed: bool
    evidence_hash: str
    benchmarked_at: str
```

## Core Class

```pseudocode
class FloorConstraint:
    """Enforces minimum viable node universality.

    The floor is the supreme design constraint. Everything else
    (4090 acceleration, reverse scaling, URP contribution) is built
    ON TOP of this floor. Without the floor, there is no building.

    البذرة — every seed has infinite potential. The seed is the
    minimum viable node. If the seed can't germinate on poor soil,
    the forest never grows.
    """

    def __init__(
        self,
        floor_profile: Optional[FloorProfile] = None,
        evidence_ledger: Optional["EvidenceLedger"] = None,
    ):
        self._floor = floor_profile or FloorProfile()
        self._evidence = evidence_ledger
        self._benchmarks: List[FloorBenchmark] = []

    # ── Constraint checking ────────────────────────────────────

    def check(self, body: "NodeBody") -> FloorCheckResult:
        """Check if a node meets the floor constraint.

        PSEUDOCODE:
        1. violations = []
        2. margin = {}
        3. IF body.total_capacity["ram"] < floor.min_ram_gb:
            violations.append("RAM below minimum")
            margin["ram"] = body.ram - floor.min_ram_gb (negative = violation)
        4. IF body.total_capacity["disk"] < floor.min_storage_gb:
            violations.append("Storage below minimum")
        5. IF body.total_capacity["cpu"] < floor.min_cpu_cores:
            violations.append("CPU cores below minimum")
        6. NOTE: GPU is NEVER checked — GPU is NEVER required
        7. NOTE: Network is NEVER checked — offline-first always
        8. compliant = len(violations) == 0
        9. Return FloorCheckResult(compliant, body, floor, violations, margin, ...)
        """
        ...

    def check_pipeline_time(
        self,
        measured_time_s: float,
        measured_memory_mb: float,
    ) -> bool:
        """Check if the constitutional pipeline meets time/memory floor.

        PSEUDOCODE:
        - IF measured_time_s > floor.max_pipeline_time_s:
            return False  # pipeline too slow for minimum viable node
        - IF measured_memory_mb > floor.max_memory_usage_mb:
            return False  # pipeline uses too much memory
        - Return True
        """
        ...

    # ── Benchmarking ───────────────────────────────────────────

    async def benchmark_pipeline(
        self,
        mission_text: str = "What is the capital of Egypt?",
    ) -> FloorBenchmark:
        """Run the constitutional pipeline and measure floor compliance.

        PSEUDOCODE:
        1. Record start time and memory
        2. Execute minimal mission pipeline:
            a. Load smallest available model
            b. Run inference
            c. Run Ihsan gate on output
            d. Generate evidence receipt
        3. Record end time and peak memory
        4. pipeline_time = end - start
        5. pipeline_memory = peak - baseline
        6. ihsan_score = gate result
        7. Create FloorBenchmark record
        8. Append to evidence ledger
        9. Return benchmark
        """
        ...

    # ── Floor simulation ───────────────────────────────────────

    def simulate_floor_node(self) -> FloorProfile:
        """Return the floor profile for testing.

        This profile represents the weakest node BIZRA must support:
        - 2 CPU cores at 1 GHz
        - 2 GB RAM
        - 4 GB storage
        - No GPU
        - No network
        - Smallest model (500M params)
        - 60s pipeline time budget
        - 512 MB memory budget

        Tests MUST pass under this profile. If they don't,
        the 8-billion-node thesis is falsified.
        """
        return FloorProfile()

    # ── Dashboard ──────────────────────────────────────────────

    def floor_report(self, body: "NodeBody") -> Dict[str, Any]:
        """Human-readable floor compliance report.

        Returns:
        {
            "compliant": true,
            "floor": "2-core, 2GB RAM, 4GB disk, no GPU",
            "this_node": "32-core, 62GB RAM, 1TB disk, RTX 4090",
            "headroom": {
                "cpu": "16x above floor",
                "ram": "31x above floor",
                "disk": "250x above floor",
                "gpu": "bonus (not required)"
            },
            "surplus_for_urp": {
                "cpu": "30 cores available",
                "ram": "60 GB available",
                "gpu": "16 GB VRAM (100% surplus)"
            },
            "pipeline_benchmark": {
                "floor_time": "60.0s",
                "this_node_time": "2.1s",
                "speedup": "28.6x",
            },
            "daughter_test": "PASSED — this pipeline works for her"
        }
        """
        ...
```

## Architecture Decision: GPU is NEVER Required

This is a hard constraint, not a preference. BIZRA nodes that happen
to have GPUs contribute them to the URP. But the constitutional
pipeline MUST run without a GPU. The moment GPU becomes required,
BIZRA excludes billions of potential nodes.

```pseudocode
# This is CORRECT:
if gpu_available:
    accelerate_inference(gpu)        # bonus: faster for this node
    contribute_surplus_to_urp(gpu)   # إيثار: help others
else:
    run_inference_on_cpu()           # works. slower, but works.
    # This node is STILL a full sovereign node.
    # It can still earn SEED through CPU/RAM/storage contributions.
    # The floor includes it. The ceiling doesn't exclude it.

# This is WRONG and MUST NEVER be written:
if not gpu_available:
    raise RuntimeError("GPU required")  # THIS KILLS THE THESIS
```

## Daughter Test Integration

The floor constraint IS the Daughter Test applied to hardware:

```pseudocode
def daughter_test(body: NodeBody) -> bool:
    """Would you deploy this for your daughter?

    She has:
    - A mid-range phone from 3 years ago
    - Unreliable mobile data
    - No GPU, no cloud subscription
    - But she IS a node. She IS sovereign.

    If the pipeline doesn't work for her,
    the ceiling is irrelevant.
    """
    floor = FloorProfile()  # minimum viable node
    result = FloorConstraint(floor).check(body)
    return result.compliant
```

## TDD Anchors

```
TEST: FloorProfile defaults are sane (2GB RAM, 2 cores, no GPU)
TEST: check passes for node that meets all floor requirements
TEST: check fails for node below RAM minimum
TEST: check fails for node below CPU minimum
TEST: check NEVER fails for missing GPU (GPU is never required)
TEST: check NEVER fails for missing network (offline-first)
TEST: check_pipeline_time passes at 59s (under 60s limit)
TEST: check_pipeline_time fails at 61s (over 60s limit)
TEST: floor_report includes headroom calculations
TEST: floor_report includes surplus_for_urp
TEST: simulate_floor_node returns most constrained profile
TEST: daughter_test passes for minimum viable hardware
TEST: daughter_test passes for high-end hardware
TEST: benchmark_pipeline records evidence receipt
```
