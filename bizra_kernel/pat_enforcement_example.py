"""
PAT Enforcement Pipeline — Integration Example
==============================================
Demonstrates integration with existing BIZRA components.

Shows:
- Full pipeline execution
- Integration with SAPE, FATE, Ihsan
- Receipt emission
- Telemetry tracking
- Error handling
"""

import asyncio
import json
from pathlib import Path

from bizra_kernel.pat_enforcement_pipeline import (
    PATEnforcementPipeline,
    PATRequest,
    PATTelemetry,
)


async def example_1_basic_execution():
    """Example 1: Basic pipeline execution."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Pipeline Execution")
    print("=" * 80)

    # Initialize pipeline
    pipeline = PATEnforcementPipeline()

    # Create request
    request = PATRequest(
        session_id="example_session_001",
        task_id="example_task_001",
        query="Optimize BIZRA data lake ingestion pipeline using parallel processing",
        context={"environment": "production", "stakes": "high"},
        synthesis_nodes=[
            {
                "id": "node_1",
                "content": "Implement parallel file processing with worker pools",
                "snr": 0.98,
                "claim_tag": "DERIVED",
                "domains": ["Distributed Systems", "Data Engineering"],
            },
            {
                "id": "node_2",
                "content": "Use Apache Arrow for zero-copy data transfer",
                "snr": 0.97,
                "claim_tag": "IMPLEMENTED",
                "domains": ["Data Engineering", "Performance"],
            },
            {
                "id": "node_3",
                "content": "Apply ML-based anomaly detection for data quality",
                "snr": 0.96,
                "claim_tag": "NOVEL",
                "domains": ["Machine Learning", "Data Engineering"],
            },
        ],
        domains=[
            {
                "name": "Distributed Systems",
                "cluster_id": "cluster_ds",
                "keywords": ["parallel", "concurrent", "distributed"],
            },
            {
                "name": "Data Engineering",
                "cluster_id": "cluster_de",
                "keywords": ["pipeline", "etl", "processing"],
            },
            {
                "name": "Machine Learning",
                "cluster_id": "cluster_ml",
                "keywords": ["anomaly", "detection", "model"],
            },
        ],
        practitioners=[
            {
                "name": "Dr. Alice Chen",
                "tier": "top_1%",
                "domains": ["Distributed Systems", "Data Engineering"],
                "relevance_score": 0.88,
                "publications": 75,
                "h_index": 35,
            },
            {
                "name": "Prof. Bob Martinez",
                "tier": "top_1%",
                "domains": ["Machine Learning", "Data Engineering"],
                "relevance_score": 0.82,
                "publications": 120,
                "h_index": 50,
            },
            {
                "name": "Dr. Carol Singh",
                "tier": "top_1%",
                "domains": ["Distributed Systems", "Performance"],
                "relevance_score": 0.79,
                "publications": 60,
                "h_index": 28,
            },
            {
                "name": "Dr. David Kim",
                "tier": "top_1%",
                "domains": ["Data Engineering"],
                "relevance_score": 0.85,
                "publications": 90,
                "h_index": 42,
            },
        ],
        response_sections=[
            {
                "id": "executive_synthesis",
                "claims": [
                    {"text": "Parallel processing improves throughput 3x", "tag": "MEASURED"},
                    {"text": "Arrow reduces memory overhead 40%", "tag": "IMPLEMENTED"},
                ],
            },
            {
                "id": "domain_cross_pollination_map",
                "domains": ["Distributed Systems", "Data Engineering", "Machine Learning"],
            },
            {
                "id": "elite_practitioner_anchoring",
                "practitioners": 4,
            },
            {
                "id": "novel_insight_synthesis",
                "claims": [
                    {"text": "ML-based anomaly detection during ingestion", "tag": "NOVEL"}
                ],
            },
            {
                "id": "validation_evidence_trail",
                "gate_statuses": [],
                "snr_scores": [],
                "ihsan_scores": [],
                "receipt_ids": [],
            },
            {
                "id": "actionable_recommendations",
                "claims": [
                    {"text": "Deploy parallel workers", "tag": "TARGET"},
                    {"text": "Implement Arrow integration", "tag": "DESIGNED"},
                ],
            },
        ],
        running_snr=0.97,
        novelty_score=0.82,
    )

    # Execute enforcement
    result = await pipeline.enforce(request)

    # Print results
    print(f"\n✓ Enforcement Complete")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  Receipt ID: {result.receipt_id}")
    print(f"  Total Latency: {result.total_latency_ms}ms")
    print(f"  Final SNR: {result.final_snr:.4f}")
    print(f"  Final Novelty: {result.final_novelty:.4f}")
    print(f"  Final Ihsan: {result.final_ihsan:.4f}")

    print(f"\nGate Results:")
    for gate in result.gate_results:
        status_emoji = "✓" if gate.passed else "✗"
        print(
            f"  {status_emoji} {gate.gate_id.value}: {gate.status.value} ({gate.latency_ms}ms)"
        )

    if result.receipt_path:
        print(f"\nReceipt saved: {result.receipt_path}")


async def example_2_failure_handling():
    """Example 2: Handling gate failures."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Failure Handling")
    print("=" * 80)

    pipeline = PATEnforcementPipeline()

    # Create request that will fail Gate 1 (insufficient domains)
    request = PATRequest(
        session_id="example_session_002",
        task_id="example_task_002",
        query="Simple optimization task",
        context={},
        synthesis_nodes=[
            {"id": "node_1", "content": "Use caching", "snr": 0.95, "claim_tag": "DERIVED"}
        ],
        domains=[
            {"name": "Performance", "cluster_id": "cluster_perf"}
        ],  # Only 1 domain, need 3
        practitioners=[],
        response_sections=[],
    )

    # Execute enforcement (will fail)
    result = await pipeline.enforce(request)

    # Handle failure
    print(f"\n✗ Enforcement Failed")
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  Failed at: {result.gate_results[-1].gate_id.value}")

    print(f"\nFailure Evidence:")
    for evidence in result.gate_results[-1].evidence:
        print(f"  - {evidence}")

    print(f"\nCorrection Attempts: {result.correction_attempts}")


async def example_3_telemetry_tracking():
    """Example 3: Telemetry and monitoring."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Telemetry Tracking")
    print("=" * 80)

    pipeline = PATEnforcementPipeline()
    telemetry = PATTelemetry()

    # Create multiple requests
    requests = []
    for i in range(5):
        request = PATRequest(
            session_id=f"example_session_{i:03d}",
            task_id=f"example_task_{i:03d}",
            query=f"Task {i}",
            context={},
            synthesis_nodes=[
                {
                    "id": "node_1",
                    "content": "Content",
                    "snr": 0.98 - (i * 0.01),
                    "claim_tag": "DERIVED",
                }
            ],
            domains=[
                {"name": f"Domain{j}", "cluster_id": f"cluster_{j}"} for j in range(3)
            ],
            practitioners=[
                {"name": f"Expert{j}", "tier": "top_1%", "domains": [f"Domain{j}"], "relevance_score": 0.75}
                for j in range(3)
            ],
            response_sections=[
                {"id": section_id, "claims": []}
                for section_id in [
                    "executive_synthesis",
                    "domain_cross_pollination_map",
                    "elite_practitioner_anchoring",
                    "novel_insight_synthesis",
                    "validation_evidence_trail",
                    "actionable_recommendations",
                ]
            ],
        )
        requests.append(request)

    # Execute all requests
    print(f"\nExecuting {len(requests)} enforcements...")
    for i, request in enumerate(requests):
        result = await pipeline.enforce(request)
        telemetry.record_enforcement(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{i+1}/{len(requests)}] {status} - {result.total_latency_ms}ms")

    # Print telemetry stats
    stats = telemetry.get_stats()
    print(f"\nTelemetry Summary:")
    print(f"  Total Enforcements: {stats['total_enforcements']}")
    print(f"  Passes: {stats['total_passes']}")
    print(f"  Failures: {stats['total_failures']}")
    print(f"  Pass Rate: {stats['pass_rate']:.1%}")
    print(f"  Average Latency: {stats['average_latency_ms']:.0f}ms")

    if stats["gate_failure_counts"]:
        print(f"\nGate Failure Distribution:")
        for gate, count in stats["gate_failure_counts"].items():
            print(f"  {gate}: {count}")


async def example_4_component_integration():
    """Example 4: Integration with existing components."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Component Integration")
    print("=" * 80)

    # Import supporting components
    from bizra_kernel.pat_domain_validator import PATDomainValidator
    from bizra_kernel.pat_novelty_probe import PATNoveltyProbe
    from bizra_kernel.pat_citation_validator import PATCitationValidator

    # 1. Domain validation
    print("\n1. Domain Validation")
    domain_validator = PATDomainValidator()

    domains = [
        {"name": "Distributed Systems", "cluster_id": "ds", "keywords": ["parallel"]},
        {"name": "Machine Learning", "cluster_id": "ml", "keywords": ["neural"]},
        {"name": "Data Engineering", "cluster_id": "de", "keywords": ["pipeline"]},
    ]

    domain_result = await domain_validator.validate(domains)
    print(f"   Domain Count: {domain_result.domain_count}")
    print(f"   Unrelatedness: {domain_result.unrelatedness_score:.4f}")
    print(f"   Passed: {domain_result.passed}")

    # 2. Novelty probing
    print("\n2. Novelty Probing")
    novelty_probe = PATNoveltyProbe()

    insight = "Use quantum-inspired optimization for distributed consensus"
    novelty_result = await novelty_probe.probe(insight, domain="Distributed Systems")
    print(f"   Insight: {insight[:50]}...")
    print(f"   Novelty Score: {novelty_result.novelty_score:.4f}")
    print(f"   Passed: {novelty_result.passed}")

    # 3. Citation validation
    print("\n3. Citation Validation")
    citation_validator = PATCitationValidator()

    practitioners = [
        {
            "name": "Dr. Expert",
            "tier": "top_1%",
            "domains": ["Distributed Systems"],
            "relevance_score": 0.80,
        }
    ]

    citation_result = await citation_validator.validate(practitioners, domains)
    print(f"   Practitioners: {len(citation_result.practitioners)}")
    print(f"   Per Domain: {citation_result.practitioners_per_domain}")
    print(f"   Passed: {citation_result.passed}")


async def example_5_receipt_analysis():
    """Example 5: Receipt analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Receipt Analysis")
    print("=" * 80)

    pipeline = PATEnforcementPipeline()

    # Execute enforcement
    request = PATRequest(
        session_id="example_session_receipt",
        task_id="example_task_receipt",
        query="Test query",
        context={},
        synthesis_nodes=[
            {"id": "node_1", "content": "Test", "snr": 0.98, "claim_tag": "DERIVED"}
        ],
        domains=[{"name": f"Domain{i}", "cluster_id": f"c{i}"} for i in range(3)],
        practitioners=[
            {"name": f"Expert{i}", "tier": "top_1%", "domains": [f"Domain{i}"], "relevance_score": 0.75}
            for i in range(3)
        ],
        response_sections=[
            {"id": sid, "claims": []}
            for sid in [
                "executive_synthesis",
                "domain_cross_pollination_map",
                "elite_practitioner_anchoring",
                "novel_insight_synthesis",
                "validation_evidence_trail",
                "actionable_recommendations",
            ]
        ],
    )

    result = await pipeline.enforce(request)

    # Read receipt
    receipt_path = Path(result.receipt_path)
    if receipt_path.exists():
        with open(receipt_path, "r") as f:
            receipt_data = json.load(f)

        print(f"\nReceipt Analysis:")
        print(f"  Receipt ID: {receipt_data['receipt_id']}")
        print(f"  Timestamp: {receipt_data['timestamp']}")
        print(f"  Session ID: {receipt_data['session_id']}")
        print(f"  Task ID: {receipt_data['task_id']}")
        print(f"  Passed: {receipt_data['passed']}")

        print(f"\n  Final Scores:")
        print(f"    SNR: {receipt_data['final_snr']:.4f}")
        print(f"    Novelty: {receipt_data['final_novelty']:.4f}")
        print(f"    Ihsan: {receipt_data['final_ihsan']:.4f}")

        print(f"\n  Gate Breakdown:")
        for gate in receipt_data["gate_results"]:
            print(f"    {gate['gate_id']}: {gate['status']} ({gate['latency_ms']}ms)")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("PAT ENFORCEMENT PIPELINE — INTEGRATION EXAMPLES")
    print("=" * 80)

    await example_1_basic_execution()
    await example_2_failure_handling()
    await example_3_telemetry_tracking()
    await example_4_component_integration()
    await example_5_receipt_analysis()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
