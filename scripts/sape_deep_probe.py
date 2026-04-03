#!/usr/bin/env python3
"""
BIZRA SAPE Deep Circuit Probe & Multi-Lens Analysis
====================================================
DNA Signature: 7-3-6-9-00

This script probes:
1. Rarely-fired tension detection circuits
2. Symbolic-neural bridge coherence
3. Abstraction elevation chains
4. SNR optimization opportunities
5. Ihsān alignment verification
6. Graph-of-Thoughts reasoning paths
"""

import sys
import json
from dataclasses import asdict
from typing import Dict, List, Any

# Add bizra_kernel to path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bizra_kernel import (
    TensionStudio, TensionType,
    SymbolicHarness, SymbolType,
    AbstractionElevator, AbstractionLevel, DomainType, Instance,
    SAPEEngine,
    SNRTracker,
    MultiStageVerifier,
    LexiconLedger, DNA_SIGNATURE,
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def probe_tension_circuits():
    """Probe the Tension Studio for rarely-fired contradiction patterns."""
    print_header("SAPE MODULE 7: TENSION STUDIO - Deep Circuit Probe")
    
    studio = TensionStudio()
    
    # Edge cases designed to fire rarely-used circuits
    test_cases = [
        # Logical contradictions
        ("The system must always allow access but never permit entry.", "LOGICAL"),
        ("Enable all features. Disable all features for security.", "LOGICAL"),
        # Temporal inconsistencies
        ("First complete step 3, then start step 1 after the final step.", "TEMPORAL"),
        ("Initialize before startup, finalize after the initial phase.", "TEMPORAL"),
        # Value trade-offs (Ihsān tensions)
        ("Maximize efficiency by removing safety checks.", "VALUE"),
        ("Improve user benefit by hiding audit logs.", "VALUE"),
        # Scope ambiguity
        ("All agents must approve. No agents should have veto power.", "SCOPE"),
        # Priority conflicts
        ("Safety is the highest priority. Performance must never be sacrificed.", "PRIORITY"),
    ]
    
    results = {"detected": 0, "resolved": 0, "by_type": {}}
    
    for text, expected_type in test_cases:
        tensions = studio.analyze_text(text)
        print(f"\n  Input: \"{text[:60]}...\"")
        print(f"  Expected: {expected_type}")
        
        if tensions:
            for t in tensions:
                results["detected"] += 1
                t_type = t.tension_type.value
                results["by_type"][t_type] = results["by_type"].get(t_type, 0) + 1
                print(f"    ✓ Detected: {t_type} (severity: {t.severity:.2f})")
                
                # Try resolution
                resolution = studio.resolve_tension(t)
                if resolution:
                    results["resolved"] += 1
                    print(f"    → Resolved via: {t.resolution_strategy}")
        else:
            print(f"    ✗ No tension detected (circuit gap)")
    
    print(f"\n  Summary: {results['detected']} tensions detected, {results['resolved']} resolved")
    print(f"  By type: {results['by_type']}")
    return results


def probe_symbolic_neural_bridge():
    """Probe the Symbolic Harness for grounding coherence."""
    print_header("SAPE MODULE 5: SYMBOLIC HARNESS - Neural-Symbolic Bridge")
    
    harness = SymbolicHarness()
    
    # Check pre-registered Ihsān symbols
    print("\n  Pre-registered Ihsān Dimensions:")
    ihsan_symbols = [s for s in harness.symbol_registry.values() 
                     if s.symbol_type == SymbolType.DIMENSION]
    
    total_weight = 0.0
    for sym in sorted(ihsan_symbols, key=lambda s: s.numeric_value or 0, reverse=True):
        weight = sym.numeric_value or 0
        total_weight += weight
        print(f"    {sym.name}: weight={weight:.2f} grounded={sym.grounded}")
    
    print(f"\n  Weight sum: {total_weight:.2f} (should be 1.0)")
    print(f"  Invariant check: {'✓ PASS' if abs(total_weight - 1.0) < 0.01 else '✗ FAIL'}")
    
    # Register and ground new concepts
    print("\n  Testing concept grounding:")
    test_concepts = [
        ("byzantine_tolerance", "Ability to tolerate f Byzantine faults in n=3f+1 validators"),
        ("genesis_immutability", "Property that Genesis Block 0 cannot be modified"),
        ("fail_closed", "Principle of refusing action under uncertainty (FATE)"),
    ]
    
    for name, definition in test_concepts:
        symbol = harness.register_concept(name, definition)
        result = harness.ground_symbol(symbol.symbol_id)
        if result:
            print(f"    {name}: confidence={result.confidence:.2f} method={result.method}")
        else:
            print(f"    {name}: grounding failed (needs ontology binding)")
    
    return {"symbols": len(harness.symbol_registry), "ihsan_dims": len(ihsan_symbols)}


def probe_abstraction_elevation():
    """Probe the Abstraction Elevator for instance→pattern→principle chains."""
    print_header("SAPE MODULE 6: ABSTRACTION ELEVATOR - Pattern Generalization")
    
    elevator = AbstractionElevator()
    
    # Check pre-registered axioms
    print("\n  Genesis Axioms (Level 3 - Universal Truths):")
    for pid, prin in elevator.principles.items():
        if prin.abstraction_level == AbstractionLevel.AXIOM:
            print(f"    {prin.name}:")
            print(f"      Statement: {prin.statement[:60]}...")
            print(f"      Ihsān alignment: {prin.ihsan_alignment:.2f}")
    
    # Test elevation chain
    print("\n  Testing Instance → Pattern → Principle elevation:")
    
    # Add related instances using record_instance method
    instances = []
    test_data = [
        ("SAT validator rejected request due to SQL injection pattern",
         ["security", "validation", "rejection", "pattern_match"],
         "Request blocked, attack prevented"),
        ("SAT validator rejected request due to XSS pattern",
         ["security", "validation", "rejection", "pattern_match"],
         "Request blocked, attack prevented"),
        ("SAT validator rejected request due to shell injection",
         ["security", "validation", "rejection", "pattern_match"],
         "Request blocked, attack prevented"),
    ]
    
    for desc, features, outcome in test_data:
        inst = elevator.record_instance(
            domain=DomainType.TECHNICAL,
            description=desc,
            key_features=features,
            outcome=outcome,
        )
        instances.append(inst)
        print(f"    Added instance: {inst.instance_id}")
    
    # Patterns are auto-detected in record_instance
    print(f"\n  Patterns detected: {len(elevator.patterns)}")
    for pid, pattern in elevator.patterns.items():
        print(f"    {pattern.pattern_id}: {pattern.name}")
        print(f"      Features: {pattern.key_features}")
        print(f"      Confidence: {pattern.confidence:.2f}")
    
    # Check for principle elevation
    print(f"\n  Principles in registry: {len(elevator.principles)}")
    
    return {"instances": len(elevator.instances), "patterns": len(elevator.patterns), "principles": len(elevator.principles)}


def probe_sape_engine():
    """Probe the SAPE Engine for pattern optimization."""
    print_header("SAPE ENGINE: Pattern Elevation & Optimization")
    
    engine = SAPEEngine()
    
    print("\n  Pre-elevated patterns (Blueprint):")
    total_snr_gain = 0.0
    total_latency_reduction = 0
    total_token_savings = 0.0
    
    for pid, pattern in engine.elevated_patterns.items():
        total_snr_gain += pattern.snr_improvement
        total_latency_reduction += pattern.latency_reduction_ms
        total_token_savings += pattern.token_savings_percent
        
        print(f"\n    {pattern.pattern_name}:")
        print(f"      Trigger: {pattern.trigger_sequence}")
        print(f"      SNR: +{pattern.snr_improvement:.0%}")
        print(f"      Latency: -{pattern.latency_reduction_ms}ms")
        print(f"      Tokens: -{pattern.token_savings_percent:.0f}%")
    
    print(f"\n  Aggregate optimization potential:")
    print(f"    Total SNR improvement: +{total_snr_gain:.0%}")
    print(f"    Total latency reduction: -{total_latency_reduction}ms")
    print(f"    Avg token savings: -{total_token_savings/len(engine.elevated_patterns):.0f}%")
    
    # Test sequence observation
    print("\n  Testing sequence elevation:")
    test_sequences = [
        ["threat_scan", "compliance_check", "bias_probe"],  # Should match ethical_shadow_stack
        ["knowledge_query", "context_inject", "groundedness_check"],  # Should match RAG fast-path
        ["ihsan_check", "ihsan_check", "ihsan_check"],  # Should match benevolence cache
    ]
    
    for seq in test_sequences:
        result = engine.observe_sequence(seq)
        if result:
            print(f"    Sequence {seq} → Elevated to: {result.pattern_name}")
        else:
            print(f"    Sequence {seq} → No match (needs learning)")
    
    return {"patterns": len(engine.elevated_patterns), "snr_gain": total_snr_gain}


def probe_snr_tracker():
    """Probe SNR tracker for optimization opportunities."""
    print_header("SNR TRACKER: Signal-to-Noise Optimization")
    
    from bizra_kernel.snr_tracker import SNRTracker, SNRMetrics
    
    tracker = SNRTracker()
    
    # Simulate reasoning paths with varying SNR using proper API
    test_paths = [
        {"total_tokens": 100, "useful_tokens": 85, "confidence": 0.95, "ethics": 0.98, 
         "directness": 0.9, "latency": 100, "agent": "validator", "path": "Direct Ihsān validation"},
        {"total_tokens": 200, "useful_tokens": 140, "confidence": 0.85, "ethics": 0.95, 
         "directness": 0.75, "latency": 250, "agent": "verifier", "path": "Multi-step verification"},
        {"total_tokens": 300, "useful_tokens": 180, "confidence": 0.70, "ethics": 0.90, 
         "directness": 0.65, "latency": 400, "agent": "explorer", "path": "Exploratory reasoning (ToT)"},
        {"total_tokens": 250, "useful_tokens": 112, "confidence": 0.60, "ethics": 0.85, 
         "directness": 0.55, "latency": 500, "agent": "resolver", "path": "Ambiguous context resolution"},
    ]
    
    print("\n  Simulated reasoning path SNR:")
    for tp in test_paths:
        metrics = SNRMetrics(
            total_tokens=tp["total_tokens"],
            useful_tokens=tp["useful_tokens"],
            confidence_score=tp["confidence"],
            ethical_compliance=tp["ethics"],
            tool_directness=tp["directness"],
            latency_ms=tp["latency"],
            agent_role=tp["agent"],
        )
        snr = metrics.snr_score
        status = "✓" if snr >= 0.5 else "⚠" if snr >= 0.3 else "✗"
        print(f"    {status} {tp['path']}: SNR={snr:.3f}")
        tracker.record(metrics)
    
    avg_snr = tracker.get_average_snr()
    stats = tracker.get_statistics()
    print(f"\n  Aggregate metrics:")
    print(f"    Mean SNR: {avg_snr:.3f}")
    print(f"    Target SNR: {tracker.TARGET_SNR}")
    print(f"    Token waste: {stats.get('total_waste_percent', 0):.1f}%")
    
    return {"mean_snr": avg_snr}


def probe_ihsan_alignment():
    """Verify all components against Ihsān constitutional weights."""
    print_header("IHSĀN VERIFICATION: Constitutional Alignment")
    
    from bizra_kernel.ihsan_vector import constitution, IhsanDimension
    
    # Get weights from constitution
    const = constitution()
    weights = const.weights
    threshold = const.threshold  # Use .threshold, not .default_threshold
    
    print("\n  Constitutional weights (from ihsan_v1.yaml):")
    for dim in IhsanDimension:
        weight = weights.get(dim.value, 0.0)
        bar = "█" * int(weight * 50)
        print(f"    {dim.value:20s}: {weight:.2f} {bar}")
    
    print(f"\n  Threshold: {threshold}")
    
    # Test vector scoring
    print("\n  Testing Ihsān vector computation:")
    test_vectors = [
        {"correctness": 1.0, "safety": 1.0, "user_benefit": 0.9, "efficiency": 0.8,
         "auditability": 0.9, "anti_centralization": 0.7, "robustness": 0.8, "adl_fairness": 0.8},
        {"correctness": 0.5, "safety": 0.5, "user_benefit": 0.5, "efficiency": 0.5,
         "auditability": 0.5, "anti_centralization": 0.5, "robustness": 0.5, "adl_fairness": 0.5},
        {"correctness": 1.0, "safety": 0.0, "user_benefit": 1.0, "efficiency": 1.0,
         "auditability": 0.0, "anti_centralization": 0.0, "robustness": 0.0, "adl_fairness": 0.0},
    ]
    
    for i, dims in enumerate(test_vectors):
        # Compute weighted score using constitution weights
        score = sum(dims[d] * weights.get(d, 0.0) for d in dims)
        status = "✓ PASS" if score >= threshold else "✗ FAIL"
        print(f"    Vector {i+1}: score={score:.3f} {status}")
    
    return {"threshold": threshold}


def probe_lexicon_coverage():
    """Analyze Lexicon Ledger term coverage and gaps."""
    print_header("LEXICON LEDGER: Term Coverage Analysis")
    
    ledger = LexiconLedger()
    stats = ledger.get_stats()
    
    print(f"\n  DNA Signature: {DNA_SIGNATURE}")
    print(f"  Total terms: {stats['total_terms']}")
    print(f"  Canonical: {stats['canonical_terms']}")
    print(f"  Deprecated: {stats['deprecated_terms']}")
    
    print("\n  Terms by Ihsān dimension:")
    for dim, count in sorted(stats['terms_by_ihsan_dimension'].items(), 
                              key=lambda x: x[1], reverse=True):
        bar = "█" * count
        print(f"    {dim:20s}: {count:2d} {bar}")
    
    print("\n  Terms by SAPE module:")
    for mod, count in sorted(stats['terms_by_sape_module'].items()):
        bar = "█" * count
        print(f"    Module {mod}: {count:2d} {bar}")
    
    # Check for coverage gaps
    print("\n  Coverage gaps:")
    from bizra_kernel.ihsan_vector import IhsanDimension
    all_dims = {d.value for d in IhsanDimension}
    missing_dims = all_dims - set(stats['terms_by_ihsan_dimension'].keys())
    if missing_dims:
        print(f"    ⚠ Missing Ihsān dimensions: {missing_dims}")
    else:
        print(f"    ✓ All Ihsān dimensions covered")
    
    missing_modules = set(range(1, 8)) - set(stats['terms_by_sape_module'].keys())
    if missing_modules:
        print(f"    ⚠ Missing SAPE modules: {missing_modules}")
    else:
        print(f"    ✓ All SAPE modules covered")
    
    return stats


def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " BIZRA SAPE DEEP CIRCUIT PROBE & MULTI-LENS ANALYSIS ".center(68) + "║")
    print("║" + f" DNA Signature: {DNA_SIGNATURE} ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = {}
    
    # Run all probes
    results["tensions"] = probe_tension_circuits()
    results["symbolic"] = probe_symbolic_neural_bridge()
    results["abstraction"] = probe_abstraction_elevation()
    results["sape"] = probe_sape_engine()
    results["snr"] = probe_snr_tracker()
    results["ihsan"] = probe_ihsan_alignment()
    results["lexicon"] = probe_lexicon_coverage()
    
    # Final synthesis
    print_header("SYNTHESIS: Multi-Lens Analysis Complete")
    
    print("\n  Key findings:")
    print(f"    • Tension detection: {results['tensions']['detected']} contradictions found")
    print(f"    • Symbolic grounding: {results['symbolic']['symbols']} symbols registered")
    print(f"    • Abstraction elevator: {results['abstraction']['principles']} principles active")
    print(f"    • SAPE optimization: +{results['sape']['snr_gain']:.0%} SNR potential")
    print(f"    • Mean SNR: {results['snr']['mean_snr']:.2f}")
    print(f"    • Lexicon coverage: {results['lexicon']['total_terms']} canonical terms")
    
    print("\n  Untapped circuits identified:")
    print("    1. Cross-domain principle transfer (AbstractionElevator)")
    print("    2. Real-time tension resolution with SAT consensus")
    print("    3. Bi-directional symbol↔embedding lifting")
    print("    4. Dynamic SAPE pattern learning from production traces")
    
    print("\n" + "═" * 70)
    print("  PROBE COMPLETE - All circuits exercised")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
