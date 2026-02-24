#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              PEAK MASTERPIECE — COMPREHENSIVE TEST SUITE                                     ║
║                           Giants Protocol + SNR + FATE + Ihsān Verification                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

print("=" * 80)
print("🔥 PEAK MASTERPIECE ENGINE — COMPREHENSIVE TEST SUITE")
print("=" * 80)

print("\n⏳ Loading Peak Masterpiece Engine...")
start_load = time.time()

from peak_masterpiece import (
    SNR_ACCEPTABLE,
    SNR_IHSAN,
    SNR_MINIMUM,
    PeakMasterpieceEngine,
)

engine = PeakMasterpieceEngine()
load_time = time.time() - start_load

print(f"\n✅ Engine Loaded in {load_time:.2f}s")
print("-" * 80)

# Test Queries across different domains
test_queries = [
    # Domain 1: IEEE Survey (Known document)
    {
        "query": "What does the IEEE survey say about Agentic Data Cleaning with LLMs?",
        "domain": "Research/AI",
        "expected_source": "IEEE_TKDE",
    },
    # Domain 2: Technical Architecture
    {
        "query": "Explain the Graph of Thoughts reasoning paradigm",
        "domain": "Cognitive Architecture",
        "expected_source": None,
    },
    # Domain 3: BIZRA-specific
    {
        "query": "What are the DDAGI kernel invariants?",
        "domain": "BIZRA Constitution",
        "expected_source": "DDAGI",
    },
    # Domain 4: Cross-domain synthesis
    {
        "query": "How can information theory improve data quality assessment?",
        "domain": "Interdisciplinary",
        "expected_source": None,
    },
]

results_summary = []
total_time = 0

for i, test in enumerate(test_queries, 1):
    query = test["query"]
    domain = test["domain"]

    print(f"\n{'='*80}")
    print(f"📝 TEST {i}/{len(test_queries)}: [{domain}]")
    print(f"   Query: {query[:60]}...")
    print("=" * 80)

    start = time.time()
    result = engine.process_query(query)
    elapsed = time.time() - start
    total_time += elapsed

    # Determine SNR grade
    if result.snr_score >= SNR_IHSAN:
        grade = "🏆 IHSĀN"
        grade_symbol = "🏆"
    elif result.snr_score >= SNR_ACCEPTABLE:
        grade = "✅ ACCEPTABLE"
        grade_symbol = "✅"
    elif result.snr_score >= SNR_MINIMUM:
        grade = "⚠️ MINIMUM"
        grade_symbol = "⚠️"
    else:
        grade = "❌ BELOW"
        grade_symbol = "❌"

    # Check if expected source was retrieved
    source_found = "N/A"
    if test.get("expected_source"):
        synthesis_lower = result.synthesis.lower() if result.synthesis else ""
        if test["expected_source"].lower() in synthesis_lower:
            source_found = "✅ Found"
        else:
            source_found = "❌ Missing"

    print(f"\n📊 RESULTS:")
    print(f"   SNR Score: {result.snr_score:.4f} — {grade}")
    print(f"   Discipline Coverage: {result.discipline_coverage:.1%}")
    print(f"   Synergies: {len(result.synergies)}")
    print(f"   Execution: {elapsed:.2f}s")
    print(f"   Source Check: {source_found}")

    if result.synergies:
        print(
            f"\n🔗 Top Synergy: {result.synergies[0].source_domain} ↔ {result.synergies[0].target_domain}"
        )

    # Show full synthesis for first test
    if i == 1:
        print(f"\n📜 FULL SYNTHESIS:")
        print("-" * 60)
        print(result.synthesis[:1200] if result.synthesis else "No synthesis")
        print("-" * 60)

    # Store for summary
    results_summary.append(
        {
            "test": i,
            "domain": domain,
            "snr": result.snr_score,
            "grade": grade_symbol,
            "time": elapsed,
            "synergies": len(result.synergies),
            "source": source_found,
        }
    )

# Final Summary
print("\n" + "=" * 80)
print("📈 TEST SUITE SUMMARY")
print("=" * 80)

print(
    "\n┌─────┬──────────────────────┬─────────┬───────┬─────────┬──────────┬──────────┐"
)
print("│ # │ Domain │ SNR │ Grade │ Time(s) │ Synergies│ Source │")
print(
    "├─────┼──────────────────────┼─────────┼───────┼─────────┼──────────┼──────────┤"
)

for r in results_summary:
    print(
        f"│ {r['test']:>3} │ {r['domain'][:20]:<20} │ {r['snr']:.4f} │   {r['grade']}   │  {r['time']:>5.2f}  │    {r['synergies']:>2}    │ {r['source']:<8} │"
    )

print(
    "└─────┴──────────────────────┴─────────┴───────┴─────────┴──────────┴──────────┘"
)

avg_snr = sum(r["snr"] for r in results_summary) / len(results_summary)
avg_time = total_time / len(results_summary)

print(f"\n📊 AGGREGATE METRICS:")
print(f"   Total Tests: {len(test_queries)}")
print(f"   Average SNR: {avg_snr:.4f}")
print(f"   Average Time: {avg_time:.2f}s")
print(f"   Total Time: {total_time:.2f}s")
print(f"   Engine Load: {load_time:.2f}s")

# System info
if hasattr(engine, "_warp_retriever") and engine._warp_retriever:
    kb_size = (
        len(engine._warp_retriever.metadata)
        if hasattr(engine._warp_retriever, "metadata")
        else "N/A"
    )
    backend = (
        engine._warp_retriever.backend.name
        if hasattr(engine._warp_retriever, "backend")
        else "Unknown"
    )
else:
    kb_size = "Legacy"
    backend = "KnowledgeRetriever"

print(f"\n🔧 SYSTEM INFO:")
print(f"   Retriever Backend: {backend}")
print(f"   Knowledge Base: {kb_size} documents")
print(f"   LLM Endpoint: http://192.168.56.1:1234")

print("\n" + "=" * 80)
print("✅ COMPREHENSIVE TEST SUITE COMPLETE")
print("=" * 80)
