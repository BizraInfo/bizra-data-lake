#!/usr/bin/env python3
"""
Standalone example of BIZRA Multi-Modal Router.
Does not require full project dependencies.
"""

import sys
from pathlib import Path

# Add just the multimodal module
sys.path.insert(0, str(Path(__file__).parent))

from multimodal import (  # type: ignore[import-not-found]
    ModelCapability,
    get_multimodal_router,
)


def main():
    print("\n" + "=" * 75)
    print("  BIZRA Multi-Modal Router - Standalone Example")
    print("=" * 75)

    # Get router (singleton)
    router = get_multimodal_router()

    # Example 1: Basic routing
    print("\n1. BASIC TASK ROUTING")
    print("-" * 75)

    tasks = [
        ("What is 2 + 2?", "simple math"),
        ("Prove that sqrt(2) is irrational", "mathematical reasoning"),
        ("Describe the objects in this image", "vision understanding"),
        ("Transcribe the speech in this audio", "speech recognition"),
        ("Write a haiku about nature", "creative writing"),
    ]

    for task_text, description in tasks:
        decision = router.route(task_text)
        print(f"\n  Task: {description}")
        print(f"    Query: '{task_text[:45]}...'")
        print(f"    Model: {decision.model.name} ({decision.model.params_b}B)")
        print(f"    Capability: {decision.capability_match.value}")
        print(f"    Confidence: {decision.confidence:.0%}")
        print(f"    Backend: {decision.model.backend}")

    # Example 2: Explicit type override
    print("\n\n2. EXPLICIT TYPE OVERRIDE")
    print("-" * 75)

    ambiguous_task = "Extract text from this"
    decision_text = router.route(ambiguous_task)
    decision_image = router.route(ambiguous_task, explicit_type="image")

    print(f"\n  Ambiguous task: '{ambiguous_task}'")
    print("\n  Default routing (text detection):")
    print(f"    -> {decision_text.model.name}")
    print(f"    -> {decision_text.reason}")
    print("\n  With explicit_type='image':")
    print(f"    -> {decision_image.model.name}")
    print(f"    -> {decision_image.reason}")

    # Example 3: Capability selection
    print("\n\n3. CAPABILITY-BASED SELECTION")
    print("-" * 75)

    for capability in ModelCapability:
        decision = router.select_model(capability)
        print(f"\n  {capability.value.upper()}:")
        print(f"    Primary model: {decision.model.name}")
        print(f"    Parameters: {decision.model.params_b}B")
        print(f"    Speed: ~{decision.model.speed_tok_per_sec:.0f} tokens/sec")
        print(f"    Confidence: {decision.confidence:.0%}")

    # Example 4: Model inventory
    print("\n\n4. AVAILABLE MODELS")
    print("-" * 75)

    all_models = router.list_models()
    print(f"\n  Total models: {len(all_models)}")

    print("\n  By capability:")
    for cap in ModelCapability:
        models = router.list_by_capability(cap)
        print(f"\n    {cap.value.upper()} ({len(models)} models):")
        for m in models:
            is_primary = "PRIMARY" if m.primary_capability == cap else "secondary"
            print(f"      • {m.name:20} {m.params_b:5.1f}B ({is_primary})")

    # Example 5: Confidence-aware decisions
    print("\n\n5. CONFIDENCE SCORES & FALLBACKS")
    print("-" * 75)

    decision = router.select_model(ModelCapability.REASONING)

    print("\n  Reasoning model selection:")
    print(f"    Model: {decision.model.name}")
    print(f"    Confidence: {decision.confidence:.0%}")
    print(f"    Reason: {decision.reason}")

    if decision.alternatives:
        print("\n  Fallback alternatives:")
        for i, alt in enumerate(decision.alternatives, 1):
            print(f"      {i}. {alt.name} ({alt.params_b}B)")

    # Example 6: Fast vision model for real-time
    print("\n\n6. REAL-TIME OPTIMIZATION")
    print("-" * 75)

    vision_models = sorted(
        router.list_by_capability(ModelCapability.VISION),
        key=lambda m: m.speed_tok_per_sec,
        reverse=True,
    )

    print("\n  Vision models by speed (for real-time processing):")
    for model in vision_models:
        print(f"    {model.name:20} {model.speed_tok_per_sec:5.0f} tok/s")

    fastest = vision_models[0]
    print(f"\n  Recommended for real-time: {fastest.name}")

    # Example 7: Large reasoning model for accuracy
    print("\n\n7. ACCURACY-FOCUSED ROUTING")
    print("-" * 75)

    reasoning_models = sorted(
        router.list_by_capability(ModelCapability.REASONING),
        key=lambda m: m.params_b,
        reverse=True,
    )

    print("\n  Reasoning models by size (for maximum accuracy):")
    for model in reasoning_models:
        print(f"    {model.name:20} {model.params_b:6.1f}B")

    largest = reasoning_models[0]
    print(f"\n  Recommended for complex analysis: {largest.name}")

    # Summary
    print("\n\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    print(f"""
  The BIZRA Multi-Modal Router provides intelligent task routing:

  Key Features:
    ✓ Automatic capability detection from task description
    ✓ Specialized models for reasoning, vision, voice
    ✓ Confidence-based fallback chains
    ✓ Speed-optimized selection for real-time tasks
    ✓ Accuracy-focused selection for complex analysis
    ✓ Easy extensibility for custom models

  Default Backends:
    • LM Studio: {router.config.lmstudio_endpoint}
    • Ollama: {router.config.ollama_endpoint}

  Available Models: {len(router.list_models())}
    • Reasoning: {len(router.list_by_capability(ModelCapability.REASONING))}
    • Vision: {len(router.list_by_capability(ModelCapability.VISION))}
    • Voice: {len(router.list_by_capability(ModelCapability.VOICE))}
    • General: {len(router.list_by_capability(ModelCapability.GENERAL))}
    """)

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
