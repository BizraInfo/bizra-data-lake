"""
Examples: Using BIZRA Multi-Modal Router

Shows practical patterns for routing tasks to the right model.
"""

from core.inference.multimodal import (
    ModelCapability,
    MultiModalConfig,
    MultiModalRouter,
    get_multimodal_router,
)


def example_1_basic_routing():
    """Example 1: Basic task routing."""
    print("\n" + "=" * 75)
    print("Example 1: Basic Task Routing")
    print("=" * 75)

    router = get_multimodal_router()

    tasks = [
        "What is 2 + 2?",
        "Prove that all odd numbers > 5 can be expressed as sum of three primes",
        "Describe the objects in this image",
        "Transcribe the speech in this audio",
    ]

    for task in tasks:
        decision = router.route(task)
        print(f"\nTask: {task[:50]}...")
        print(f"  Model: {decision.model.name} ({decision.model.params_b}B)")
        print(f"  Capability: {decision.capability_match.value}")
        print(f"  Confidence: {decision.confidence:.0%}")


def example_2_explicit_types():
    """Example 2: Using explicit type hints."""
    print("\n" + "=" * 75)
    print("Example 2: Explicit Type Hints")
    print("=" * 75)

    router = get_multimodal_router()

    # Sometimes text doesn't clearly indicate modality
    # Use explicit_type to override detection
    decision = router.route(
        "What does this say?", explicit_type="image"  # Override: treat as vision task
    )

    print("\nTask: 'What does this say?'")
    print("  Explicit type: image")
    print(f"  Selected model: {decision.model.name}")
    print(f"  Reason: {decision.reason}")


def example_3_dict_specification():
    """Example 3: Using dict task specification."""
    print("\n" + "=" * 75)
    print("Example 3: Dict Task Specification")
    print("=" * 75)

    router = get_multimodal_router()

    # More explicit specification
    task = {
        "text": "Extract all text from this document image",
        "has_image": True,
        "has_audio": False,
    }

    decision = router.route(task)

    print("\nTask specification:")
    print(f"  Text: {task['text'][:50]}...")
    print(f"  Has image: {task['has_image']}")
    print(f"  Has audio: {task['has_audio']}")
    print(f"  Selected model: {decision.model.name}")
    print(f"  Backend: {decision.model.backend}")


def example_4_capability_selection():
    """Example 4: Direct capability selection."""
    print("\n" + "=" * 75)
    print("Example 4: Direct Capability Selection")
    print("=" * 75)

    router = get_multimodal_router()

    # When you know exactly what capability you need
    for capability in ModelCapability:
        decision = router.select_model(capability)
        print(f"\n{capability.value.upper()}:")
        print(f"  Primary model: {decision.model.name} ({decision.model.params_b}B)")
        print(f"  Backend: {decision.model.backend}")
        print(f"  Speed: ~{decision.model.speed_tok_per_sec:.0f} tokens/sec")

        if decision.alternatives:
            print("  Alternatives:")
            for alt in decision.alternatives[:2]:
                print(f"    - {alt.name}")


def example_5_custom_model_registration():
    """Example 5: Registering custom models."""
    print("\n" + "=" * 75)
    print("Example 5: Custom Model Registration")
    print("=" * 75)

    from core.inference.multimodal import ModelInfo

    # Create custom config and router (not singleton)
    config = MultiModalConfig(
        lmstudio_endpoint="192.168.1.100:5000",
    )
    router = MultiModalRouter(config)

    # Register a large custom model
    custom_model = ModelInfo(
        name="local-llama-70b",
        capabilities=[ModelCapability.REASONING, ModelCapability.GENERAL],
        primary_capability=ModelCapability.REASONING,
        backend="lmstudio",
        endpoint="192.168.1.100:5000",
        params_b=70.0,
        context_length=4096,
        speed_tok_per_sec=8.0,
        description="Llama 2 70B for complex reasoning tasks",
    )

    router.register_model(custom_model)

    # Now it will be considered for reasoning tasks
    decision = router.select_model(ModelCapability.REASONING)

    print(f"\nRegistered custom model: {custom_model.name}")
    print(f"Model registry now has {len(router.list_models())} models")
    print(f"\nBest reasoning model: {decision.model.name}")
    print(f"  Size: {decision.model.params_b}B")
    print(f"  Speed: ~{decision.model.speed_tok_per_sec} tok/s")


def example_6_confidence_aware_routing():
    """Example 6: Using confidence scores for decision making."""
    print("\n" + "=" * 75)
    print("Example 6: Confidence-Aware Routing")
    print("=" * 75)

    router = get_multimodal_router()

    ambiguous_tasks = [
        "Convert speech to text",  # Clear: voice
        "Summarize the document",  # Ambiguous: could be vision (OCR) or text
        "What's in the picture?",  # Clear: vision
    ]

    for task in ambiguous_tasks:
        decision = router.route(task)

        # Use confidence to decide if we need verification
        if decision.confidence < 0.8:
            action = "Ask user for clarification"
        else:
            action = "Proceed with routing"

        print(f"\nTask: {task}")
        print(f"  Selected: {decision.model.name}")
        print(f"  Confidence: {decision.confidence:.0%}")
        print(f"  Action: {action}")


def example_7_listing_models():
    """Example 7: Discovering available models."""
    print("\n" + "=" * 75)
    print("Example 7: Discovering Available Models")
    print("=" * 75)

    router = get_multimodal_router()

    print("\nAll available models:")
    print(f"Total: {len(router.list_models())}")

    for model in router.list_models():
        ", ".join(c.value for c in model.capabilities)
        print(
            f"  {model.name:20} ({model.params_b:5.1f}B) - "
            f"{model.primary_capability.value:10} | {model.backend}"
        )

    print("\n\nModels by capability:")
    for capability in ModelCapability:
        models = router.list_by_capability(capability)
        print(f"\n{capability.value.upper()} ({len(models)} models):")
        for model in models:
            speed = f"~{model.speed_tok_per_sec:.0f} tok/s"
            print(f"  {model.name:20} | {speed:15} | {model.description}")


def example_8_adaptive_speed_selection():
    """Example 8: Speed-optimized selection for real-time tasks."""
    print("\n" + "=" * 75)
    print("Example 8: Speed-Optimized Selection")
    print("=" * 75)

    router = get_multimodal_router()

    print("\nFor real-time applications, prefer fastest models:")

    # Vision tasks need low latency (streaming video analysis)
    vision_models = router.list_by_capability(ModelCapability.VISION)
    vision_by_speed = sorted(
        vision_models, key=lambda m: m.speed_tok_per_sec, reverse=True
    )

    print("\nVision models (fastest first):")
    for model in vision_by_speed:
        print(f"  {model.name:20} - ~{model.speed_tok_per_sec:5.0f} tok/s")

    fastest_vision = vision_by_speed[0]
    print(f"\nFor real-time video: use {fastest_vision.name}")

    # Reasoning tasks can tolerate higher latency (offline analysis)
    reasoning_models = router.list_by_capability(ModelCapability.REASONING)
    reasoning_by_size = sorted(reasoning_models, key=lambda m: m.params_b, reverse=True)

    print("\nReasoning models (largest first):")
    for model in reasoning_by_size:
        print(f"  {model.name:20} - {model.params_b:6.1f}B")

    largest_reasoning = reasoning_by_size[0]
    print(f"\nFor complex analysis: use {largest_reasoning.name}")


def example_9_fallback_chain():
    """Example 9: Understanding fallback behavior."""
    print("\n" + "=" * 75)
    print("Example 9: Fallback Chain")
    print("=" * 75)

    router = get_multimodal_router()

    decision = router.select_model(ModelCapability.REASONING)

    print("\nReasoning model selection:")
    print(f"  Primary choice: {decision.model.name}")
    print(f"  Confidence: {decision.confidence:.0%}")
    print(f"  Reason: {decision.reason}")

    if decision.alternatives:
        print("\n  Fallback alternatives:")
        for i, alt in enumerate(decision.alternatives, 1):
            print(f"    {i}. {alt.name} ({alt.params_b}B)")


def example_10_batch_routing():
    """Example 10: Routing multiple tasks efficiently."""
    print("\n" + "=" * 75)
    print("Example 10: Batch Task Routing")
    print("=" * 75)

    router = get_multimodal_router()

    # Batch of mixed tasks
    tasks = [
        {"text": "Solve x^2 + 2x + 1 = 0", "type": "reasoning"},
        {"text": "What objects are in this photo?", "type": "vision"},
        {"text": "Convert speech to text", "type": "voice"},
        {"text": "Write a haiku", "type": "general"},
    ]

    print("\nRouting batch of tasks:")
    for task in tasks:
        decision = router.route(task["text"])
        print(
            f"  {task['type']:10} -> {decision.model.name:20} "
            f"({decision.confidence:.0%} confidence)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 75)
    print("█" + " " * 73 + "█")
    print("█" + "  BIZRA Multi-Modal Router Examples".center(73) + "█")
    print("█" + " " * 73 + "█")
    print("█" * 75)

    # Run all examples
    example_1_basic_routing()
    example_2_explicit_types()
    example_3_dict_specification()
    example_4_capability_selection()
    example_5_custom_model_registration()
    example_6_confidence_aware_routing()
    example_7_listing_models()
    example_8_adaptive_speed_selection()
    example_9_fallback_chain()
    example_10_batch_routing()

    print("\n" + "█" * 75)
    print("Examples complete!")
    print("█" * 75 + "\n")
