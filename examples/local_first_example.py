"""
Example: Using Local-First Backend Detection

Demonstrates how to use LocalFirstDetector for zero-token inference.

Run:
  python examples/local_first_example.py
"""

import asyncio
from core.inference import (
    LocalFirstDetector,
    get_local_first_backend,
    LocalBackend,
)


async def example_1_auto_detect():
    """Example 1: Auto-detect best backend."""
    print("=" * 60)
    print("EXAMPLE 1: Auto-Detect Best Backend")
    print("=" * 60)

    backend = await get_local_first_backend()
    print(f"Selected backend: {backend.value}")

    if backend == LocalBackend.NONE:
        print("No local backends available. Start LM Studio or Ollama.")
    print()


async def example_2_full_status():
    """Example 2: Get status of all backends."""
    print("=" * 60)
    print("EXAMPLE 2: Full Backend Status Report")
    print("=" * 60)

    statuses = await LocalFirstDetector.detect_available()

    print(f"{'Backend':<12} {'Status':<10} {'Latency':<12} Reason")
    print("-" * 60)

    for status in statuses:
        status_str = "READY" if status.available else "offline"
        print(
            f"{status.backend.value:<12} {status_str:<10} "
            f"{status.latency_ms:>6.1f}ms    {status.reason}"
        )

    if statuses:
        available = [s for s in statuses if s.available]
        if available:
            best = available[0]
            print("-" * 60)
            print(f"BEST CHOICE: {best.backend.value} ({best.latency_ms:.1f}ms)")
    print()


async def example_3_select_logic():
    """Example 3: Custom selection logic."""
    print("=" * 60)
    print("EXAMPLE 3: Custom Backend Selection")
    print("=" * 60)

    statuses = await LocalFirstDetector.detect_available()

    # Strategy: Prefer low latency
    available = [s for s in statuses if s.available]

    if not available:
        print("No backends available!")
        return

    # Find fastest
    fastest = min(available, key=lambda s: s.latency_ms)
    print(f"Fastest backend: {fastest.backend.value} ({fastest.latency_ms:.1f}ms)")

    # Find most reliable (based on reason)
    print("\nAvailable backends (sorted by latency):")
    for status in sorted(available, key=lambda s: s.latency_ms):
        print(f"  {status.backend.value:<12} {status.latency_ms:>6.1f}ms")
    print()


async def example_4_with_config():
    """Example 4: Using selected backend in configuration."""
    print("=" * 60)
    print("EXAMPLE 4: Pass to Configuration")
    print("=" * 60)

    backend = await get_local_first_backend()

    # Simulated config usage
    config = {
        "inference_backend": backend.value,
        "fallback_to_api": backend == LocalBackend.NONE,
    }

    print(f"Configuration:")
    print(f"  inference_backend: {config['inference_backend']}")
    print(f"  fallback_to_api: {config['fallback_to_api']}")
    print()


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║  BIZRA LOCAL-FIRST BACKEND DETECTION - Examples        ║")
    print("╚" + "═" * 58 + "╝")
    print()

    await example_1_auto_detect()
    await example_2_full_status()
    await example_3_select_logic()
    await example_4_with_config()

    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
