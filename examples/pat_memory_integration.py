#!/usr/bin/env python3
"""
PAT Memory Integration Example
===============================
Demonstrates how to use PAT memory in the BIZRA kernel.
"""

import asyncio
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pat_memory import get_pat_memory


async def example_user_preferences():
    """Example: Store and retrieve user preferences."""
    memory = await get_pat_memory()

    print("\n📝 Storing user preferences...")
    await memory.store("user_preferences", "theme", "dark")
    await memory.store("user_preferences", "language", "en")
    await memory.store("user_preferences", "favorite_model", "deepseek-r1:14b")
    await memory.store("user_preferences", "local_first", True)

    print("✅ Preferences stored")

    # Retrieve a specific preference
    theme = await memory.retrieve("user_preferences", "theme")
    print(f"   Theme: {theme}")

    # Retrieve all preferences
    prefs = await memory.retrieve_all("user_preferences")
    print(f"   All preferences: {prefs}")


async def example_session_tracking():
    """Example: Track session history."""
    memory = await get_pat_memory()

    print("\n📊 Tracking session...")
    session_data = {
        "session_id": "sess-demo-001",
        "timestamp": "2026-02-14T07:00:00Z",
        "task": "Implement SAPE validation",
        "outcome": "success",
        "duration_seconds": 120,
        "model_used": "deepseek-r1:14b",
    }

    await memory.store("session_history", session_data["session_id"], session_data)
    print(f"✅ Session {session_data['session_id']} recorded")


async def example_pattern_learning():
    """Example: Learn patterns about user workflow."""
    memory = await get_pat_memory()

    print("\n🧠 Learning patterns...")
    pattern = {
        "pattern": "User prefers local-first models for privacy",
        "confidence": 0.90,
        "observations": 15,
        "evidence": [
            "Always selects local Ollama models",
            "Avoids cloud API calls",
            "Prefers offline operation",
        ],
    }

    await memory.learn_pattern("privacy_preference", pattern)
    print("✅ Pattern learned: privacy_preference")


async def example_model_routing():
    """Example: Track which models work best for which tasks."""
    memory = await get_pat_memory()

    print("\n🤖 Recording model performance...")
    routing = {
        "task_type": "code_generation",
        "best_model": "deepseek-r1:14b",
        "avg_latency_ms": 850,
        "success_rate": 0.95,
        "trials": 20,
    }

    await memory.store("model_routing", "code_generation", routing)
    print("✅ Model routing preference stored")


async def example_llm_context_injection():
    """Example: Inject user context into LLM system prompt."""
    memory = await get_pat_memory()

    print("\n🧬 Generating LLM context...")
    context = await memory.get_user_context()

    # This context can be injected into system prompts
    system_prompt = f"""
You are PAT (Personal Agentic Team) for this user.

User Profile:
- Theme: {context['user_preferences'].get('theme', 'light')}
- Language: {context['user_preferences'].get('language', 'en')}
- Favorite Model: {context['user_preferences'].get('favorite_model', 'default')}
- Privacy Mode: {context['user_preferences'].get('local_first', False)}

Recent Sessions: {len(context['recent_sessions'])}
Learned Patterns: {len(context['learned_patterns'])}

System:
- GPU: {context['system_config'].get('detected', {}).get('gpu', {}).get('name', 'None')}
- RAM: {context['system_config'].get('detected', {}).get('ram', {}).get('total_gb', 0)} GB
- Available Models: {len(context['system_config'].get('detected', {}).get('ollama_models', []))}

Adapt your responses to this user's preferences and system capabilities.
"""

    print(system_prompt)
    print("✅ Context ready for LLM injection")


async def example_persistence():
    """Example: Demonstrate persistence across restarts."""
    memory = await get_pat_memory()

    print("\n💾 Testing persistence...")

    # Store something
    await memory.store("user_preferences", "test_key", "test_value")

    # Sync to disk
    await memory.sync_to_disk()
    print("✅ Data synced to disk")

    # Close and reopen (simulates restart)
    await memory.close()

    # Create new instance (simulates restart)
    memory2 = await get_pat_memory()

    # Retrieve from cold storage
    value = await memory2.retrieve("user_preferences", "test_key")
    print(f"✅ Retrieved after restart: {value}")

    assert value == "test_value", "Persistence failed!"
    print("✅ Persistence verified")


async def main():
    """Run all examples."""
    print("=" * 70)
    print("  PAT MEMORY INTEGRATION EXAMPLES")
    print("=" * 70)

    await example_user_preferences()
    await example_session_tracking()
    await example_pattern_learning()
    await example_model_routing()
    await example_llm_context_injection()
    await example_persistence()

    print("\n" + "=" * 70)
    print("  ALL EXAMPLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
