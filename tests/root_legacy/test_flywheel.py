#!/usr/bin/env python3
"""
BIZRA Flywheel Quick Test
Tests the flywheel components locally before Docker deployment.
"""

import asyncio
import os
import subprocess
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_lmstudio():
    """Test LM Studio connectivity."""
    import httpx

    print("\n[1] Testing LM Studio at http://192.168.56.1:1234...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://192.168.56.1:1234/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                print(f"    ✅ LM Studio available with {len(models)} models")
                for m in models[:3]:
                    print(f"       - {m.get('id')}")
                if len(models) > 3:
                    print(f"       ... and {len(models) - 3} more")
                return True
            else:
                print(f"    ❌ LM Studio returned {response.status_code}")
                return False
    except Exception as e:
        print(f"    ❌ LM Studio not reachable: {e}")
        return False


async def test_ollama():
    """Test Ollama connectivity."""
    import httpx

    print("\n[2] Testing Ollama at http://localhost:11434...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"    ✅ Ollama available with {len(models)} models")
                for m in models[:3]:
                    print(f"       - {m.get('name')}")
                return True
            else:
                print(f"    ❌ Ollama returned {response.status_code}")
                return False
    except Exception as e:
        print(f"    ❌ Ollama not reachable: {e}")
        return False


async def test_inference():
    """Test LLM inference."""
    print("\n[3] Testing LLM inference...")

    try:
        from flywheel import LocalInference

        inference = LocalInference()

        # Quick inference test
        response = await inference.generate(
            prompt="Say 'Flywheel test successful' in 4 words or less.",
            max_tokens=20,
            temperature=0.1,
        )

        print(f"    ✅ Inference works: '{response.strip()}'")
        return True

    except Exception as e:
        print(f"    ❌ Inference failed: {e}")
        return False


async def test_audio():
    """Test audio processing availability."""
    print("\n[4] Testing audio processing...")

    # Check faster-whisper
    try:
        import faster_whisper

        print(f"    ✅ faster-whisper available (v{faster_whisper.__version__})")
        stt_ok = True
    except ImportError:
        print("    ⚠️  faster-whisper not installed")
        stt_ok = False

    # Check edge-tts
    try:
        import edge_tts

        print("    ✅ edge-tts available")
        tts_ok = True
    except ImportError:
        print("    ⚠️  edge-tts not installed")
        tts_ok = False

    return stt_ok or tts_ok


async def test_auth():
    """Test fail-closed authentication."""
    print("\n[5] Testing fail-closed auth...")

    try:
        from flywheel import AuthResult, FailClosedAuth

        auth = FailClosedAuth("FAIL_CLOSED")

        # Without token, should deny
        result = auth.authenticate(None, "test")
        if result == AuthResult.DENIED or result == AuthResult.MISSING:
            print("    ✅ Correctly denies without token")
        else:
            print(f"    ❌ Should deny, got {result}")
            return False

        # Set a token and test
        os.environ["BIZRA_API_TOKEN"] = "test_token_12345"
        auth = FailClosedAuth("FAIL_CLOSED")

        result = auth.authenticate("test_token_12345", "test")
        if result == AuthResult.ALLOWED:
            print("    ✅ Correctly allows with valid token")
        else:
            print(f"    ❌ Should allow, got {result}")
            return False

        result = auth.authenticate("wrong_token", "test")
        if result == AuthResult.DENIED:
            print("    ✅ Correctly denies wrong token")
        else:
            print(f"    ❌ Should deny, got {result}")
            return False

        return True

    except Exception as e:
        print(f"    ❌ Auth test failed: {e}")
        return False


async def main():
    print("═" * 60)
    print("    BIZRA FLYWHEEL — Quick Test")
    print("═" * 60)

    results = {}

    results["lmstudio"] = await test_lmstudio()
    results["ollama"] = await test_ollama()
    results["inference"] = await test_inference()
    results["audio"] = await test_audio()
    results["auth"] = await test_auth()

    print("\n" + "═" * 60)
    print("    RESULTS")
    print("═" * 60)

    all_pass = True
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"    {icon} {name}")
        if not passed and name in ["inference", "auth"]:
            all_pass = False

    print()
    if all_pass:
        print("    🎯 Ready for deployment!")
        print("    Run: ./activate_flywheel.sh")
    else:
        print("    ⚠️  Some issues to resolve")

    print("═" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        import httpx
    except ImportError:
        print("Installing httpx...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "httpx"]
        )  # noqa: S603
        import httpx

    sys.exit(asyncio.run(main()))
