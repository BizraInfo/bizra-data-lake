#!/usr/bin/env python3
"""
BIZRA Node0 — Auto Model Loader
═══════════════════════════════════════════════════════════════════════════════

Ensures the always_loaded models from proactive_config.yaml are in VRAM
before Node0 starts missions. Eliminates the recurring "0 models loaded"
problem by auto-loading via LM Studio's /api/v1/models/load endpoint.

Usage:
    python scripts/ensure_models_loaded.py           # Load always_loaded models
    python scripts/ensure_models_loaded.py --status   # Just check, don't load
    python scripts/ensure_models_loaded.py --model X  # Load a specific model

Can also be imported:
    from scripts.ensure_models_loaded import ensure_fleet_loaded
    loaded = await ensure_fleet_loaded()

Standing on Giants:
- Boyd (OODA pre-staging): absorb cold-start latency before first request
- Shannon (capacity planning): verify channel before transmitting
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load .env (same as node0_activate.py)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(PROJECT_ROOT) / ".env", override=True)
except ImportError:
    pass


def _resolve_base_url() -> str:
    """Resolve LM Studio base URL with auto-detected WSL gateway."""
    url = os.getenv("LM_STUDIO_URL")
    if url:
        return url.rstrip("/").replace("/v1", "")

    from core.integration.constants import LMSTUDIO_HOST, LMSTUDIO_PORT

    return f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}"


def _resolve_token() -> str:
    """Resolve LM Studio auth token."""
    return os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY") or ""


def _load_config() -> Dict[str, Any]:
    """Load proactive_config.yaml for model routing settings."""
    config_path = Path(PROJECT_ROOT) / "config" / "proactive_config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


async def get_loaded_models(
    base_url: str, token: str
) -> tuple[List[str], List[str]]:
    """Check which models are available and which are loaded in VRAM.

    Returns (available_ids, loaded_ids).
    """
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
        # Native API — has accurate loaded_instances field
        try:
            resp = await client.get(f"{base_url}/api/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", data.get("data", []))
                available = [m.get("key", m.get("id", "")) for m in models]
                loaded = [
                    m.get("key", m.get("id", ""))
                    for m in models
                    if m.get("loaded_instances")  # non-empty list = loaded
                ]
                return available, loaded
        except Exception:
            pass

        # Fallback: OpenAI-compat (no load state info)
        try:
            resp = await client.get(f"{base_url}/v1/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                available = [m["id"] for m in models]
                return available, []  # can't determine loaded state
        except Exception:
            pass

    return [], []


async def load_model(
    base_url: str,
    token: str,
    model_id: str,
    timeout: float = 180.0,
) -> bool:
    """Load a single model into VRAM via LM Studio native API."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/api/v1/models/load",
                json={"model": model_id},
            )
            if resp.status_code == 200:
                data = resp.json()
                load_time = data.get("load_time_seconds", "?")
                print(f"  Loaded: {model_id} ({load_time}s)")
                return True
            else:
                print(f"  FAILED: {model_id} -> HTTP {resp.status_code}")
                return False
    except Exception as e:
        print(f"  ERROR: {model_id} -> {e}")
        return False


async def ensure_fleet_loaded(
    models: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Ensure specified models (or always_loaded from config) are in VRAM.

    Returns {model_id: loaded_ok} dict.
    """
    base_url = _resolve_base_url()
    token = _resolve_token()

    # Determine which models to load
    if models is None:
        config = _load_config()
        routing = config.get("model_routing", {})
        models = routing.get("always_loaded", [])
        if not models:
            # Fallback: use the planner and reasoner
            models = [
                routing.get("planner", "agentflow-planner-7b-i1"),
                routing.get("reasoner", "deepseek/deepseek-r1-0528-qwen3-8b"),
            ]

    if not models:
        print("No models configured for auto-loading.")
        return {}

    # Check current state
    print(f"LM Studio: {base_url}")
    available, loaded = await get_loaded_models(base_url, token)

    if not available:
        print("ERROR: Cannot reach LM Studio or no models available.")
        return {m: False for m in models}

    print(f"Available: {len(available)} | Already loaded: {len(loaded)}")

    # Load missing models
    status: Dict[str, bool] = {}
    for model_id in models:
        if model_id in loaded:
            print(f"  Already loaded: {model_id}")
            status[model_id] = True
        elif model_id in available:
            ok = await load_model(base_url, token, model_id)
            status[model_id] = ok
        else:
            print(f"  NOT FOUND: {model_id} (not in LM Studio)")
            status[model_id] = False

    # Summary
    ok_count = sum(1 for v in status.values() if v)
    print(f"\nResult: {ok_count}/{len(status)} models ready in VRAM")

    return status


async def show_status() -> None:
    """Print current model load status."""
    base_url = _resolve_base_url()
    token = _resolve_token()

    print(f"LM Studio: {base_url}")
    available, loaded = await get_loaded_models(base_url, token)

    print(f"Available: {len(available)}")
    print(f"Loaded:    {len(loaded)}")

    if loaded:
        for m in loaded:
            print(f"  VRAM: {m}")
    else:
        print("  (no models in VRAM)")

    if available and not loaded:
        config = _load_config()
        routing = config.get("model_routing", {})
        always = routing.get("always_loaded", [])
        if always:
            print(f"\nConfigured always_loaded: {always}")
            print("Run without --status to auto-load them.")


def main():
    parser = argparse.ArgumentParser(description="BIZRA Node0 Auto Model Loader")
    parser.add_argument("--status", action="store_true", help="Just show status")
    parser.add_argument("--model", type=str, help="Load a specific model")
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    elif args.model:
        result = asyncio.run(ensure_fleet_loaded(models=[args.model]))
        sys.exit(0 if all(result.values()) else 1)
    else:
        result = asyncio.run(ensure_fleet_loaded())
        sys.exit(0 if all(result.values()) else 1)


if __name__ == "__main__":
    main()
