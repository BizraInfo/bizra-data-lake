"""
BIZRA Sovereign Gateway Launcher

Bridges the sovereign runtime to messaging platforms.
This module connects a ChannelAdapter to SovereignRuntime.query()
to create a full end-to-end messaging pipeline.

Usage:
    python -m core.sovereign gateway telegram
    python -m core.sovereign gateway telegram --webhook https://example.com/hook

Standing on Giants: Hexagonal Architecture (Cockburn) + Constitutional AI (Anthropic)
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def _make_query_fn():
    """
    Create a query function backed by SovereignRuntime.

    Returns a coroutine that accepts (content, context) and returns
    the response string.
    """
    from ..sovereign.runtime import RuntimeConfig, SovereignRuntime

    config = RuntimeConfig(autonomous_enabled=False)
    runtime = await SovereignRuntime.create(config).__aenter__()

    async def query_fn(content: str, context: Optional[Dict[str, Any]] = None) -> str:
        ctx = context or {}
        ctx.setdefault("source", "gateway")
        result = await runtime.query(content, context=ctx)
        if result.success:
            return result.response
        return f"Error: {result.error}"

    return query_fn, runtime


async def run_telegram_gateway(
    token: str = "",
    node_id: str = "",
    webhook_url: str = "",
) -> None:
    """
    Run the Telegram gateway.

    Args:
        token: Telegram bot token (or BIZRA_TELEGRAM_TOKEN env)
        node_id: Node ID to display in status
        webhook_url: If set, use webhook mode instead of polling
    """
    from .adapters.telegram import TelegramAdapter

    # Load node_id from credentials if not provided
    if not node_id:
        try:
            from .onboarding import get_node_credentials

            creds = get_node_credentials()
            if creds:
                node_id = creds.node_id
        except (ImportError, FileNotFoundError):
            pass  # Module not available or no credentials file
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load credentials for gateway: {e}")

    print("\n  Starting BIZRA Telegram Gateway...")
    if node_id:
        print(f"  Node: {node_id}")

    # Create query function backed by sovereign runtime
    query_fn, runtime = await _make_query_fn()

    try:
        adapter = TelegramAdapter(
            token=token,
            query_fn=query_fn,
            node_id=node_id,
            webhook_url=webhook_url,
        )

        print(f"  Mode: {'webhook' if webhook_url else 'long-polling'}")
        print("  Connecting to Telegram...\n")

        await adapter.start()

    except KeyboardInterrupt:
        print("\n\n  Shutting down...")
        await adapter.stop()
    except ConnectionError as e:
        print(f"\n  Connection error: {e}")
    finally:
        await runtime.__aexit__(None, None, None)
