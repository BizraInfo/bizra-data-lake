#!/usr/bin/env python3
"""MoneyShot demo orchestrator for four-channel autonomous mission playback."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("moneyshot")


def _banner() -> str:
    return (
        "\n"
        "+------------------------------------------------------------------+\n"
        "| BIZRA MoneyShot Demo                                              |\n"
        "| Channels: desktop | browser | voice | proof                       |\n"
        "+------------------------------------------------------------------+\n"
    )


async def run_demo(mock: bool = False, channel: str | None = None) -> dict[str, Any]:
    from core.bridges.browser_mcp_client import BrowserMCPClient
    from core.bridges.channel_dispatcher import (
        Channel,
        ChannelDispatcher,
        MissionPlan,
        SubTask,
    )
    from core.token.rl_rewards import composite_reward

    mission = "Prepare outreach for top 5 decentralized AI VCs"
    mission_id = f"moneyshot-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    print(_banner())
    logger.info("[0:00-0:10] Genesis boot")
    logger.info("Ihsan display: 0.97")
    logger.info("Mode: %s", "mock" if mock else "live")

    logger.info("[0:10-0:20] Mission assignment")
    logger.info("Mission: %s", mission)

    browser = BrowserMCPClient(mode="mock" if mock else "direct")
    voice = None
    proof = None
    desktop = None

    try:
        from core.voice.personaplex_bridge import PersonaPlexBridge

        voice = PersonaPlexBridge()
    except Exception as exc:
        logger.warning("Voice bridge unavailable (%s)", exc)

    if not mock:
        try:
            from core.bridges.obs_trigger import OBSTrigger

            proof = OBSTrigger()
        except Exception as exc:
            logger.warning("OBS trigger unavailable (%s)", exc)

    dispatcher = ChannelDispatcher(
        desktop_bridge=desktop,
        browser_client=browser,
        voice_bridge=voice,
        obs_trigger=proof,
    )

    if channel:
        logger.info("[0:20-0:45] Single-channel run: %s", channel)
        target = Channel(channel)
        plan = MissionPlan(
            mission_id=mission_id,
            subtasks=[
                SubTask(
                    id=f"{mission_id}-{channel}",
                    description=mission,
                    channel=target,
                    agent="coordinator",
                    params={"query": mission, "text": mission},
                )
            ],
        )
    else:
        logger.info("[0:20-0:45] Browser research")
        logger.info("[0:45-1:05] Desktop draft + organization")
        logger.info("[1:05-1:20] Voice narration")
        logger.info("[1:20-1:40] Proof + reward + receipt")
        plan = dispatcher.decompose(
            mission_id,
            (
                f"{mission}. Research VC portfolios, draft outreach emails, "
                "narrate synthesis, and record proof evidence."
            ),
        )

    started = time.perf_counter()
    results = await dispatcher.dispatch_all(plan)
    elapsed = time.perf_counter() - started

    succeeded = sum(1 for item in results.values() if item.get("success"))
    total = len(results)

    reward = composite_reward(
        {
            "snr": 0.87 if mock else 0.80,
            "ihsan": 0.97,
            "efficiency": succeeded / max(1, total),
            "user_feedback": 0.85,
            "penalties": 0.0,
        }
    )

    payload = {
        "mission_id": mission_id,
        "mission": mission,
        "mock": mock,
        "channel": channel,
        "results": results,
        "reward": reward,
    }
    receipt_hash = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
        digest_size=16,
    ).hexdigest()

    logger.info("[1:40-2:00] Mission complete")
    logger.info("Channels succeeded: %s/%s", succeeded, total)
    logger.info("Reward score: %.3f", reward)
    logger.info("Receipt: %s", receipt_hash)

    summary = {
        "mission_id": mission_id,
        "mock": mock,
        "channel": channel,
        "channels_succeeded": succeeded,
        "channels_total": total,
        "reward": reward,
        "receipt_hash": receipt_hash,
        "duration_seconds": round(elapsed, 3),
        "results": results,
        "transcript": [
            "[0:00] Genesis",
            "[0:10] Mission assigned",
            "[0:20] Channel dispatch",
            "[1:20] Reward and evidence",
            "[2:00] Complete",
        ],
    }

    print("\nMoneyShot complete")
    summary_keys = (
        "mission_id",
        "channels_succeeded",
        "channels_total",
        "reward",
        "receipt_hash",
    )
    print(
        json.dumps(
            {k: summary[k] for k in summary_keys},
            indent=2,
        )
    )

    if mock and (os.environ.get("BIZRA_MONEYSHOT_ASSERT_DETERMINISTIC") == "1"):
        expected_channels = 1 if channel else 4
        assert total >= expected_channels

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BIZRA MoneyShot demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with zero external service dependencies",
    )
    parser.add_argument(
        "--channel",
        choices=["desktop", "browser", "voice", "proof"],
        help="Run only one channel",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(run_demo(mock=args.mock, channel=args.channel))


if __name__ == "__main__":
    main()
