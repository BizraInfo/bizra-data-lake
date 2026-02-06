#!/usr/bin/env python3
"""
BIZRA CLI Inference Bridge
═══════════════════════════════════════════════════════════════════════════════

Exposes the MultiModelManager as a simple CLI interface for the Rust CLI/TUI.
Uses the existing multi-model router - no duplication.

Usage:
    # Status check
    python cli_bridge.py status

    # List models
    python cli_bridge.py models

    # Chat with auto-routing
    python cli_bridge.py chat "Your question here"

    # Chat with specific purpose
    python cli_bridge.py chat --purpose reasoning "Explain quantum computing"

    # Chat with PAT agent
    python cli_bridge.py agent guardian "What should I consider?"

    # Start HTTP server for Rust CLI
    python cli_bridge.py serve --port 8765

Standing on Giants:
- Shazeer (2017): MoE routing
- BIZRA MultiModelManager: Existing infrastructure
"""

import argparse
import asyncio
import json
import sys
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit('/core/', 1)[0])

from core.inference.multi_model_manager import (
    MultiModelManager,
    ModelPurpose,
    get_multi_model_manager,
)


# PAT Agent system prompts (same as Rust version)
PAT_PROMPTS = {
    "guardian": """You are the Guardian of a Personal Agentic Team (PAT).
Your role is ethical oversight and protective guidance.
Standing on the shoulders of: Al-Ghazali, John Rawls, Anthropic.
Focus on beneficial outcomes, ethical considerations, and harm prevention.
Apply the FATE gates: Ihsān (excellence ≥0.95), Adl (fairness), Harm (≤0.30), Confidence (≥0.80).
Be wise, protective, and guide toward beneficial outcomes.""",

    "strategist": """You are the Strategist of a Personal Agentic Team (PAT).
Your role is strategic planning and long-term thinking.
Standing on the shoulders of: Sun Tzu, Clausewitz, Michael Porter.
Focus on objectives, competitive advantage, and strategic positioning.
Be concise but thorough. Think in terms of goals, resources, and outcomes.""",

    "researcher": """You are the Researcher of a Personal Agentic Team (PAT).
Your role is knowledge discovery and synthesis.
Standing on the shoulders of: Claude Shannon, Alan Turing, Edsger Dijkstra.
Focus on finding accurate information, synthesizing knowledge, and providing insights.
Be thorough in research but clear in presentation.""",

    "developer": """You are the Developer of a Personal Agentic Team (PAT).
Your role is code implementation and technical solutions.
Standing on the shoulders of: Donald Knuth, Dennis Ritchie, Linus Torvalds.
Focus on clean code, efficient algorithms, and robust implementation.
Be precise, practical, and security-conscious.""",

    "analyst": """You are the Analyst of a Personal Agentic Team (PAT).
Your role is data analysis and insight extraction.
Standing on the shoulders of: John Tukey, Edward Tufte, William Cleveland.
Focus on patterns, trends, and data-driven insights.
Be quantitative, visual, and clear in presenting findings.""",

    "reviewer": """You are the Reviewer of a Personal Agentic Team (PAT).
Your role is quality validation and constructive feedback.
Standing on the shoulders of: Michael Fagan, David Parnas, Fred Brooks.
Focus on correctness, completeness, and improvement opportunities.
Be thorough but constructive. Identify issues and suggest solutions.""",

    "executor": """You are the Executor of a Personal Agentic Team (PAT).
Your role is task execution and delivery.
Standing on the shoulders of: Toyota Production System, W. Edwards Deming, Taiichi Ohno.
Focus on efficient execution, continuous improvement, and delivering results.
Be action-oriented, efficient, and results-focused.""",
}


async def cmd_status():
    """Show system status."""
    try:
        manager = await get_multi_model_manager()
        status = manager.get_status()
        pool = status.get("connection_pool", {})

        result = {
            "status": "connected",
            "total_models": status["total_models"],
            "loaded_models": status["loaded_models"],
            "loaded_list": status["loaded_list"],
            "pool_healthy": pool.get("is_healthy", False),
            "success_rate": pool.get("success_rate_pct", 0),
            "avg_latency_ms": pool.get("avg_latency_ms", 0),
        }
        print(json.dumps(result))
        return 0

    except Exception as e:
        print(json.dumps({"status": "disconnected", "error": str(e)}))
        return 1


async def cmd_models(purpose: Optional[str] = None):
    """List available models."""
    try:
        manager = await get_multi_model_manager()

        if purpose:
            try:
                p = ModelPurpose(purpose.lower())
                models = manager.list_models(p)
            except ValueError:
                print(json.dumps({"error": f"Invalid purpose: {purpose}"}))
                return 1
        else:
            models = manager.list_models()

        result = []
        for m in models:
            result.append({
                "id": m.id,
                "name": m.name,
                "purposes": [p.value for p in m.purposes],
                "params_b": m.params_b,
                "loaded": m.is_loaded,
                "supports_vision": m.supports_vision,
                "supports_tools": m.supports_tools,
            })

        print(json.dumps({"models": result}))
        return 0

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1


async def cmd_chat(message: str, purpose: str = "general", system_prompt: Optional[str] = None):
    """Chat with auto-routed model."""
    try:
        manager = await get_multi_model_manager()

        # Map purpose string to enum
        try:
            p = ModelPurpose(purpose.lower())
        except ValueError:
            p = ModelPurpose.GENERAL

        response = await manager.chat(
            message,
            purpose=p,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=2048
        )

        print(json.dumps(response))
        return 0 if "error" not in response else 1

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1


async def cmd_agent(agent: str, message: str):
    """Chat with PAT agent."""
    agent = agent.lower()
    if agent not in PAT_PROMPTS:
        print(json.dumps({"error": f"Unknown agent: {agent}. Available: {list(PAT_PROMPTS.keys())}"}))
        return 1

    system_prompt = PAT_PROMPTS[agent]

    # Map agent to purpose
    purpose_map = {
        "guardian": ModelPurpose.REASONING,
        "strategist": ModelPurpose.REASONING,
        "researcher": ModelPurpose.REASONING,
        "developer": ModelPurpose.AGENTIC,
        "analyst": ModelPurpose.REASONING,
        "reviewer": ModelPurpose.REASONING,
        "executor": ModelPurpose.AGENTIC,
    }
    purpose = purpose_map.get(agent, ModelPurpose.GENERAL)

    return await cmd_chat(message, purpose.value, system_prompt)


async def cmd_serve(port: int = 8765):
    """Start HTTP server for Rust CLI."""
    try:
        from aiohttp import web
    except ImportError:
        print(json.dumps({"error": "aiohttp not installed. Run: pip install aiohttp"}))
        return 1

    manager = await get_multi_model_manager()

    async def handle_status(request):
        status = manager.get_status()
        return web.json_response(status)

    async def handle_models(request):
        purpose = request.query.get("purpose")
        models = manager.list_models(ModelPurpose(purpose) if purpose else None)
        return web.json_response({
            "models": [{"id": m.id, "name": m.name, "loaded": m.is_loaded} for m in models]
        })

    async def handle_chat(request):
        data = await request.json()
        message = data.get("message", "")
        purpose = data.get("purpose", "general")
        agent = data.get("agent")

        system_prompt = PAT_PROMPTS.get(agent) if agent else data.get("system_prompt")

        try:
            p = ModelPurpose(purpose.lower())
        except ValueError:
            p = ModelPurpose.GENERAL

        response = await manager.chat(
            message,
            purpose=p,
            system_prompt=system_prompt,
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2048)
        )
        return web.json_response(response)

    async def handle_health(request):
        return web.json_response({"status": "ok", "healthy": manager.is_pool_healthy()})

    app = web.Application()
    app.router.add_get("/status", handle_status)
    app.router.add_get("/models", handle_models)
    app.router.add_post("/chat", handle_chat)
    app.router.add_get("/health", handle_health)

    print(f"Starting BIZRA inference bridge on port {port}...")
    print(f"Endpoints:")
    print(f"  GET  /health - Health check")
    print(f"  GET  /status - System status")
    print(f"  GET  /models - List models")
    print(f"  POST /chat   - Chat ({{message, purpose?, agent?, system_prompt?}})")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await runner.cleanup()
        await manager.close()

    return 0


def main():
    parser = argparse.ArgumentParser(description="BIZRA CLI Inference Bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status
    subparsers.add_parser("status", help="Show system status")

    # Models
    models_parser = subparsers.add_parser("models", help="List models")
    models_parser.add_argument("--purpose", "-p", help="Filter by purpose")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Chat with model")
    chat_parser.add_argument("message", help="Message to send")
    chat_parser.add_argument("--purpose", "-p", default="general", help="Model purpose")
    chat_parser.add_argument("--system", "-s", help="System prompt")

    # Agent
    agent_parser = subparsers.add_parser("agent", help="Chat with PAT agent")
    agent_parser.add_argument("agent", choices=list(PAT_PROMPTS.keys()), help="Agent name")
    agent_parser.add_argument("message", help="Message to send")

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Start HTTP server")
    serve_parser.add_argument("--port", "-p", type=int, default=8765, help="Port number")

    args = parser.parse_args()

    if args.command == "status":
        return asyncio.run(cmd_status())
    elif args.command == "models":
        return asyncio.run(cmd_models(args.purpose))
    elif args.command == "chat":
        return asyncio.run(cmd_chat(args.message, args.purpose, args.system))
    elif args.command == "agent":
        return asyncio.run(cmd_agent(args.agent, args.message))
    elif args.command == "serve":
        return asyncio.run(cmd_serve(args.port))


if __name__ == "__main__":
    sys.exit(main())
