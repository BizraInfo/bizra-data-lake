#!/usr/bin/env python3
"""
BIZRA CLI Inference Bridge (Standalone)
═══════════════════════════════════════════════════════════════════════════════

Standalone bridge for the Rust CLI - uses LM Studio's v1 API directly.
Avoids core module import issues.

Usage:
    python bizra_cli_bridge.py status
    python bizra_cli_bridge.py models
    python bizra_cli_bridge.py chat "Your question"
    python bizra_cli_bridge.py agent guardian "What should I consider?"
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    print(json.dumps({"error": "httpx not installed. Run: pip install httpx"}))
    sys.exit(1)


# Configuration
LM_STUDIO_HOST = os.environ.get("LM_STUDIO_HOST", "192.168.56.1")
LM_STUDIO_PORT = int(os.environ.get("LM_STUDIO_PORT", "1234"))
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "")  # Optional auth
BASE_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}"


class ModelPurpose(str, Enum):
    REASONING = "reasoning"
    VISION = "vision"
    AGENTIC = "agentic"
    EMBEDDING = "embedding"
    GENERAL = "general"
    NANO = "nano"


# PAT Agent system prompts
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
Be concise but thorough.""",

    "researcher": """You are the Researcher of a Personal Agentic Team (PAT).
Your role is knowledge discovery and synthesis.
Standing on the shoulders of: Claude Shannon, Alan Turing, Edsger Dijkstra.
Focus on finding accurate information, synthesizing knowledge, and providing insights.""",

    "developer": """You are the Developer of a Personal Agentic Team (PAT).
Your role is code implementation and technical solutions.
Standing on the shoulders of: Donald Knuth, Dennis Ritchie, Linus Torvalds.
Focus on clean code, efficient algorithms, and robust implementation.""",

    "analyst": """You are the Analyst of a Personal Agentic Team (PAT).
Your role is data analysis and insight extraction.
Standing on the shoulders of: John Tukey, Edward Tufte, William Cleveland.
Focus on patterns, trends, and data-driven insights.""",

    "reviewer": """You are the Reviewer of a Personal Agentic Team (PAT).
Your role is quality validation and constructive feedback.
Standing on the shoulders of: Michael Fagan, David Parnas, Fred Brooks.
Focus on correctness, completeness, and improvement opportunities.""",

    "executor": """You are the Executor of a Personal Agentic Team (PAT).
Your role is task execution and delivery.
Standing on the shoulders of: Toyota Production System, W. Edwards Deming, Taiichi Ohno.
Focus on efficient execution, continuous improvement, and delivering results.""",
}


@dataclass
class ModelInfo:
    id: str
    name: str
    loaded: bool
    params_b: float = 0.0
    context_length: int = 4096


class LMStudioBridge:
    """Simple LM Studio bridge using v1 API."""

    def __init__(self):
        headers = {}
        if LM_STUDIO_API_KEY:
            headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(120.0, connect=5.0),
            headers=headers
        )

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/api/v1/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[ModelInfo]:
        try:
            resp = await self.client.get("/api/v1/models")
            data = resp.json()
            models = []

            # Handle both formats: {"models": [...]} and {"data": [...]}
            model_list = data.get("models", data.get("data", []))

            for m in model_list:
                loaded_instances = m.get("loaded_instances", [])
                loaded = len(loaded_instances) > 0

                # Parse params (handle "8B", "8x3.6B", etc.)
                params_str = m.get("params_string") or "0B"
                params_str = params_str.upper().replace("B", "").replace("X", "*")
                try:
                    if "*" in params_str:
                        # Handle MoE format like "8x3.6B"
                        parts = params_str.split("*")
                        params = float(parts[0]) * float(parts[1])
                    else:
                        params = float(params_str) if params_str else 0.0
                except:
                    params = 0.0

                models.append(ModelInfo(
                    id=m.get("key", m.get("id", "")),
                    name=m.get("display_name", m.get("key", m.get("id", ""))),
                    loaded=loaded,
                    params_b=params,
                    context_length=m.get("max_context_length", 4096)
                ))

            return models
        except Exception as e:
            return []

    async def get_loaded_model(self) -> Optional[str]:
        models = await self.list_models()
        for m in models:
            if m.loaded:
                return m.id
        return None

    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Chat using OpenAI-compatible endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        if model:
            payload["model"] = model

        try:
            resp = await self.client.post("/v1/chat/completions", json=payload)
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text}"}

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Strip <think>...</think> tokens from reasoning models
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            return {
                "content": content,
                "model": data.get("model", "unknown"),
                "usage": data.get("usage", {})
            }
        except httpx.ConnectError:
            return {"error": f"Cannot connect to LM Studio at {BASE_URL}"}
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return {
                    "error": "LM Studio requires authentication. Set LM_STUDIO_API_KEY environment variable or disable auth in LM Studio settings."
                }
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}

    async def close(self):
        await self.client.aclose()


async def cmd_status():
    """Show system status."""
    bridge = LMStudioBridge()
    try:
        healthy = await bridge.health_check()
        if healthy:
            models = await bridge.list_models()
            loaded = [m for m in models if m.loaded]
            result = {
                "status": "connected",
                "host": f"{LM_STUDIO_HOST}:{LM_STUDIO_PORT}",
                "total_models": len(models),
                "loaded_models": len(loaded),
                "loaded_list": [m.name for m in loaded]
            }
        else:
            result = {
                "status": "disconnected",
                "host": f"{LM_STUDIO_HOST}:{LM_STUDIO_PORT}",
                "error": "LM Studio not responding"
            }
        print(json.dumps(result, indent=2))
        return 0 if healthy else 1
    finally:
        await bridge.close()


async def cmd_models():
    """List available models."""
    bridge = LMStudioBridge()
    try:
        models = await bridge.list_models()
        if not models:
            print(json.dumps({"error": "No models found or LM Studio not running"}))
            return 1

        result = []
        for m in models:
            result.append({
                "id": m.id,
                "name": m.name,
                "loaded": m.loaded,
                "params_b": m.params_b
            })
        print(json.dumps({"models": result}, indent=2))
        return 0
    finally:
        await bridge.close()


async def cmd_chat(message: str, system_prompt: Optional[str] = None):
    """Chat with model."""
    bridge = LMStudioBridge()
    try:
        response = await bridge.chat(message, system_prompt=system_prompt)
        print(json.dumps(response, indent=2))
        return 0 if "error" not in response else 1
    finally:
        await bridge.close()


async def cmd_agent(agent: str, message: str):
    """Chat with PAT agent."""
    agent = agent.lower()
    if agent not in PAT_PROMPTS:
        print(json.dumps({"error": f"Unknown agent: {agent}", "available": list(PAT_PROMPTS.keys())}))
        return 1

    return await cmd_chat(message, system_prompt=PAT_PROMPTS[agent])


async def cmd_quick(message: str):
    """Quick chat - just returns content."""
    bridge = LMStudioBridge()
    try:
        response = await bridge.chat(message)
        if "error" in response:
            print(f"Error: {response['error']}", file=sys.stderr)
            return 1
        print(response.get("content", ""))
        return 0
    finally:
        await bridge.close()


def main():
    parser = argparse.ArgumentParser(
        description="BIZRA CLI Inference Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                     # Check LM Studio connection
  %(prog)s models                     # List available models
  %(prog)s chat "Hello world"         # Simple chat
  %(prog)s agent guardian "Help me"   # Chat with PAT agent
  %(prog)s quick "Say hello"          # Quick chat (content only)
"""
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status
    subparsers.add_parser("status", help="Show LM Studio status")

    # Models
    subparsers.add_parser("models", help="List available models")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Chat with model")
    chat_parser.add_argument("message", help="Message to send")
    chat_parser.add_argument("--system", "-s", help="System prompt")

    # Agent
    agent_parser = subparsers.add_parser("agent", help="Chat with PAT agent")
    agent_parser.add_argument("agent", choices=list(PAT_PROMPTS.keys()), help="Agent name")
    agent_parser.add_argument("message", help="Message to send")

    # Quick (content only output)
    quick_parser = subparsers.add_parser("quick", help="Quick chat (content only)")
    quick_parser.add_argument("message", help="Message to send")

    args = parser.parse_args()

    if args.command == "status":
        return asyncio.run(cmd_status())
    elif args.command == "models":
        return asyncio.run(cmd_models())
    elif args.command == "chat":
        return asyncio.run(cmd_chat(args.message, args.system))
    elif args.command == "agent":
        return asyncio.run(cmd_agent(args.agent, args.message))
    elif args.command == "quick":
        return asyncio.run(cmd_quick(args.message))


if __name__ == "__main__":
    sys.exit(main())
