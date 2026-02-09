#!/usr/bin/env python3
"""
BIZRA Local System Runner — Multi-Model Orchestration

Usage:
    python scripts/bizra_local.py status          # Check system status
    python scripts/bizra_local.py chat "question" # Chat with auto-routing
    python scripts/bizra_local.py reason "query"  # Use DeepSeek R1 reasoning
    python scripts/bizra_local.py plan "task"     # Use AgentFlow for planning
    python scripts/bizra_local.py pat             # Run PAT team
    python scripts/bizra_local.py vision "prompt" # Vision model (needs image)

Environment:
    LM_API_TOKEN=your-token  (or use --token)
"""

import argparse
import asyncio
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


async def cmd_status(args):
    """Show system status and available models."""
    from core.inference.multi_model_manager import get_multi_model_manager

    mgr = await get_multi_model_manager()
    models = mgr.list_models()  # Returns ModelProfile objects
    
    print("═" * 60)
    print("BIZRA LOCAL SYSTEM STATUS")
    print("═" * 60)
    print(f"  LM Studio: http://192.168.56.1:1234")
    print(f"  Models Available: {len(models)}")
    print()
    
    loaded = [m for m in models if m.is_loaded]
    print(f"LOADED MODELS ({len(loaded)}):")
    for m in loaded:
        purposes = ", ".join(p.value for p in m.purposes)
        print(f"  ✓ {m.id} ({m.params_b}B) - {purposes}")
    
    print()
    print("ALL MODELS:")
    for m in models:
        status = "✓" if m.is_loaded else "○"
        print(f"  {status} {m.id}")
    print("═" * 60)


async def cmd_chat(args):
    """Chat with auto-routing to best model."""
    from core.inference.multi_model_manager import get_multi_model_manager, ModelPurpose

    mgr = await get_multi_model_manager()
    
    purpose = ModelPurpose.GENERAL
    if args.purpose:
        purpose = ModelPurpose(args.purpose)
    
    print(f"Routing to: {purpose.value}")
    result = await mgr.chat(args.query, purpose=purpose, max_tokens=args.max_tokens)
    
    print("\n" + "═" * 60)
    print(f"Model: {result.get('model', 'unknown')}")
    print("═" * 60)
    
    # Handle reasoning models with separate thinking
    if result.get('raw_content') and result['raw_content'] != result.get('content'):
        print("[Thinking hidden - use --show-thinking to display]")
    
    print(result.get('content', result.get('error', 'No response')))
    print("═" * 60)
    
    if result.get('usage'):
        u = result['usage']
        print(f"Tokens: {u.get('prompt_tokens', 0)} in / {u.get('completion_tokens', 0)} out")


async def cmd_reason(args):
    """Use DeepSeek R1 for deep reasoning."""
    import httpx
    
    token = os.getenv("LM_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    print("═" * 60)
    print("DEEPSEEK R1 REASONING")
    print("═" * 60)
    print(f"Query: {args.query[:60]}...")
    print("(Processing... deep reasoning may take 30-120 seconds)")
    print()
    
    async with httpx.AsyncClient(headers=headers, timeout=300.0) as client:
        resp = await client.post("http://192.168.56.1:1234/v1/chat/completions", json={
            "model": "deepseek/deepseek-r1-0528-qwen3-8b",
            "messages": [{"role": "user", "content": args.query}],
            "max_tokens": args.max_tokens,
        })
        data = resp.json()
        
        if "choices" in data:
            msg = data["choices"][0]["message"]
            
            if args.show_thinking and msg.get("reasoning_content"):
                print("─" * 40)
                print("THINKING:")
                print("─" * 40)
                print(msg["reasoning_content"])
                print()
            
            print("─" * 40)
            print("ANSWER:")
            print("─" * 40)
            print(msg.get("content", "[No content]"))
            
            if data.get("usage"):
                u = data["usage"]
                print()
                print(f"Tokens: {u['prompt_tokens']} in / {u['completion_tokens']} out")
        else:
            print(f"Error: {data}")


async def cmd_plan(args):
    """Use AgentFlow for task planning."""
    import httpx
    
    token = os.getenv("LM_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    system_prompt = """You are AgentFlow, an expert task planner.
Break down the task into clear, actionable steps.
Format: Numbered list with brief descriptions.
Focus on practical execution."""

    print("═" * 60)
    print("AGENTFLOW PLANNER")
    print("═" * 60)
    print(f"Task: {args.task[:60]}...")
    print()
    
    async with httpx.AsyncClient(headers=headers, timeout=120.0) as client:
        resp = await client.post("http://192.168.56.1:1234/v1/chat/completions", json={
            "model": "agentflow-planner-7b-i1",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Plan this task: {args.task}"},
            ],
            "max_tokens": args.max_tokens,
        })
        data = resp.json()
        
        if "choices" in data:
            print("PLAN:")
            print("─" * 40)
            print(data["choices"][0]["message"]["content"])
        else:
            print(f"Error: {data}")


async def cmd_pat(args):
    """Run PAT team evaluation."""
    import subprocess
    
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts/pat_evaluator.py")]
    if args.quick:
        cmd.append("--quick")
    else:
        cmd.append("--full")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Local System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--token", help="LM Studio API token")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Status
    subparsers.add_parser("status", help="Show system status")
    
    # Chat
    p_chat = subparsers.add_parser("chat", help="Chat with auto-routing")
    p_chat.add_argument("query", help="Your message")
    p_chat.add_argument("--purpose", choices=["reasoning", "vision", "agentic", "general", "nano"], default="general")
    p_chat.add_argument("--max-tokens", type=int, default=1024)
    
    # Reason
    p_reason = subparsers.add_parser("reason", help="Deep reasoning with DeepSeek R1")
    p_reason.add_argument("query", help="Your question")
    p_reason.add_argument("--show-thinking", action="store_true", help="Show chain-of-thought")
    p_reason.add_argument("--max-tokens", type=int, default=2048)
    
    # Plan
    p_plan = subparsers.add_parser("plan", help="Task planning with AgentFlow")
    p_plan.add_argument("task", help="Task to plan")
    p_plan.add_argument("--max-tokens", type=int, default=1024)
    
    # PAT
    p_pat = subparsers.add_parser("pat", help="Run PAT team")
    p_pat.add_argument("--quick", action="store_true")
    
    args = parser.parse_args()
    
    if args.token:
        os.environ["LM_API_TOKEN"] = args.token
    
    if not args.command:
        parser.print_help()
        print("\n" + "═" * 50)
        print("QUICK START:")
        print("  export LM_API_TOKEN='your-token'")
        print("  python scripts/bizra_local.py status")
        print("  python scripts/bizra_local.py reason 'What is consciousness?'")
        return
    
    commands = {
        "status": cmd_status,
        "chat": cmd_chat,
        "reason": cmd_reason,
        "plan": cmd_plan,
        "pat": cmd_pat,
    }
    
    asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    main()
