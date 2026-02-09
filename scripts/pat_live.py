#!/usr/bin/env python3
"""
BIZRA PAT Team — Live Local Execution

Runs actual PAT agents against your local LLM (DeepSeek R1).
Each agent has a specialized role and system prompt.

Usage:
    export LM_API_TOKEN="your-token"
    python scripts/pat_live.py "Design a secure authentication system"
"""

import argparse
import asyncio
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# PAT Agent Definitions
PAT_AGENTS = {
    "strategist": {
        "name": "Strategist",
        "system": """You are the PAT Strategist. Your role is strategic planning.
Standing on Giants: Sun Tzu, John Boyd (OODA), Michael Porter.
Analyze the task and provide a high-level strategic approach.
Be concise (2-3 paragraphs max). Focus on goals, risks, and key decisions.""",
    },
    "researcher": {
        "name": "Researcher", 
        "system": """You are the PAT Researcher. Your role is deep investigation.
Standing on Giants: Vannevar Bush, Claude Shannon, Douglas Engelbart.
Research the technical aspects and provide evidence-based insights.
Be concise (2-3 paragraphs max). Focus on facts, patterns, and sources.""",
    },
    "analyst": {
        "name": "Analyst",
        "system": """You are the PAT Analyst. Your role is systematic analysis.
Standing on Giants: Herbert Simon, Daniel Kahneman, Judea Pearl.
Analyze the problem structure and identify key variables.
Be concise (2-3 paragraphs max). Focus on data, models, and implications.""",
    },
    "guardian": {
        "name": "Guardian",
        "system": """You are the PAT Guardian. Your role is ethical oversight and security.
Standing on Giants: Al-Ghazali (Ihsān), John Rawls, Anthropic.
Evaluate risks, ethical implications, and security concerns.
Be concise (2-3 paragraphs max). Focus on harm prevention and alignment.""",
    },
    "coordinator": {
        "name": "Coordinator",
        "system": """You are the PAT Coordinator. Your role is synthesis and integration.
Standing on Giants: Norbert Wiener, W. Edwards Deming, Peter Senge.
Synthesize the team's insights into a coherent action plan.
Be concise (2-3 paragraphs max). Focus on integration and next steps.""",
    },
}


async def run_agent(agent_id: str, task: str, token: str) -> dict:
    """Run a single PAT agent."""
    import httpx
    
    agent = PAT_AGENTS[agent_id]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    start = time.perf_counter()
    
    async with httpx.AsyncClient(headers=headers, timeout=180.0) as client:
        resp = await client.post("http://192.168.56.1:1234/v1/chat/completions", json={
            "model": "deepseek/deepseek-r1-0528-qwen3-8b",
            "messages": [
                {"role": "system", "content": agent["system"]},
                {"role": "user", "content": f"Task: {task}"},
            ],
            "max_tokens": 800,
        })
        
        elapsed = (time.perf_counter() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "")
            thinking = data["choices"][0]["message"].get("reasoning_content", "")
            usage = data.get("usage", {})
            
            return {
                "agent": agent_id,
                "name": agent["name"],
                "content": content,
                "thinking": thinking,
                "tokens": usage.get("total_tokens", 0),
                "latency_ms": elapsed,
                "success": True,
            }
        else:
            return {
                "agent": agent_id,
                "name": agent["name"],
                "content": f"Error: {resp.status_code}",
                "success": False,
            }


async def run_pat_team(task: str, token: str, agents: list = None):
    """Run the full PAT team on a task."""
    
    if agents is None:
        agents = ["strategist", "researcher", "analyst", "guardian", "coordinator"]
    
    print("═" * 70)
    print("PAT TEAM — LIVE LOCAL EXECUTION")
    print("═" * 70)
    print(f"Task: {task[:60]}...")
    print(f"Model: DeepSeek R1 8B (reasoning)")
    print(f"Agents: {', '.join(agents)}")
    print("═" * 70)
    print()
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for agent_id in agents:
        print(f"🤖 Running {PAT_AGENTS[agent_id]['name']}...")
        result = await run_agent(agent_id, task, token)
        results.append(result)
        
        if result["success"]:
            total_tokens += result.get("tokens", 0)
            total_time += result.get("latency_ms", 0)
            
            print(f"   ✓ Complete ({result['latency_ms']:.0f}ms, {result.get('tokens', 0)} tokens)")
        else:
            print(f"   ✗ Failed: {result['content']}")
    
    print()
    print("═" * 70)
    print("PAT TEAM RESPONSES")
    print("═" * 70)
    
    for result in results:
        if result["success"]:
            print()
            print(f"┌─ {result['name'].upper()} {'─' * (50 - len(result['name']))}")
            print(f"│")
            for line in result["content"].split("\n"):
                print(f"│  {line}")
            print(f"│")
            print(f"└─ ({result['latency_ms']:.0f}ms)")
    
    print()
    print("═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print(f"  Agents Run:    {len(results)}")
    print(f"  Total Tokens:  {total_tokens}")
    print(f"  Total Time:    {total_time/1000:.1f}s")
    print(f"  Avg per Agent: {total_time/len(results)/1000:.1f}s")
    
    # Calculate SNR (simplified)
    successful = sum(1 for r in results if r["success"])
    snr = successful / len(results)
    ihsan_pass = snr >= 0.95
    
    print(f"  Success Rate:  {snr*100:.0f}%")
    print(f"  Ihsān Status:  {'✓ PASS' if ihsan_pass else '⚠ NEEDS IMPROVEMENT'}")
    print("═" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="PAT Team Live Execution")
    parser.add_argument("task", nargs="?", default="Analyze the best approach for building a sovereign AI system", 
                       help="Task for the PAT team")
    parser.add_argument("--agents", nargs="+", 
                       choices=["strategist", "researcher", "analyst", "guardian", "coordinator"],
                       help="Specific agents to run")
    parser.add_argument("--token", help="LM Studio API token")
    
    args = parser.parse_args()
    
    token = args.token or os.getenv("LM_API_TOKEN", "")
    if not token:
        print("ERROR: Set LM_API_TOKEN or use --token")
        return
    
    asyncio.run(run_pat_team(args.task, token, args.agents))


if __name__ == "__main__":
    main()
