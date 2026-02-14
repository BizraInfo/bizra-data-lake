#!/usr/bin/env python3
"""
Local Model Delegation — Reduce Claude API Token Usage

Routes tasks to local LM Studio models instead of Claude API.
Use for: code generation, research, analysis, documentation.

Usage:
    python scripts/local_delegate.py "Your task here"
    python scripts/local_delegate.py --reasoning "Complex problem"
    python scripts/local_delegate.py --code "Write a function"
"""

import asyncio
import sys
import httpx
from typing import Optional

LM_STUDIO_URL = "http://192.168.56.1:1234/v1"

MODELS = {
    "reasoning": "deepseek/deepseek-r1-0528-qwen3-8b",
    "code": "deepseek/deepseek-r1-0528-qwen3-8b",
    "general": "llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b",
    "fast": "chuanli11_-_llama-3.2-3b-instruct-uncensored",
}

async def delegate(prompt: str, task_type: str = "general", max_tokens: int = 4096) -> str:
    """Delegate task to local model."""
    model = MODELS.get(task_type, MODELS["general"])

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1 if task_type == "reasoning" else 0.7,
            }
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

async def main():
    if len(sys.argv) < 2:
        print("Usage: python local_delegate.py [--reasoning|--code] 'prompt'")
        return

    task_type = "general"
    prompt_start = 1

    if sys.argv[1] == "--reasoning":
        task_type = "reasoning"
        prompt_start = 2
    elif sys.argv[1] == "--code":
        task_type = "code"
        prompt_start = 2

    prompt = " ".join(sys.argv[prompt_start:])
    print(f"Delegating to local {task_type} model...")
    result = await delegate(prompt, task_type)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
