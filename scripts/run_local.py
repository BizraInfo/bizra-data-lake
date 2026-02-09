#!/usr/bin/env python3
"""
BIZRA Local Runner — Run with your local LLM backend

Usage (from PowerShell/CMD on Windows):
  set LM_API_TOKEN=your-token-here
  python scripts/run_local.py query "Your question"

Usage (from WSL/Linux):
  export LM_API_TOKEN="your-token-here"
  python scripts/run_local.py query "Your question"

Or pass token directly:
  python scripts/run_local.py --token YOUR_TOKEN query "Your question"
"""

import argparse
import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="BIZRA Local Runner")
    parser.add_argument("--token", help="LM Studio API token")
    parser.add_argument("--url", default="http://192.168.56.1:1234", help="LM Studio URL")
    parser.add_argument("--model", default="liquid/lfm2.5-1.2b", help="Model name")
    
    subparsers = parser.add_subparsers(dest="command")
    
    p_query = subparsers.add_parser("query", help="Run a query")
    p_query.add_argument("text", help="Query text")
    
    p_test = subparsers.add_parser("test", help="Test LM Studio connection")
    
    p_pat = subparsers.add_parser("pat", help="Run PAT team")
    p_pat.add_argument("--quick", action="store_true")
    
    args = parser.parse_args()
    
    # Set environment from args
    if args.token:
        os.environ["LM_API_TOKEN"] = args.token
    if args.url:
        os.environ["LM_STUDIO_URL"] = args.url
    if args.model:
        os.environ["BIZRA_MODEL"] = args.model
    
    if args.command == "test":
        asyncio.run(test_connection(args))
    elif args.command == "query":
        asyncio.run(run_query(args))
    elif args.command == "pat":
        run_pat(args)
    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START:")
        print("=" * 60)
        print("""
1. Get your LM Studio API token from LM Studio settings

2. Test connection:
   python scripts/run_local.py --token YOUR_TOKEN test

3. Run a query:
   python scripts/run_local.py --token YOUR_TOKEN query "What is consciousness?"

4. Run PAT team:
   python scripts/run_local.py --token YOUR_TOKEN pat --quick
        """)


async def test_connection(args):
    """Test LM Studio connection."""
    import httpx
    
    url = os.getenv("LM_STUDIO_URL", "http://192.168.56.1:1234")
    token = os.getenv("LM_API_TOKEN", "")
    
    print(f"Testing connection to: {url}")
    print(f"Token: {'*' * 8}..." if token else "NO TOKEN SET!")
    
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        try:
            # Test models endpoint
            resp = await client.get(f"{url}/v1/models")
            if resp.status_code == 200:
                models = resp.json()
                print("\n✓ Connection successful!")
                print(f"  Available models: {len(models.get('data', []))}")
                for m in models.get("data", [])[:5]:
                    print(f"    - {m.get('id', 'unknown')}")
            else:
                print(f"\n✗ Error: {resp.status_code}")
                print(resp.text)
        except Exception as e:
            print(f"\n✗ Connection failed: {e}")
            print("\nTroubleshooting:")
            print("  1. Is LM Studio running?")
            print("  2. Is a model loaded?")
            print("  3. Check Settings > Developer > Enable API Server")
            print("  4. Check your API token")


async def run_query(args):
    """Run a query through Sovereign Command Center."""
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "sovereign_command",
        os.path.join(PROJECT_ROOT, "core/command/sovereign_command.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sovereign_command"] = module
    spec.loader.exec_module(module)
    
    center = module.SovereignCommandCenter()
    result = await center.execute(args.text, verbose=True)
    
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(result.response)
    print("=" * 60)
    print(f"\nSNR: {result.snr_score:.4f} | Ihsān: {'✓' if result.ihsan_compliant else '✗'}")
    print(f"Backend: {result.backend_used} | Latency: {result.latency_ms:.0f}ms")


def run_pat(args):
    """Run PAT team evaluation."""
    import subprocess
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts/pat_evaluator.py")]
    if args.quick:
        cmd.append("--quick")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
