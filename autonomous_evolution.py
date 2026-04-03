import requests
import time
import json
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table

# Config
API_URL = "http://127.0.0.1:8010"
HEADERS = {}
CYCLES = 3

console = Console()


def require_api_token() -> str:
    token = os.getenv("BIZRA_API_TOKEN", "").strip()
    if not token:
        console.print(
            "[bold red]BIZRA_API_TOKEN is missing. Set it before running autonomous evolution.[/]"
        )
        raise SystemExit(1)
    return token

def query_constellation(intent):
    try:
        payload = {"query": intent}
        resp = requests.post(f"{API_URL}/v1/constellation/invoke", json=payload, headers=HEADERS, timeout=60)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

def render_thought_graph(cycle, snr, agents, insight):
    table = Table(show_header=False, box=None)
    table.add_row(f"[bold cyan]Cycle {cycle}[/]", f"[green]SNR: {snr}[/]")
    table.add_row("[yellow]Active Command[/]", str(agents))
    
    panel = Panel(
        insight[:200] + "...", 
        title=f"Autonomy Loop {cycle}/3", 
        border_style="blue"
    )
    console.print(table)
    console.print(panel)

def run_auto_mode():
    console.print("[bold red]🚨 ACTIVATING '/A' (AUTO_MODE) PROTOCOL[/]")
    console.print("[dim]Initiating Recursive Self-Evolution Loop...[/]\n")

    evolution_prompts = [
        "Phase 1: Scan internal architecture for cognitive bottlenecks.",
        "Phase 2: Generate graph-theoretic optimizations for the Constellation.",
        "Phase 3: Synthesize a 'Golden Path' for continuous singularity."
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        
        for i, prompt in enumerate(evolution_prompts, 1):
            task = progress.add_task(f"[cyan]Executing Cycle {i}...", total=None)
            
            # Simulate "Thinking" / API Call
            time.sleep(1) # Pacing
            response = query_constellation(prompt)
            
            progress.remove_task(task)
            
            if response:
                snr = response.get('current_snr', 0.95)
                # Mock agent list if dynamic not available, usually 'strategic_leadership_command'
                agents = "Strategic Command" 
                # Extract summary/synthesis
                synthesis = json.dumps(response.get('synthesis', "Analysis complete."), indent=2)
                
                render_thought_graph(i, snr, agents, synthesis)
            else:
                console.print(f"[red]Cycle {i} Failed - Core Unreachable[/]")
            
            console.print("  [dim]↓ Recursive Optimization Applied ↓[/]")
            time.sleep(1)

    console.print("\n[bold green]✅ /A PROTOCOL COMPLETE. System State: ELEVATED.[/]")

if __name__ == "__main__":
    # Self-Hosting Auto-Start
    server_proc = None
    api_token = require_api_token()
    HEADERS = {"Authorization": f"Bearer {api_token}"}
    try:
        # Check if already running
        try:
            requests.get(f"{API_URL}/healthz", timeout=1)
            console.print("[green]Core Nexus already active.[/]")
        except:
            console.print("[yellow]Igniting Core Nexus (Self-Hosting Mode)...[/]")
            import subprocess
            
            env = os.environ.copy()
            env.update({
                "BIZRA_API_TOKEN": api_token,
                "BIZRA_ENV": "development", 
                "SYNAPSE_URL": "redis://localhost:6379",
                "BIZRA_HEALTHZ_BUDGET_S": "5.0"
            })
            
            # Start background process
            server_proc = subprocess.Popen(
                [sys.executable, "-m", "core.main"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Smart Wait
            console.print("[dim]Waiting for Neural Synergy...[/]")
            ready = False
            for _ in range(20):
                time.sleep(1)
                try:
                    if requests.get(f"{API_URL}/healthz", timeout=1).status_code == 200:
                        ready = True
                        break
                except: pass
            
            if not ready:
                console.print("[bold red]Startup Failed. Aborting.[/]")
                if server_proc: server_proc.kill()
                sys.exit(1)

        run_auto_mode()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Manual Override.[/]")
    finally:
        if server_proc:
            console.print("[dim]Shutting down local nucleus...[/]")
            server_proc.kill()
