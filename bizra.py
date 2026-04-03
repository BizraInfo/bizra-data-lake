
import sys
import os
import time
import json
import asyncio
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import print as rprint

# Internal BIZRA Imports
from bizra_kernel.kernel import SystemProtocolKernel, KernelConfig
from bizra_kernel.identity import get_identity
from bizra_kernel.omni_awareness import OmniAwareness
from bizra_kernel.memory_system import CognitivePermanence
from bizra_kernel.plugins import PluginContext, PluginRuntime, PluginRuntimeError

app = typer.Typer(help="BIZRA Sovereign Organism: Official PAT CLI")
plugin_app = typer.Typer(help="Plugin commands for spearpoint features.")
app.add_typer(plugin_app, name="plugin")
console = Console()

def get_kernel():
    config = KernelConfig(
        enable_verification=True,
        enable_sape=True,
        ihsan_threshold=0.99,
        snr_target=0.95
    )
    return SystemProtocolKernel(config)


def get_plugin_runtime() -> PluginRuntime:
    return PluginRuntime()


def build_plugin_context() -> PluginContext:
    return PluginContext(
        identity=get_identity(),
        kernel_factory=get_kernel,
        console=console,
        metadata={"entrypoint": "bizra-cli"},
    )

@app.command()
def status():
    """Display the Sovereign Organism's current health and hardware status."""
    identity = get_identity()
    mem = CognitivePermanence()
    awareness = OmniAwareness(mem)
    report = awareness.synchronize_self_model()
    
    # Header
    console.print(Panel(
        f"[bold gold1]{identity.get_sovereignty_declaration()}[/bold gold1]\n"
        f"[dim]Root Anchor: {identity.root_hash[:16]}... | Activated: {identity.activated_at}[/dim]",
        title="[bold]Sovereign Identity[/bold]",
        border_style="gold1"
    ))
    
    # Hardware Telemetry
    hw_table = Table(title="Hardware Budget & Telemetry", show_header=True, header_style="bold blue")
    hw_table.add_column("Component", style="dim")
    hw_table.add_column("Status", justify="right")
    hw_table.add_column("Metrics", justify="right")
    
    gpu_metrics = report['budget'].get('gpu')
    gpu_active = gpu_metrics is not None
    vram_total = gpu_metrics.get('vram_total_gb', 0) if gpu_active else 0
    hw_table.add_row(
        "RTX 4090", 
        "[green]ACTIVE[/green]" if gpu_active else "[red]INACTIVE[/red]",
        f"{vram_total:.1f} GB Dedicated"
    )
    hw_table.add_row(
        "CPU Cognitive Ceiling", 
        "[green]OPTIMAL[/green]", 
        f"{os.cpu_count()} Cores"
    )
    hw_table.add_row(
        "Local LLM (Ollama)", 
        "[green]READY[/green]", 
        "Model: DeepSeek-R1-Sovereign"
    )
    
    console.print(hw_table)
    
    # Resource Pool Telemetry
    census = report.get('census', {})
    if census:
        res_table = Table(title="Sovereign Resource Pool (Home Base)", show_header=True, header_style="bold magenta")
        res_table.add_column("Resource Type", style="dim")
        res_table.add_column("Census Metric", justify="right")
        
        data_uni = census.get('data_universe', {})
        res_table.add_row("Total Files Indexed", f"{data_uni.get('total_files', 0):,}")
        res_table.add_row("3-Year Work Universe", f"{data_uni.get('modified_in_range', 0):,} files")
        res_table.add_row("Total Data Footprint", f"{data_uni.get('total_size_gb', 0):.2f} GB")
        
        sw = census.get('software', {}).get('runtimes', {})
        active_sw = [k for k, v in sw.items() if "Not Detected" not in v]
        res_table.add_row("Detected Runtimes", ", ".join(active_sw))
        
        console.print(res_table)
    
    # Ecosystem Territory
    console.print(f"\n[bold]Ecosystem Territory:[/bold] [cyan]{report['territory']['total_nodes']}[/cyan] nodes recognized.")
    for entry in report['territory']['map']:
        console.print(f"  • [dim]{entry['node']}:[/dim] [blue]{entry['path']}[/blue] [dim]({entry['status']})[/dim]")

@app.command()
def chat():
    """Enter the interactive Sovereign Interactive Workspace (REPL)."""
    identity = get_identity()
    kernel = get_kernel()
    
    console.print(f"\n[bold gold1]Welcome home, {identity.architect.name}.[/bold gold1]")
    console.print("[dim]I am standing on the shoulders of giants. How shall we build today?[/dim]\n")
    
    while True:
        query = console.input(f"[bold green]{identity.architect.name}@Bizra-Node0[/bold green]:[bold blue]~$ [/bold blue]")
        
        if query.lower() in ["exit", "quit", "sleep"]:
            console.print("\n[bold red]Shutting down cognitive circuits. Peace be upon the Architect.[/bold red]")
            break
            
        if not query.strip():
            continue
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Thinking...", total=None)
            result = asyncio.run(kernel.execute_local_inference(query))
            
        console.print("\n" + "─"*console.width)
        console.print(Panel(
            Markdown(result.response),
            title=f"[bold]BIZRA v{kernel.VERSION}[/bold]",
            border_style="cyan"
        ))
        
        # Metadata footer
        stats = (
            f"[dim]SNR: [bold]{result.snr_metrics.snr_score:.3f}[/bold] | "
            f"Ihsān: [bold]{result.ihsan_vector.composite_score:.3f}[/bold] | "
            f"Latency: [bold]{result.latency_ms}ms[/bold] | "
            f"Protocol: {result.protocol_hash[:8]}[/dim]"
        )
        console.print(stats, justify="right")
        console.print("─"*console.width + "\n")

@app.command()
def ask(query: str):
    """Execute a single sovereign query and return the result."""
    kernel = get_kernel()
    result = asyncio.run(kernel.execute_local_inference(query))
    console.print(Markdown(result.response))


@app.command()
def onboard(
    json_output: bool = typer.Option(
        False, "--json", help="Emit plugin output as machine-readable JSON."
    )
):
    """Run the onboarding plugin entrypoint (front-door hook)."""
    try:
        runtime = get_plugin_runtime()
        result = runtime.invoke(
            name="onboarding",
            action="start",
            payload={},
            context=build_plugin_context(),
        )
    except PluginRuntimeError as exc:
        console.print(f"[red]Onboarding plugin error:[/red] {exc}")
        raise typer.Exit(code=1)

    if json_output:
        console.print_json(data=result)
        return

    identity = result.get("identity", {})
    checklist = result.get("checklist", [])
    console.print(
        Panel(
            f"{result.get('message', 'Onboarding plugin executed.')}\n"
            f"[dim]Architect: {identity.get('architect', 'unknown')} | "
            f"Node: {identity.get('node_id', 'pending')}[/dim]",
            title="[bold]BIZRA Onboarding Plugin[/bold]",
            border_style="cyan",
        )
    )
    if checklist:
        for idx, step in enumerate(checklist, 1):
            console.print(f"{idx}. {step}")
    console.print(
        "[dim]Registry: config/plugins_registry.json "
        "(override with BIZRA_PLUGIN_REGISTRY)[/dim]"
    )


@plugin_app.command("list")
def plugin_list():
    """List configured CLI plugins from registry."""
    try:
        runtime = get_plugin_runtime()
        specs = runtime.list_plugins()
    except PluginRuntimeError as exc:
        console.print(f"[red]Plugin registry error:[/red] {exc}")
        raise typer.Exit(code=1)

    if not specs:
        console.print("[yellow]No plugins configured.[/yellow]")
        return

    table = Table(title="BIZRA CLI Plugins", show_header=True, header_style="bold blue")
    table.add_column("Name", style="bold")
    table.add_column("Enabled", justify="center")
    table.add_column("Module")
    table.add_column("Description")
    for spec in specs:
        table.add_row(
            spec.name,
            "yes" if spec.enabled else "no",
            spec.module,
            spec.description or "-",
        )
    console.print(table)


@plugin_app.command("run")
def plugin_run(
    name: str = typer.Argument(..., help="Plugin name from registry."),
    action: str = typer.Option("start", "--action", help="Plugin action to execute."),
    payload: str = typer.Option("{}", "--payload", help="JSON payload."),
):
    """Run an arbitrary plugin action for testing/integration."""
    try:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("Payload must be a JSON object.")
    except Exception as exc:
        console.print(f"[red]Invalid payload:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        runtime = get_plugin_runtime()
        result = runtime.invoke(
            name=name,
            action=action,
            payload=parsed,
            context=build_plugin_context(),
        )
    except PluginRuntimeError as exc:
        console.print(f"[red]Plugin execution error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print_json(data=result)

@app.command()
def memory():
    """Display the status of the 5-Layer Cognitive Workspace."""
    mem = CognitivePermanence()
    
    table = Table(title="Cognitive Workspace Layers", border_style="magenta")
    table.add_column("Layer", style="bold")
    table.add_column("Content Type", style="dim")
    table.add_column("Persistence", justify="right")
    
    table.add_row("L1: Immediate", "Volatile Perception", f"{len(mem.layers['L1'])} blocks")
    table.add_row("L2: Working", "Granular Condensation", f"{len(mem.layers['L2'])} blocks")
    table.add_row("L3: Episodic", "Deep Consolidation", f"{len(mem.layers['L3'])} episodes")
    table.add_row("L4: Semantic", "HyperGraph Facts", f"{len(mem.layers['L4'])} facts")
    table.add_row("L5: Procedural", "Expertise (AATC)", f"{len(mem.layers['L5'])} skills")
    
    console.print(table)

if __name__ == "__main__":
    app()
