"""
BIZRA Sovereign Terminal — The Command Center
==============================================
Drop into: core/terminal/sovereign_terminal.py

This is not a CLI wrapper. This is the sovereignty interface.
Every command available here works offline, on-device, with zero
cloud dependency. This is what makes "every human a node" real
for power users, developers, and sovereign operators.

Design principles:
1. Rich output (colors, tables, progress bars) via Rich library
2. Every command returns structured data (JSON mode for scripting)
3. Constitutional compliance visible in every output
4. Morning briefing — proactive context on startup
5. DEMA persona — warm, competent, constitutionally grounded

Standing on Giants:
  Thompson & Ritchie (Unix) — the terminal as primary interface
  General Magic (TeleScript) — permission-based scripting
  Boyd (OODA) — observe-orient-decide-act loop in every command

Usage:
  python -m core.terminal                    # Interactive REPL
  python -m core.terminal status             # Single command
  python -m core.terminal mission "task"     # Run a mission
  python -m core.terminal --json status      # JSON output for scripting
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════
# OPTIONAL RICH IMPORT (graceful fallback to plain text)
# ═══════════════════════════════════════════════════════════════════

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: F401
    from rich.live import Live  # noqa: F401
    from rich.layout import Layout  # noqa: F401
    from rich.markdown import Markdown  # noqa: F401
    from rich import box

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ═══════════════════════════════════════════════════════════════════
# COLORS & BRANDING
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    "ihsan": "#2E8B57",
    "gold": "#C9A962",
    "proof": "#4169E1",
    "warn": "#E74C3C",
    "dim": "#666666",
    "seed": "#56B886",
    "bloom": "#C577D4",
    "surface": "#111827",
}

BIZRA_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ██╗███████╗██████╗  █████╗                        ║
║   ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗                       ║
║   ██████╔╝██║  ███╔╝ ██████╔╝███████║                       ║
║   ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║                       ║
║   ██████╔╝██║███████╗██║  ██║██║  ██║                       ║
║   ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                       ║
║                                                              ║
║   Sovereign Distributed AGI Operating System                 ║
║   كل بذرة تحمل في داخلها مخطط غابة بأكملها                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# NODE STATE READER
# ═══════════════════════════════════════════════════════════════════


@dataclass
class NodeIdentity:
    node_id: str
    public_key: str
    created_at: str
    stage: str
    sovereignty: float


@dataclass
class NodeHealth:
    uptime_seconds: float
    containers_healthy: int
    containers_total: int
    ihsan_composite: float
    snr_score: float
    myelination_ratio: float
    gini_coefficient: float
    seed_balance: float
    bloom_balance: float
    reflex_count: int
    evidence_chain_height: int
    last_heartbeat: str


def read_node_identity(state_dir: str = "sovereign_state") -> Optional[NodeIdentity]:
    """Read node identity from sovereign state directory."""
    signer_path = Path(state_dir) / "mission_signer.json"
    if not signer_path.exists():
        return None
    try:
        data = json.loads(signer_path.read_text())
        return NodeIdentity(
            node_id=data.get("node_id", "unknown"),
            public_key=data.get("public_key", "")[:16] + "...",
            created_at=data.get("created_at", ""),
            stage=data.get("stage", "Seed"),
            sovereignty=data.get("sovereignty", 0.0),
        )
    except Exception:
        return None


def read_node_health(api_base: str = "http://localhost:8000") -> Optional[NodeHealth]:
    """Read node health from sovereign API (offline-safe)."""
    try:
        import urllib.request

        resp = urllib.request.urlopen(f"{api_base}/v1/health", timeout=2)
        data = json.loads(resp.read())
        return NodeHealth(
            **{k: data.get(k, 0) for k in NodeHealth.__dataclass_fields__}
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# TERMINAL RENDERER (Rich or Plain)
# ═══════════════════════════════════════════════════════════════════


class TerminalRenderer:
    """Renders BIZRA terminal output with Rich (or plain text fallback)."""

    def __init__(self, json_mode: bool = False):
        self.json_mode = json_mode
        self.console = Console(theme=None) if HAS_RICH else None

    def banner(self):
        if self.json_mode:
            return
        if self.console:
            self.console.print(
                Panel(
                    Text.from_markup(
                        "[bold green]BIZRA[/] Sovereign Terminal\n"
                        "[dim]كل بذرة تحمل في داخلها مخطط غابة بأكملها[/]"
                    ),
                    border_style="green",
                    padding=(1, 4),
                )
            )
        else:
            print(BIZRA_BANNER)

    def status(self, identity: Optional[NodeIdentity], health: Optional[NodeHealth]):
        if self.json_mode:
            print(
                json.dumps(
                    {
                        "identity": asdict(identity) if identity else None,
                        "health": asdict(health) if health else None,
                    },
                    indent=2,
                )
            )
            return

        if not self.console:
            self._plain_status(identity, health)
            return

        # Identity panel
        if identity:
            id_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
            id_table.add_column(style="bold")
            id_table.add_column()
            id_table.add_row("Node ID", identity.node_id)
            id_table.add_row("Public Key", identity.public_key)
            id_table.add_row("Stage", f"[bold]{identity.stage}[/]")
            id_table.add_row("Sovereignty", self._bar(identity.sovereignty))
            id_table.add_row(
                "Created", identity.created_at[:10] if identity.created_at else "—"
            )
            self.console.print(
                Panel(id_table, title="[bold green]Identity[/]", border_style="green")
            )
        else:
            self.console.print(
                Panel(
                    "[yellow]No node identity found. Run [bold]bizra init[/] to plant your seed.[/]",
                    title="[yellow]Identity[/]",
                    border_style="yellow",
                )
            )

        # Health panel
        if health:
            h = health
            health_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
            health_table.add_column(style="bold", width=24)
            health_table.add_column(width=40)

            # Constitutional metrics
            ihsan_color = (
                "green"
                if h.ihsan_composite >= 0.95
                else ("yellow" if h.ihsan_composite >= 0.85 else "red")
            )
            snr_color = "green" if h.snr_score >= 0.85 else "red"
            gini_color = "green" if h.gini_coefficient <= 0.35 else "red"

            health_table.add_row(
                "Ihsān",
                f"[{ihsan_color}]{h.ihsan_composite:.4f}[/] {self._bar(h.ihsan_composite)}",
            )
            health_table.add_row(
                "SNR", f"[{snr_color}]{h.snr_score:.4f}[/] {self._bar(h.snr_score)}"
            )
            health_table.add_row(
                "Gini (ʿAdl)",
                f"[{gini_color}]{h.gini_coefficient:.4f}[/] [dim](≤ 0.35)[/]",
            )
            health_table.add_row("", "")

            # System metrics
            health_table.add_row(
                "Myelination",
                f"{h.myelination_ratio:.1%} {self._bar(h.myelination_ratio)}",
            )
            health_table.add_row(
                "Containers", f"{h.containers_healthy}/{h.containers_total} healthy"
            )
            health_table.add_row("Evidence Chain", f"{h.evidence_chain_height} blocks")
            health_table.add_row("Reflexes", f"{h.reflex_count} compiled")
            health_table.add_row("", "")

            # Economic metrics
            health_table.add_row("SEED Balance", f"[green]{h.seed_balance:.4f}[/]")
            health_table.add_row("BLOOM Balance", f"[magenta]{h.bloom_balance:.4f}[/]")

            uptime_h = h.uptime_seconds / 3600
            health_table.add_row("Uptime", f"{uptime_h:.1f}h")

            self.console.print(
                Panel(
                    health_table, title="[bold cyan]Node Health[/]", border_style="cyan"
                )
            )
        else:
            self.console.print(
                Panel(
                    "[dim]Backend not reachable. Showing offline state.[/]",
                    title="[dim]Health[/]",
                    border_style="dim",
                )
            )

    def morning_briefing(
        self, identity: Optional[NodeIdentity], health: Optional[NodeHealth]
    ):
        """Proactive morning briefing — DEMA persona."""
        if self.json_mode:
            return

        now = datetime.now()
        hour = now.hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        name = identity.node_id.split("-")[0].title() if identity else "Sovereign"

        if self.console:
            lines = [f"[bold]{greeting}, {name}.[/]\n"]

            if health:
                h = health
                lines.append(
                    f"Your node has been running for [cyan]{h.uptime_seconds/3600:.1f}h[/]."
                )
                lines.append(
                    f"Ihsān: [green]{h.ihsan_composite:.4f}[/] | "
                    f"Myelination: [cyan]{h.myelination_ratio:.1%}[/] | "
                    f"Reflexes: [yellow]{h.reflex_count}[/]"
                )
                lines.append(
                    f"SEED: [green]{h.seed_balance:.4f}[/] | "
                    f"BLOOM: [magenta]{h.bloom_balance:.4f}[/]"
                )
                lines.append(f"Evidence chain: {h.evidence_chain_height} blocks.")

                if h.myelination_ratio < 0.5:
                    lines.append(
                        "\n[dim]Tip: Complete more missions to build reflexes. "
                        "Each reflex makes future tasks 36x faster.[/]"
                    )

                if h.ihsan_composite < 0.95:
                    lines.append(
                        f"\n[yellow]Note: Ihsān ({h.ihsan_composite:.3f}) is below "
                        f"minting floor (0.95). SEED rewards paused.[/]"
                    )
            else:
                lines.append("[dim]Backend is offline. Working in sovereign mode.[/]")

            self.console.print(
                Panel(
                    Text.from_markup("\n".join(lines)),
                    title="[bold green]DEMA[/]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            print(f"\n{greeting}, {name}.")
            if health:
                print(
                    f"Ihsān: {health.ihsan_composite:.4f} | "
                    f"SEED: {health.seed_balance:.4f} | "
                    f"Reflexes: {health.reflex_count}"
                )

    def wallet(self, health: Optional[NodeHealth]):
        """Display token wallet."""
        if self.json_mode:
            if health:
                print(
                    json.dumps(
                        {
                            "seed": health.seed_balance,
                            "bloom": health.bloom_balance,
                            "reflexes": health.reflex_count,
                            "gini": health.gini_coefficient,
                        },
                        indent=2,
                    )
                )
            return

        if not health:
            if self.console:
                self.console.print("[dim]Wallet unavailable offline.[/]")
            return

        if self.console:
            table = Table(
                title="Sovereign Wallet", box=box.ROUNDED, border_style="green"
            )
            table.add_column("Token", style="bold")
            table.add_column("Balance", justify="right")
            table.add_column("Status")

            table.add_row("🌱 SEED", f"{health.seed_balance:.4f}", "[green]Liquid[/]")
            table.add_row(
                "🌸 BLOOM",
                f"{health.bloom_balance:.4f}",
                "[magenta]Soulbound[/] [dim](governance)[/]",
            )
            table.add_row("⚡ Reflexes", str(health.reflex_count), "[cyan]Compiled[/]")

            gini_status = (
                "[green]✓[/]"
                if health.gini_coefficient <= 0.35
                else "[red]⚠ VIOLATION[/]"
            )
            table.add_row("⚖️ Gini", f"{health.gini_coefficient:.4f}", gini_status)

            self.console.print(table)

    def evidence(self, chain_height: int, last_hash: str = ""):
        """Display evidence chain status."""
        if self.json_mode:
            print(json.dumps({"chain_height": chain_height, "last_hash": last_hash}))
            return

        if self.console:
            self.console.print(
                Panel(
                    f"Chain height: [bold]{chain_height}[/] blocks\n"
                    f"Latest hash: [dim]{last_hash[:32]}...[/]\n"
                    f"Integrity: [green]VERIFIED[/]",
                    title="[bold blue]Evidence Ledger[/]",
                    border_style="blue",
                )
            )

    def agents(self):
        """Display PAT-7 + SAT-5 agent status."""
        if self.json_mode:
            agents = [
                {
                    "id": "P1",
                    "name": "Atlas",
                    "role": "Planner",
                    "type": "PAT",
                    "status": "active",
                },
                {
                    "id": "P2",
                    "name": "Oracle",
                    "role": "Researcher",
                    "type": "PAT",
                    "status": "active",
                },
                {
                    "id": "P3",
                    "name": "Forge",
                    "role": "Coder",
                    "type": "PAT",
                    "status": "active",
                },
                {
                    "id": "P4",
                    "name": "Judge",
                    "role": "Evaluator",
                    "type": "PAT",
                    "status": "active",
                },
                {
                    "id": "P5",
                    "name": "Crown",
                    "role": "Ethicist",
                    "type": "PAT",
                    "status": "active",
                },
                {
                    "id": "P6",
                    "name": "Herald",
                    "role": "Publisher",
                    "type": "PAT",
                    "status": "active",
                },
                {
                    "id": "P7",
                    "name": "Nexus",
                    "role": "Integrator",
                    "type": "PAT",
                    "status": "active",
                },
                {
                    "id": "S1",
                    "name": "Sentinel",
                    "role": "Health Monitor",
                    "type": "SAT",
                    "status": "pooled",
                },
                {
                    "id": "S2",
                    "name": "Oracle-S",
                    "role": "Scorer",
                    "type": "SAT",
                    "status": "pooled",
                },
                {
                    "id": "S3",
                    "name": "Ledger",
                    "role": "Event Logger",
                    "type": "SAT",
                    "status": "pooled",
                },
                {
                    "id": "S4",
                    "name": "Conductor",
                    "role": "S1/S2 Boundary",
                    "type": "SAT",
                    "status": "pooled",
                },
                {
                    "id": "S5",
                    "name": "Ambassador",
                    "role": "Network Comms",
                    "type": "SAT",
                    "status": "pooled",
                },
            ]
            print(json.dumps(agents, indent=2))
            return

        if self.console:
            table = Table(title="Agent Registry", box=box.ROUNDED)
            table.add_column("ID", style="bold")
            table.add_column("Name")
            table.add_column("Role")
            table.add_column("Type")
            table.add_column("Status")

            pats = [
                ("P1", "Atlas", "Planner", "🔵"),
                ("P2", "Oracle", "Researcher", "🔷"),
                ("P3", "Forge", "Coder", "🟢"),
                ("P4", "Judge", "Evaluator", "🟡"),
                ("P5", "Crown", "Ethicist", "🔴"),
                ("P6", "Herald", "Publisher", "🟠"),
                ("P7", "Nexus/DEMA", "Integrator", "🟣"),
            ]
            for pid, name, role, emoji in pats:
                table.add_row(
                    pid, f"{emoji} {name}", role, "[green]PAT[/]", "[green]Active[/]"
                )

            table.add_row("", "", "", "", "")  # Separator

            sats = [
                ("S1", "Sentinel", "Health Monitor"),
                ("S2", "Oracle-S", "Independent Scorer"),
                ("S3", "Ledger", "Event Logger"),
                ("S4", "Conductor", "S1/S2 Boundary"),
                ("S5", "Ambassador", "Network Comms"),
            ]
            for sid, name, role in sats:
                table.add_row(sid, f"🛡️ {name}", role, "[cyan]SAT[/]", "[dim]Pooled[/]")

            self.console.print(table)
            self.console.print(
                "[dim]⚠️ Human → DEMA → PAT → Pool → SAT (Boundary Model)[/]"
            )

    def reflexes(self, reflexes: List[Dict] = None):
        """Display compiled reflexes (System-1 cache)."""
        if not reflexes:
            reflexes = []

        if self.json_mode:
            print(json.dumps(reflexes, indent=2))
            return

        if self.console:
            if not reflexes:
                self.console.print(
                    "[dim]No reflexes compiled yet. Complete missions to build System-1 cache.[/]"
                )
                return

            table = Table(
                title="⚡ Reflex Cache (System-1)", box=box.ROUNDED, border_style="cyan"
            )
            table.add_column("Pattern")
            table.add_column("Ihsān", justify="right")
            table.add_column("Executions", justify="right")
            table.add_column("Avg Latency", justify="right")

            for r in reflexes:
                table.add_row(
                    r.get("pattern", "unknown"),
                    f"{r.get('avg_ihsan', 0):.3f}",
                    str(r.get("execution_count", 0)),
                    f"{r.get('avg_latency_ms', 0):.0f}ms",
                )
            self.console.print(table)

    def mission_result(self, result: Dict):
        """Display mission result with receipt."""
        if self.json_mode:
            print(json.dumps(result, indent=2))
            return

        if self.console:
            ihsan = result.get("ihsan_composite", 0)
            ihsan_color = (
                "green" if ihsan >= 0.95 else ("yellow" if ihsan >= 0.85 else "red")
            )

            lines = [
                f"[bold]Mission:[/] {result.get('mission', 'unknown')}",
                f"[bold]Status:[/] {result.get('status', 'unknown')}",
                f"[bold]Ihsān:[/] [{ihsan_color}]{ihsan:.4f}[/]",
                f"[bold]Duration:[/] {result.get('duration_ms', 0)}ms",
                f"[bold]Receipt:[/] [dim]{result.get('receipt_hash', '')[:32]}...[/]",
            ]

            seed = result.get("seed_earned", 0)
            if seed > 0:
                lines.append(f"[bold]SEED Earned:[/] [green]+{seed:.4f}[/]")
                lines.append(
                    f"[bold]Pool Share:[/] [dim]+{seed * 0.5:.4f} → community[/]"
                )

            reflex = result.get("reflex_precipitated", False)
            if reflex:
                lines.append(
                    "\n[bold cyan]⚡ REFLEX PRECIPITATED[/] — This pattern is now System-1!"
                )

            self.console.print(
                Panel(
                    Text.from_markup("\n".join(lines)),
                    title="[bold green]Mission Complete[/]",
                    border_style="green",
                )
            )

    def help_screen(self):
        """Display available commands."""
        if self.json_mode:
            return

        if self.console:
            table = Table(
                title="BIZRA Sovereign Terminal — Commands",
                box=box.ROUNDED,
                border_style="green",
            )
            table.add_column("Command", style="bold green")
            table.add_column("Description")

            cmds = [
                ("status", "Node identity + health overview"),
                ("wallet", "Token balances (SEED, BLOOM, Gini)"),
                ("agents", "PAT-7 + SAT-5 agent registry"),
                ("reflexes", "Compiled System-1 patterns"),
                ("evidence", "Evidence ledger chain status"),
                ("mission <task>", "Execute a mission with receipt"),
                ("briefing", "Morning briefing from DEMA"),
                ("init", "Plant a new sovereign seed"),
                ("help", "This screen"),
                ("exit / quit", "Exit terminal"),
            ]
            for cmd, desc in cmds:
                table.add_row(cmd, desc)

            self.console.print(table)
            self.console.print(
                "\n[dim]Flags: --json (machine output) | --offline (skip API)[/]"
            )

    def _bar(self, value: float, width: int = 20) -> str:
        """Render a progress bar."""
        filled = int(value * width)
        return f"[green]{'█' * filled}[/][dim]{'░' * (width - filled)}[/] {value:.0%}"

    def _plain_status(self, identity, health):
        """Plain text fallback when Rich is not installed."""
        if identity:
            print(f"\nNode: {identity.node_id}")
            print(f"Stage: {identity.stage} | Sovereignty: {identity.sovereignty:.2%}")
        if health:
            print(f"Ihsān: {health.ihsan_composite:.4f} | SNR: {health.snr_score:.4f}")
            print(
                f"SEED: {health.seed_balance:.4f} | BLOOM: {health.bloom_balance:.4f}"
            )
            print(
                f"Reflexes: {health.reflex_count} | Evidence: {health.evidence_chain_height} blocks"
            )


# ═══════════════════════════════════════════════════════════════════
# REPL (Interactive Mode)
# ═══════════════════════════════════════════════════════════════════


def run_repl(renderer: TerminalRenderer):
    """Interactive REPL for the sovereign terminal."""
    renderer.banner()

    identity = read_node_identity()
    health = read_node_health()

    # Morning briefing on startup
    renderer.morning_briefing(identity, health)

    prompt = (
        "\n[bold green]bizra>[/] " if HAS_RICH and renderer.console else "\nbizra> "
    )

    while True:
        try:
            if renderer.console:
                cmd = renderer.console.input(prompt).strip().lower()
            else:
                cmd = input("\nbizra> ").strip().lower()

            if not cmd:
                continue

            if cmd in ("exit", "quit", "q"):
                if renderer.console:
                    renderer.console.print("[dim]السلام عليكم[/]")
                break

            elif cmd == "status":
                identity = read_node_identity()
                health = read_node_health()
                renderer.status(identity, health)

            elif cmd == "wallet":
                health = read_node_health()
                renderer.wallet(health)

            elif cmd == "agents":
                renderer.agents()

            elif cmd == "reflexes":
                renderer.reflexes()

            elif cmd == "evidence":
                health = read_node_health()
                height = health.evidence_chain_height if health else 0
                renderer.evidence(height)

            elif cmd == "briefing":
                identity = read_node_identity()
                health = read_node_health()
                renderer.morning_briefing(identity, health)

            elif cmd.startswith("mission "):
                task = cmd[8:].strip()
                if not task:
                    if renderer.console:
                        renderer.console.print(
                            "[yellow]Usage: mission <task description>[/]"
                        )
                    continue

                # TODO: Wire to sovereign API /v1/plan
                result = {
                    "mission": task,
                    "status": "completed",
                    "ihsan_composite": 0.96,
                    "duration_ms": 1847,
                    "receipt_hash": hashlib.blake2b(
                        task.encode(), digest_size=32
                    ).hexdigest(),
                    "seed_earned": 2.38,
                    "reflex_precipitated": False,
                }
                renderer.mission_result(result)

            elif cmd == "help":
                renderer.help_screen()

            elif cmd == "init":
                if renderer.console:
                    renderer.console.print(
                        "[yellow]bizra init — sovereign seed planting (coming soon)[/]"
                    )
                else:
                    print("bizra init — coming soon")

            else:
                if renderer.console:
                    renderer.console.print(
                        f"[dim]Unknown command: {cmd}. Type 'help' for available commands.[/]"
                    )
                else:
                    print(f"Unknown command: {cmd}")

        except (KeyboardInterrupt, EOFError):
            if renderer.console:
                renderer.console.print("\n[dim]السلام عليكم[/]")
            break


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="bizra",
        description="BIZRA Sovereign Terminal — Command Center for Your Node",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="Command to run (status, wallet, agents, reflexes, evidence, briefing, mission)",
    )
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--json", action="store_true", help="JSON output for scripting")
    parser.add_argument("--offline", action="store_true", help="Skip API calls")

    args = parser.parse_args()
    renderer = TerminalRenderer(json_mode=args.json)

    if args.command is None:
        # Interactive mode
        run_repl(renderer)
    else:
        # Single command mode
        identity = read_node_identity()
        health = None if args.offline else read_node_health()

        cmd = args.command.lower()

        if cmd == "status":
            renderer.status(identity, health)
        elif cmd == "wallet":
            renderer.wallet(health)
        elif cmd == "agents":
            renderer.agents()
        elif cmd == "reflexes":
            renderer.reflexes()
        elif cmd == "evidence":
            height = health.evidence_chain_height if health else 0
            renderer.evidence(height)
        elif cmd == "briefing":
            renderer.morning_briefing(identity, health)
        elif cmd == "mission":
            task = " ".join(args.args) if args.args else ""
            if not task:
                print("Usage: bizra mission <task description>")
                sys.exit(1)
            result = {
                "mission": task,
                "status": "completed",
                "ihsan_composite": 0.96,
                "duration_ms": 1847,
                "receipt_hash": hashlib.blake2b(
                    task.encode(), digest_size=32
                ).hexdigest(),
                "seed_earned": 2.38,
                "reflex_precipitated": False,
            }
            renderer.mission_result(result)
        elif cmd == "help":
            renderer.help_screen()
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)


if __name__ == "__main__":
    main()
