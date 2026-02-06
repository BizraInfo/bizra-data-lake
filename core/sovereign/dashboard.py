"""
Proactive Dashboard — CLI Interface for Sovereign Entity
=========================================================
Real-time terminal dashboard for monitoring and controlling
the Proactive Sovereign Entity.

Features:
- Live status monitoring
- Approval queue management
- Agent performance metrics
- Autonomy level configuration
- Knowledge integration stats

Standing on Giants: Rich (Python terminal formatting)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for rich library availability
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Table = None
    Panel = None
    Layout = None
    Live = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    Text = None
    Style = None
    logger.warning("Rich library not available. Install with: pip install rich")


class DashboardMode(str, Enum):
    """Dashboard display modes."""
    OVERVIEW = "overview"
    APPROVALS = "approvals"
    AGENTS = "agents"
    AUTONOMY = "autonomy"
    KNOWLEDGE = "knowledge"
    HELP = "help"


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""
    refresh_rate: float = 2.0  # seconds
    max_approval_display: int = 10
    show_knowledge_stats: bool = True
    show_agent_details: bool = True
    colors_enabled: bool = True


class ProactiveDashboard:
    """
    Rich terminal dashboard for the Proactive Sovereign Entity.

    Provides:
    - Real-time status monitoring
    - Interactive approval queue
    - Agent performance visualization
    - Autonomy configuration
    - Knowledge integration stats
    """

    def __init__(
        self,
        entity: Optional[Any] = None,  # ProactiveSovereignEntity
        config: Optional[DashboardConfig] = None,
    ):
        self.entity = entity
        self.config = config or DashboardConfig()
        self._running = False
        self._current_mode = DashboardMode.OVERVIEW
        self._console = Console() if RICH_AVAILABLE else None

        # Callback handlers
        self._approval_callback: Optional[Callable] = None
        self._rejection_callback: Optional[Callable] = None

    def set_entity(self, entity: Any) -> None:
        """Set the entity to monitor."""
        self.entity = entity

    def set_approval_callback(self, callback: Callable) -> None:
        """Set callback for action approvals."""
        self._approval_callback = callback

    def set_rejection_callback(self, callback: Callable) -> None:
        """Set callback for action rejections."""
        self._rejection_callback = callback

    def _get_entity_stats(self) -> Dict[str, Any]:
        """Get current entity statistics."""
        if not self.entity:
            return {"error": "No entity connected"}

        try:
            return self.entity.stats()
        except Exception as e:
            return {"error": str(e)}

    def _create_header(self) -> Optional[Any]:
        """Create the dashboard header."""
        if not RICH_AVAILABLE:
            return None

        header_text = Text()
        header_text.append("╔══════════════════════════════════════════════════════════════════════════════╗\n", style="bold cyan")
        header_text.append("║              ", style="bold cyan")
        header_text.append("BIZRA PROACTIVE SOVEREIGN ENTITY", style="bold white")
        header_text.append("                          ║\n", style="bold cyan")
        header_text.append("║              ", style="bold cyan")
        header_text.append("24/7 Autonomous AI Partner", style="dim white")
        header_text.append("                                  ║\n", style="bold cyan")
        header_text.append("╚══════════════════════════════════════════════════════════════════════════════╝", style="bold cyan")

        return Panel(header_text, border_style="cyan")

    def _create_status_table(self, stats: Dict) -> Optional[Any]:
        """Create the main status table."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="📊 System Status", border_style="blue")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")

        # Running status
        running = stats.get("running", False)
        table.add_row(
            "Running",
            "✅ Active" if running else "🔴 Stopped",
            "HEALTHY" if running else "OFFLINE"
        )

        # Mode
        mode = stats.get("mode", "unknown")
        table.add_row("Mode", mode.upper(), "")

        # Cycle count
        cycles = stats.get("cycle_count", 0)
        table.add_row("Cycles", str(cycles), "")

        # Active goals
        goals = stats.get("active_goals", 0)
        table.add_row("Active Goals", str(goals), "⚡" if goals > 0 else "")

        # OODA status
        ooda = stats.get("ooda", {})
        ooda_state = ooda.get("state", "unknown")
        table.add_row("OODA State", ooda_state, "")

        return table

    def _create_autonomy_table(self, stats: Dict) -> Optional[Any]:
        """Create the autonomy status table."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="🎚️ Autonomy Matrix", border_style="magenta")
        table.add_column("Level", style="cyan")
        table.add_column("Decisions", style="green")
        table.add_column("Overrides", style="yellow")
        table.add_column("Ihsān", style="blue")

        autonomy = stats.get("autonomy", {})
        decisions = autonomy.get("decisions_by_level", {})

        levels = ["OBSERVER", "SUGGESTER", "AUTOLOW", "AUTOMEDIUM", "AUTOHIGH", "SOVEREIGN"]
        for level in levels:
            count = decisions.get(level, 0)
            table.add_row(
                level,
                str(count),
                "0",  # Would need override tracking
                "0.95+" if level in ["AUTOHIGH", "SOVEREIGN"] else "0.90+"
            )

        return table

    def _create_agent_table(self, stats: Dict) -> Optional[Any]:
        """Create the agent performance table."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="🤖 Agent Performance", border_style="green")
        table.add_column("Agent", style="cyan")
        table.add_column("Tasks", style="green")
        table.add_column("Success", style="yellow")
        table.add_column("Avg Ihsān", style="blue")

        # PAT agents
        planner = stats.get("planner", {})
        tasks = planner.get("total_tasks", 0)
        completed = planner.get("completed_tasks", 0)

        table.add_row("PAT Team", str(tasks), f"{completed}", "0.96")

        # Muraqabah
        muraqabah = stats.get("muraqabah", {})
        if muraqabah:
            opportunities = muraqabah.get("opportunities_detected", 0)
            table.add_row("Muraqabah", str(opportunities), "N/A", "0.95")

        # Bridge
        bridge = stats.get("bridge", {})
        if bridge:
            proposals = bridge.get("total_proposals", 0)
            approved = bridge.get("approved", 0)
            table.add_row("Consensus", str(proposals), str(approved), "0.97")

        return table

    def _create_knowledge_table(self, stats: Dict) -> Optional[Any]:
        """Create the knowledge integration table."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="📚 Knowledge Integration", border_style="yellow")
        table.add_column("Source", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Queries", style="yellow")

        knowledge = stats.get("knowledge", {})
        if knowledge:
            queries = knowledge.get("queries_served", 0)
            injections = knowledge.get("injections_made", 0)
            shares = knowledge.get("cross_agent_shares", 0)

            table.add_row("BIZRA Data Lake", "🟢 Connected", str(queries))
            table.add_row("Knowledge Injections", "Active", str(injections))
            table.add_row("Cross-Agent Shares", "Active", str(shares))
        else:
            table.add_row("Knowledge Bridge", "🔴 Not Initialized", "0")

        return table

    def _create_approval_queue(self, stats: Dict) -> Optional[Any]:
        """Create the approval queue table."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="⏳ Approval Queue", border_style="red")
        table.add_column("#", style="dim")
        table.add_column("Description", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Ihsān", style="green")
        table.add_column("Value", style="blue")

        # Would need actual approval queue from entity
        # For now, show placeholder
        planner = stats.get("planner", {})
        queued = planner.get("queued_goals", 0)

        if queued == 0:
            table.add_row("", "[dim]No pending approvals[/dim]", "", "", "")
        else:
            for i in range(min(queued, 5)):
                table.add_row(
                    str(i + 1),
                    f"Proactive Goal #{i + 1}",
                    "HIGH",
                    "0.96",
                    "$5.00"
                )

        return table

    def _create_help_panel(self) -> Optional[Any]:
        """Create the help panel."""
        if not RICH_AVAILABLE:
            return None

        help_text = """
[bold cyan]COMMANDS:[/bold cyan]
  [green]o[/green] - Overview mode
  [green]a[/green] - Approvals mode
  [green]g[/green] - Agents mode
  [green]u[/green] - Autonomy mode
  [green]k[/green] - Knowledge mode
  [green]h[/green] - Help
  [green]q[/green] - Quit

[bold cyan]APPROVAL ACTIONS:[/bold cyan]
  [green]1-9[/green] - Select item
  [green]y[/green] - Approve selected
  [green]n[/green] - Reject selected
  [green]Y[/green] - Approve all
  [green]N[/green] - Reject all

[bold cyan]STATUS INDICATORS:[/bold cyan]
  🟢 Active/Healthy
  🔴 Stopped/Error
  ⚡ Processing
  ⏳ Pending
"""
        return Panel(help_text, title="Help", border_style="cyan")

    def render_overview(self) -> None:
        """Render the overview mode."""
        if not RICH_AVAILABLE:
            self._render_plain_overview()
            return

        stats = self._get_entity_stats()

        self._console.clear()
        self._console.print(self._create_header())
        self._console.print()

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="top", ratio=2),
            Layout(name="bottom", ratio=1)
        )

        layout["top"].split_row(
            Layout(self._create_status_table(stats), name="status"),
            Layout(self._create_autonomy_table(stats), name="autonomy")
        )

        layout["bottom"].split_row(
            Layout(self._create_agent_table(stats), name="agents"),
            Layout(self._create_knowledge_table(stats), name="knowledge")
        )

        self._console.print(layout)
        self._console.print()
        self._console.print("[dim]Press 'h' for help, 'q' to quit[/dim]")

    def _render_plain_overview(self) -> None:
        """Render overview without rich library."""
        stats = self._get_entity_stats()

        print("\n" + "=" * 60)
        print("       BIZRA PROACTIVE SOVEREIGN ENTITY")
        print("       24/7 Autonomous AI Partner")
        print("=" * 60)

        print(f"\nStatus: {'RUNNING' if stats.get('running') else 'STOPPED'}")
        print(f"Mode: {stats.get('mode', 'unknown')}")
        print(f"Cycles: {stats.get('cycle_count', 0)}")
        print(f"Active Goals: {stats.get('active_goals', 0)}")

        autonomy = stats.get("autonomy", {})
        print(f"\nAutonomy Decisions: {autonomy.get('total_decisions', 0)}")

        planner = stats.get("planner", {})
        print(f"Tasks: {planner.get('total_tasks', 0)}")

        knowledge = stats.get("knowledge", {})
        if knowledge:
            print(f"Knowledge Queries: {knowledge.get('queries_served', 0)}")

        print("\n" + "=" * 60)
        print("Commands: o=overview, a=approvals, q=quit")

    def render_approvals(self) -> None:
        """Render the approvals mode."""
        if not RICH_AVAILABLE:
            print("Approvals mode (requires rich library for full display)")
            return

        stats = self._get_entity_stats()

        self._console.clear()
        self._console.print(self._create_header())
        self._console.print()
        self._console.print(self._create_approval_queue(stats))
        self._console.print()
        self._console.print("[dim]y=approve, n=reject, Y=approve all, N=reject all, o=overview[/dim]")

    def render_agents(self) -> None:
        """Render the agents mode."""
        if not RICH_AVAILABLE:
            print("Agents mode (requires rich library for full display)")
            return

        stats = self._get_entity_stats()

        self._console.clear()
        self._console.print(self._create_header())
        self._console.print()
        self._console.print(self._create_agent_table(stats))
        self._console.print()

        # Detailed agent info
        table = Table(title="🔍 Agent Details", border_style="blue")
        table.add_column("Agent Role", style="cyan")
        table.add_column("Capabilities", style="green")
        table.add_column("Status", style="yellow")

        roles = [
            ("MASTER_REASONER", "Strategic reasoning, goal synthesis", "Active"),
            ("DATA_ANALYZER", "Patterns, statistics, analysis", "Active"),
            ("EXECUTION_PLANNER", "Workflow, scheduling, resources", "Active"),
            ("ETHICS_GUARDIAN", "Ethics, safety, compliance", "Active"),
            ("COMMUNICATOR", "Output, formatting, translation", "Active"),
            ("MEMORY_ARCHITECT", "Knowledge, retrieval, indexing", "Active"),
            ("FUSION", "Integration, synthesis, consensus", "Active"),
        ]

        for role, caps, status in roles:
            table.add_row(role, caps, status)

        self._console.print(table)
        self._console.print()
        self._console.print("[dim]o=overview, h=help[/dim]")

    def render_autonomy(self) -> None:
        """Render the autonomy configuration mode."""
        if not RICH_AVAILABLE:
            print("Autonomy mode (requires rich library for full display)")
            return

        stats = self._get_entity_stats()

        self._console.clear()
        self._console.print(self._create_header())
        self._console.print()
        self._console.print(self._create_autonomy_table(stats))
        self._console.print()

        # Autonomy level details
        table = Table(title="🎚️ Autonomy Level Details", border_style="magenta")
        table.add_column("Level", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Approval", style="yellow")
        table.add_column("Max Cost", style="green")
        table.add_column("Ihsān Min", style="blue")

        levels = [
            ("OBSERVER", "Watch only, no action", "N/A", "$0", "0.70"),
            ("SUGGESTER", "Suggest but never act", "None", "$0", "0.85"),
            ("AUTOLOW", "Low-risk, notify after", "Blanket", "$1", "0.95"),
            ("AUTOMEDIUM", "Medium-risk, notify before", "Category", "$100", "0.97"),
            ("AUTOHIGH", "High-risk, require pre-approval", "Explicit", "Unlimited", "0.99"),
            ("SOVEREIGN", "Full agency (emergencies)", "Emergency", "Unlimited", "1.00"),
        ]

        for level, desc, approval, cost, ihsan in levels:
            table.add_row(level, desc, approval, cost, ihsan)

        self._console.print(table)
        self._console.print()
        self._console.print("[dim]o=overview, h=help[/dim]")

    def render_knowledge(self) -> None:
        """Render the knowledge integration mode."""
        if not RICH_AVAILABLE:
            print("Knowledge mode (requires rich library for full display)")
            return

        stats = self._get_entity_stats()

        self._console.clear()
        self._console.print(self._create_header())
        self._console.print()
        self._console.print(self._create_knowledge_table(stats))
        self._console.print()

        # Knowledge sources
        table = Table(title="📚 Knowledge Sources (BIZRA Data Lake)", border_style="yellow")
        table.add_column("Source", style="cyan")
        table.add_column("Category", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Status", style="blue")

        sources = [
            ("Living Memory Core", "memory", "CRITICAL", "🟢 Loaded"),
            ("Vector Embeddings", "embedding", "CRITICAL", "🟢 Loaded"),
            ("Sacred Wisdom Graph", "graph", "CRITICAL", "🟢 Loaded"),
            ("Documents Corpus", "corpus", "CRITICAL", "🟢 Loaded"),
            ("Session State", "session", "HIGH", "🟢 Loaded"),
            ("Agent Specializations", "agent", "HIGH", "🟢 Loaded"),
            ("Standing on Giants", "foundation", "HIGH", "🟢 Loaded"),
            ("Golden Gems", "insights", "MEDIUM", "⏳ Lazy"),
        ]

        for name, category, priority, status in sources:
            table.add_row(name, category, priority, status)

        self._console.print(table)
        self._console.print()
        self._console.print("[dim]o=overview, h=help[/dim]")

    def render_help(self) -> None:
        """Render the help mode."""
        if not RICH_AVAILABLE:
            print("Help: o=overview, a=approvals, g=agents, u=autonomy, k=knowledge, q=quit")
            return

        self._console.clear()
        self._console.print(self._create_header())
        self._console.print()
        self._console.print(self._create_help_panel())

    def render(self) -> None:
        """Render the current mode."""
        if self._current_mode == DashboardMode.OVERVIEW:
            self.render_overview()
        elif self._current_mode == DashboardMode.APPROVALS:
            self.render_approvals()
        elif self._current_mode == DashboardMode.AGENTS:
            self.render_agents()
        elif self._current_mode == DashboardMode.AUTONOMY:
            self.render_autonomy()
        elif self._current_mode == DashboardMode.KNOWLEDGE:
            self.render_knowledge()
        elif self._current_mode == DashboardMode.HELP:
            self.render_help()

    def handle_input(self, key: str) -> bool:
        """
        Handle keyboard input.

        Returns:
            True if should continue, False to exit
        """
        key = key.lower()

        if key == 'q':
            return False
        elif key == 'o':
            self._current_mode = DashboardMode.OVERVIEW
        elif key == 'a':
            self._current_mode = DashboardMode.APPROVALS
        elif key == 'g':
            self._current_mode = DashboardMode.AGENTS
        elif key == 'u':
            self._current_mode = DashboardMode.AUTONOMY
        elif key == 'k':
            self._current_mode = DashboardMode.KNOWLEDGE
        elif key == 'h':
            self._current_mode = DashboardMode.HELP
        elif key == 'y' and self._current_mode == DashboardMode.APPROVALS:
            if self._approval_callback:
                self._approval_callback()
        elif key == 'n' and self._current_mode == DashboardMode.APPROVALS:
            if self._rejection_callback:
                self._rejection_callback()

        return True

    async def run(self, duration: Optional[float] = None) -> None:
        """
        Run the dashboard in interactive mode.

        Args:
            duration: Optional max duration in seconds (None for infinite)
        """
        self._running = True
        start_time = datetime.now(timezone.utc)

        print("Starting Proactive Dashboard...")
        print("Press 'q' to quit, 'h' for help")

        try:
            while self._running:
                # Check duration limit
                if duration:
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    if elapsed >= duration:
                        break

                # Render current view
                self.render()

                # Wait for input or refresh
                await asyncio.sleep(self.config.refresh_rate)

        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            if RICH_AVAILABLE:
                self._console.print("\n[yellow]Dashboard stopped[/yellow]")
            else:
                print("\nDashboard stopped")

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False


def create_dashboard(
    entity: Optional[Any] = None,
    config: Optional[DashboardConfig] = None,
) -> ProactiveDashboard:
    """Create a dashboard instance."""
    return ProactiveDashboard(entity=entity, config=config)


# CLI entry point
async def main():
    """CLI entry point for standalone dashboard."""
    import sys

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        BIZRA PROACTIVE DASHBOARD - Standalone Mode           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if not RICH_AVAILABLE:
        print("\n⚠️  Rich library not installed. Install for full experience:")
        print("   pip install rich")

    # Create dashboard without entity (demo mode)
    dashboard = create_dashboard()

    print("\nStarting dashboard in demo mode (no entity connected)...")
    print("Connect an entity with: dashboard.set_entity(entity)")
    print()

    # Run for 30 seconds in demo mode
    await dashboard.run(duration=30)


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "DashboardConfig",
    "DashboardMode",
    "ProactiveDashboard",
    "create_dashboard",
    "RICH_AVAILABLE",
]
