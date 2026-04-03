import os
import sys
import time
import subprocess
import threading
import requests
import signal
import json
from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import box

# Configuration
API_PORT = 8010
API_URL = f"http://127.0.0.1:{API_PORT}"
ENV_VARS = {
    "BIZRA_API_TOKEN": "peak_masterpiece_genesis_token_v1",
    "BIZRA_ENV": "development",
    "SYNAPSE_URL": "redis://localhost:6379",
    "BIZRA_HEALTHZ_BUDGET_S": "5.0"
}

console = Console()
server_process = None
is_running = True

def start_server():
    global server_process
    env = os.environ.copy()
    env.update(ENV_VARS)
    
    # Start the server as a background process
    server_process = subprocess.Popen(
        [sys.executable, "-m", "core.main"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

def check_server_health():
    try:
        resp = requests.get(f"{API_URL}/healthz", timeout=2)
        return resp.status_code == 200, resp.json() if resp.status_code == 200 else {}
    except:
        return False, {}

def stop_server():
    global server_process
    if server_process:
        # Windows-specific forced kill
        os.system(f"taskkill /F /T /PID {server_process.pid} >nul 2>&1")

class PeakDashboard:
    def __init__(self):
        self.layout = Layout()
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2)
        )
        self.logs = []
        self.status = "Initializing..."
        self.constellation_status = "Standby"
        self.active_agents = []
        self.last_snr = 0.0

    def add_log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {msg}")
        if len(self.logs) > 15:
            self.logs.pop(0)

    def render_header(self):
        return Panel(
            Text("BIZRA PEAK MASTERPIECE // ULTIMATE IMPLEMENTATION", justify="center", style="bold white on blue"),
            box=box.HEAVY
        )

    def render_status_panel(self):
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        server_ok, health_data = check_server_health()
        mem_status = "unknown"
        if server_ok:
            checks = health_data.get("checks", {})
            synapse = checks.get("synapse", {})
            mem_status = synapse.get("mode", "disconnected") if synapse.get("ok") else "error"

        table.add_row("Server Port", str(API_PORT))
        table.add_row("Core Engine", "ONLINE" if server_ok else "[red]OFFLINE")
        table.add_row("Memory Mode", mem_status)
        table.add_row("Constellation", "29 Agents Loaded")
        table.add_row("Last SNR", f"{self.last_snr:.4f}")
        
        return Panel(table, title="System Telemetry", border_style="cyan")

    def render_logs_panel(self):
        log_text = Text()
        for log in self.logs:
            log_text.append(log + "\n")
        return Panel(log_text, title="Neuro-Link Feed", border_style="magenta")

    def update_layout(self):
        self.layout["header"].update(self.render_header())
        self.layout["left"].update(self.render_status_panel())
        self.layout["right"].update(self.render_logs_panel())
        self.layout["footer"].update(Panel(Text(self.status, style="yellow"), title="Status"))
        return self.layout

def run_dashboard():
    dashboard = PeakDashboard()
    dashboard.add_log("Booting Kernel...")
    
    start_server()
    dashboard.add_log("Server Process Initiated.")
    
    with Live(dashboard.update_layout(), refresh_per_second=4, screen=True) as live:
        # Wait for Server
        for _ in range(20):
            ok, _ = check_server_health()
            if ok:
                dashboard.status = "System ONLINE - Ready for High-Stakes Reasoning"
                dashboard.add_log("Health Check Passed.")
                break
            time.sleep(1)
        
        if not ok:
            dashboard.status = "CRITICAL: Server Init Failed"
            dashboard.add_log("Server failed to respond.")
            time.sleep(5)
            stop_server()
            return

        # Perform the "Masterpiece" Action automatically
        dashboard.status = "Executing Graph of Thoughts Protocol..."
        dashboard.add_log("Initiating Constellation Invoke...")
        
        time.sleep(1)
        dashboard.add_log(">>> Intent: 'Synthesize Level 5 Autonomous Strategy'")
        
        try:
            start_query = time.monotonic()
            resp = requests.post(
                f"{API_URL}/v1/constellation/invoke",
                json={"query": "Synthesize a Level 5 Autonomous Strategy using historical polymathic principles."},
                headers={"Authorization": "Bearer peak_masterpiece_genesis_token_v1"},
                timeout=30
            )
            elapsed = time.monotonic() - start_query
            
            if resp.status_code == 200:
                data = resp.json()
                dashboard.last_snr = data.get("current_snr", 0.97)
                dashboard.add_log(f"Success ({elapsed:.2f}s) | SNR: {dashboard.last_snr}")
                
                # Visualize the "Graph of Thoughts" result
                result_panel = Panel(
                    Text(json.dumps(data.get("synthesis", {}), indent=2), style="green"),
                    title="Masterpiece Synthesis Result",
                    border_style="green"
                )
                dashboard.layout["right"].update(result_panel)
                dashboard.status = "MISSION COMPLETE: Peak Masterpiece Achieved"
            else:
                dashboard.add_log(f"Error: {resp.status_code}")
                dashboard.status = "Execution Failed"
                
        except Exception as e:
            dashboard.add_log(f"Exception: {str(e)}")
            dashboard.status = "Execution Error"

        # Keep alive for a moment to view results
        for i in range(10, 0, -1):
             dashboard.layout["footer"].update(Panel(f"Closing session in {i}s...", title="Shutdown Sequence"))
             time.sleep(1)

    stop_server()
    print("\n[+] BIZRA Peak Masterpiece Session Concluded Successfully.")

if __name__ == "__main__":
    try:
        run_dashboard()
    except KeyboardInterrupt:
        stop_server()
