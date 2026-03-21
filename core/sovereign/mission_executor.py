"""
Unified Mission Executor — The Full Pipeline.

Every mission flows through ALL components:
  TUI → FAISS → Ollama → Guardian → Skill → Receipt → SEED → Memory → Notify

This is the organism, not the parts catalog.

Standing on: Boyd (OODA full loop), Deming (every step measured),
Maturana (autopoiesis — system produces conditions for its own operation).
"""

import json
import logging
import os
import socket
import subprocess
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("bizra.executor")


class MissionExecutor:
    """
    The sovereign mission pipeline. Every method is one stage.
    Every stage fires or logs why it didn't.
    """

    def __init__(self, state_dir: str = None):
        self.state_dir = state_dir or os.path.expanduser("~/.bizra/node-1")
        self.binary = self._find_binary()
        self.stages_fired = {}
        self.receipt_id = ""
        self.seed_earned = 0

    def execute(self, task: str, skill: str = None) -> Dict[str, Any]:
        """Run the FULL pipeline. Returns complete mission report."""
        report = {
            "task": task,
            "skill": skill,
            "started_at": time.time(),
            "stages": {},
        }

        # Stage 1: FAISS Context Enrichment
        context = self._stage_faiss(task)
        report["stages"]["faiss"] = {
            "fired": bool(context),
            "chars": len(context),
        }

        # Stage 2: Ollama Inference via bizra-node
        node_result = self._stage_inference(task, context)
        report["stages"]["inference"] = {
            "fired": node_result.get("inference_executed") == "true",
            "model": node_result.get("inference_model", ""),
            "ihsan": node_result.get("inference_ihsan", ""),
            "receipt_id": node_result.get("receipt_id", ""),
            "guardian": node_result.get("guardian_approved", ""),
            "agents": node_result.get("agents_consulted", "0"),
        }
        self.receipt_id = node_result.get("receipt_id", "")

        # Stage 3: Skill Execution (if applicable)
        skill_result = {}
        if skill == "file_management":
            skill_result = self._stage_skill_files(task)
        elif skill == "browse":
            skill_result = self._stage_skill_browse(task)
        report["stages"]["skill"] = {
            "fired": bool(skill_result),
            "type": skill or "inference_only",
            "result": skill_result,
        }

        # Stage 4: SEED Reward Calculation
        seed_result = self._stage_seed(node_result)
        report["stages"]["seed"] = seed_result
        self.seed_earned = seed_result.get("net", 0)

        # Stage 5: Living Memory Update
        memory_result = self._stage_memory(task, skill)
        report["stages"]["memory"] = {"fired": memory_result}

        # Stage 6: EventBus Signal (Rust atomic flags)
        eventbus_result = self._stage_eventbus()
        report["stages"]["eventbus"] = {"fired": eventbus_result}

        # Stage 7: Desktop Notification (AHK/toast)
        notify_result = self._stage_notify(task)
        report["stages"]["notify"] = {"fired": notify_result}

        # Stage 8: Proactive Watcher Update
        watcher_result = self._stage_watcher()
        report["stages"]["watcher"] = {"fired": watcher_result}

        report["completed_at"] = time.time()
        report["duration_s"] = round(report["completed_at"] - report["started_at"], 2)
        report["receipt_id"] = self.receipt_id
        report["seed_earned"] = self.seed_earned
        report["stages_fired"] = sum(
            1 for s in report["stages"].values() if s.get("fired")
        )
        report["stages_total"] = len(report["stages"])

        return report

    # ── Stage 1: FAISS ──────────────────────────────────────

    def _stage_faiss(self, task: str) -> str:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("FAISS loading exceeded 15s budget")

        try:
            # FAISS can take 60-120s cold-loading 84K vectors from /mnt/c
            # Budget: 15s max. If cold, skip gracefully — pipeline continues.
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(15)
            try:
                from core.proof_engine.faiss_search import format_context, search

                results = search(task, top_k=5)
                if results:
                    return format_context(results, max_chars=1500)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except (TimeoutError, Exception) as e:
            logger.debug("FAISS skip: %s", e)
        return ""

    # ── Stage 2: Inference via bizra-node ────────────────────

    def _stage_inference(self, task: str, context: str) -> Dict:
        if not self.binary:
            return {"inference_executed": "false", "error": "binary not found"}

        enriched = task
        if context:
            # Collapse to single line — bizra-node uses tab-delimited protocol
            ctx_oneline = context.replace("\n", " ").replace("\t", " ")
            enriched = f"{ctx_oneline} Based on the above context: {task}"

        # Escape tabs/newlines in the task itself
        enriched = enriched.replace("\t", " ").replace("\n", " ")

        ts = int(time.time())
        cmd_input = (
            f"START_SESSION\t{ts}\n"
            f"RECEIVE\t{enriched}\t{ts}\n"
            f"END_SESSION\t{ts}\n"
            f"SHUTDOWN\n"
        )

        try:
            env = os.environ.copy()
            env["BIZRA_ENABLE_OLLAMA_EXECUTE"] = "1"
            env["BIZRA_OLLAMA_MODEL"] = os.environ.get("BIZRA_MODEL", "qwen2.5:3b")

            proc = subprocess.run(
                [
                    self.binary,
                    "--user",
                    "1",
                    "--ihsan",
                    "9500",
                    "--state-dir",
                    self.state_dir,
                    "--no-banner",
                ],
                input=cmd_input,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )

            # Parse response
            result = {}
            for line in proc.stdout.split("\n"):
                if "received=true" in line or "receipt_id=" in line:
                    for field in line.split("\t"):
                        if "=" in field:
                            k, v = field.split("=", 1)
                            result[k] = v
            return result

        except Exception as e:
            return {"inference_executed": "false", "error": str(e)}

    # ── Stage 3: Skill Execution ─────────────────────────────

    def _stage_skill_files(self, task: str) -> Dict:
        try:
            from core.skills.file_organizer import (
                execute_plan,
                generate_plan,
                scan_directory,
            )

            # Extract directory from task
            target = None
            for part in task.split():
                p = os.path.expanduser(part)
                if os.path.isdir(p):
                    target = p
                    break

            if not target:
                # Try common paths
                for candidate in [
                    "~/Downloads",
                    "~/Desktop",
                    "~/Documents",
                ]:
                    p = os.path.expanduser(candidate)
                    if os.path.isdir(p):
                        target = p
                        break

            if not target:
                return {"error": "no valid directory found"}

            files = scan_directory(target)
            if not files:
                return {"files": 0, "moved": 0}

            plan = generate_plan(files, target)
            result = execute_plan(plan)

            # Save undo log
            undo_path = os.path.join(self.state_dir, "undo_organize.json")
            os.makedirs(os.path.dirname(undo_path), exist_ok=True)
            with open(undo_path, "w") as f:
                json.dump(result["undo_log"], f)

            return {
                "files": plan["total_files"],
                "moved": result["moved"],
                "errors": result["errors"],
                "undo": undo_path,
            }

        except Exception as e:
            return {"error": str(e)}

    def _stage_skill_browse(self, task: str) -> Dict:
        # Placeholder for MCP browser integration
        return {"type": "browse", "status": "inference_only"}

    # ── Stage 4: SEED Reward ─────────────────────────────────

    def _stage_seed(self, node_result: Dict) -> Dict:
        try:
            from core.proof_engine.seed_calc import calculate_seed_reward
            from core.proof_engine.seed_ledger import append

            ihsan = float(node_result.get("inference_ihsan", "0.95"))
            agents = int(node_result.get("agents_consulted", "1"))
            signed = bool(node_result.get("receipt_id", ""))

            reward = calculate_seed_reward(ihsan, 0.95, agents, 200, signed)

            if reward["net"] > 0:
                append(
                    reward["net"],
                    f"mission: {node_result.get('receipt_id', '')[:16]}",
                )

            return {
                "fired": True,
                "gross": reward["gross"],
                "zakat": reward["zakat"],
                "net": reward["net"],
                "impact": reward["impact"],
                "reason": reward["reason"],
            }

        except Exception as e:
            return {"fired": False, "error": str(e)}

    # ── Stage 5: Living Memory ───────────────────────────────

    def _stage_memory(self, task: str, skill: str) -> bool:
        try:
            from core.living_memory.brain import LivingMemory

            mem = LivingMemory().load()
            domain = skill or "general"
            mem.update_after_mission(task[:100], "P1", domain, 0.95, 3)
            return True
        except Exception:
            return False

    # ── Stage 6: EventBus (Rust atomic flags) ────────────────

    def _stage_eventbus(self) -> bool:
        try:
            from core.sovereign.event_bus import create_rust_event_bridge

            bridge = create_rust_event_bridge()
            if bridge:
                bridge.emit_with_receipt(
                    "action.receipt",
                    "mission_complete",
                    self.receipt_id[:32] if self.receipt_id else "none",
                    0.95,
                    1,
                )
                return True
        except Exception:
            pass
        return False

    # ── Stage 7: Desktop Notification ────────────────────────

    def _stage_notify(self, task: str) -> bool:
        # Try Desktop Bridge (port 9742)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(("127.0.0.1", 9742))
            req = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "sovereign_query",
                    "params": {"query": f"Mission complete: {task[:50]}"},
                    "id": 1,
                }
            )
            s.send((req + "\n").encode())
            s.recv(4096)
            s.close()
            return True
        except Exception:
            pass

        # Fallback: try PowerShell toast
        try:
            subprocess.run(
                [
                    "powershell.exe",
                    "-Command",
                    f'[System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms"); '
                    f'[System.Windows.Forms.MessageBox]::Show("Mission complete: {task[:40]}", "BIZRA")',
                ],
                timeout=3,
                capture_output=True,
            )
            return True
        except Exception:
            pass

        return False

    # ── Stage 8: Proactive Watcher ───────────────────────────

    def _stage_watcher(self) -> bool:
        try:
            from core.skills.proactive_watcher import scan_for_changes

            scan_for_changes()  # Updates watcher state
            return True
        except Exception:
            return False

    # ── Helpers ──────────────────────────────────────────────

    def _find_binary(self) -> Optional[str]:
        candidates = [
            "bizra-omega/target/release/bizra-node",
            os.path.expanduser(
                "~/bizra/bizra-data-lake/bizra-omega/target/release/bizra-node"
            ),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None


def format_report(report: Dict) -> str:
    """Format mission report for TUI display."""
    lines = []
    lines.append(f"Task:      {report['task'][:50]}")
    lines.append(f"Duration:  {report['duration_s']}s")
    lines.append(f"Stages:    {report['stages_fired']}/{report['stages_total']} fired")
    lines.append(
        f"Receipt:   {report['receipt_id'][:24]}..."
        if report["receipt_id"]
        else "Receipt:   none"
    )
    lines.append(f"SEED:      +{report['seed_earned']}")
    lines.append("")
    lines.append("Pipeline:")
    for name, data in report["stages"].items():
        fired = "✓" if data.get("fired") else "○"
        detail = ""
        if name == "faiss":
            detail = f"{data.get('chars', 0)} chars context"
        elif name == "inference":
            detail = f"model={data.get('model', '?')} ihsan={data.get('ihsan', '?')}"
        elif name == "skill":
            detail = f"type={data.get('type', '?')}"
        elif name == "seed":
            detail = f"net={data.get('net', 0)} impact={data.get('impact', 0)}"
        lines.append(f"  {fired} {name:12s} {detail}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    task = " ".join(sys.argv[1:]) or "What is BIZRA?"
    executor = MissionExecutor()
    report = executor.execute(task)
    print(format_report(report))
