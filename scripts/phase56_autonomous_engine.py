#!/usr/bin/env python3
"""Phase 56 Autonomous Security Engine.

Graph-of-thought execution model for security hardening regression checks.
Computes a weighted SNR score and enforces a hard threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckNode:
    node_id: str
    label: str
    domain: str
    severity: str
    weight: float = 0.0
    deps: list[str] = field(default_factory=list)
    command: list[str] | None = None
    cwd: str | None = None
    status: str = "pending"
    duration_seconds: float = 0.0
    return_code: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    fallback_used: bool = False


def build_graph(root_dir: Path) -> dict[str, CheckNode]:
    python = sys.executable
    pytest = [python, "-m", "pytest", "-q"]

    graph = {
        "rust_api": CheckNode(
            node_id="rust_api",
            label="Rust API security hardening",
            domain="rust",
            severity="critical",
            weight=0.24,
            command=["cargo", "test", "-p", "bizra-api", "--locked"],
            cwd=str(root_dir / "bizra-omega"),
        ),
        "auth_guardrails": CheckNode(
            node_id="auth_guardrails",
            label="Auth middleware + command guardrails",
            domain="python-core",
            severity="high",
            weight=0.16,
            command=[
                *pytest,
                "tests/core/auth/test_middleware_fail_closed.py",
                "tests/core/benchmark/test_guardrails_whitespace.py",
                "tests/core/sovereign/test_tiered_verification_whitespace.py",
            ],
            cwd=str(root_dir),
        ),
        "zpk_rlm": CheckNode(
            node_id="zpk_rlm",
            label="ZPK + RLM sandbox hardening",
            domain="python-core",
            severity="high",
            weight=0.16,
            deps=["auth_guardrails"],
            command=[
                *pytest,
                "tests/core/zpk/test_zero_point_kernel.py",
                "tests/core/inference/test_rlm_sandbox.py",
            ],
            cwd=str(root_dir),
        ),
        "transport": CheckNode(
            node_id="transport",
            label="Transport identity binding (Noise/DTLS)",
            domain="federation",
            severity="critical",
            weight=0.24,
            command=[
                *pytest,
                "tests/core/federation/test_secure_transport.py",
            ],
            cwd=str(root_dir),
        ),
        "bridge_infra": CheckNode(
            node_id="bridge_infra",
            label="Bridge + medium + infra hardening",
            domain="integration",
            severity="high",
            weight=0.20,
            deps=["rust_api", "zpk_rlm"],
            command=[
                *pytest,
                "tests/integration/test_phase56_medium_hardening.py",
                "tests/integration/test_phase56_bridge_hardening.py",
                "tests/integration/test_phase56_infra_hardening.py",
            ],
            cwd=str(root_dir),
        ),
        "masterpiece": CheckNode(
            node_id="masterpiece",
            label="Ultimate hardening aggregate",
            domain="governance",
            severity="critical",
            deps=[
                "rust_api",
                "auth_guardrails",
                "zpk_rlm",
                "transport",
                "bridge_infra",
            ],
        ),
    }

    return graph


def _walk(
    node_id: str, graph: dict[str, CheckNode], visiting: set[str], visited: set[str]
) -> None:
    if node_id in visited:
        return
    if node_id in visiting:
        raise ValueError(f"Cycle detected at node '{node_id}'")
    visiting.add(node_id)
    node = graph[node_id]
    for dep in node.deps:
        if dep not in graph:
            raise ValueError(f"Node '{node_id}' depends on unknown node '{dep}'")
        _walk(dep, graph, visiting, visited)
    visiting.remove(node_id)
    visited.add(node_id)


def validate_graph(graph: dict[str, CheckNode]) -> None:
    visited: set[str] = set()
    for node_id in graph:
        _walk(node_id, graph, set(), visited)

    total_weight = sum(
        node.weight for node in graph.values() if node.command is not None
    )
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            f"Executable node weights must sum to 1.0, got {total_weight:.6f}"
        )


def _tail(text: str, max_lines: int = 60) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _run_command(
    node: CheckNode, allow_offline_cargo: bool
) -> tuple[int, str, str, bool]:
    assert node.command is not None
    process = subprocess.run(
        node.command,
        cwd=node.cwd,
        text=True,
        capture_output=True,
    )

    if process.returncode == 0:
        return process.returncode, process.stdout, process.stderr, False

    if allow_offline_cargo and node.command[:4] == ["cargo", "test", "-p", "bizra-api"]:
        fallback_cmd = ["cargo", "test", "-p", "bizra-api", "--offline"]
        fallback = subprocess.run(
            fallback_cmd,
            cwd=node.cwd,
            text=True,
            capture_output=True,
        )
        merged_stdout = process.stdout + "\n[offline fallback]\n" + fallback.stdout
        merged_stderr = process.stderr + "\n[offline fallback]\n" + fallback.stderr
        return fallback.returncode, merged_stdout, merged_stderr, True

    return process.returncode, process.stdout, process.stderr, False


def _deps_succeeded(node: CheckNode, graph: dict[str, CheckNode]) -> bool:
    return all(graph[dep].status == "passed" for dep in node.deps)


def run_graph(graph: dict[str, CheckNode], allow_offline_cargo: bool) -> None:
    print("[phase56-engine] Standing on the shoulders of giants.")
    print("[phase56-engine] Executing graph-of-thought hardening plan...")

    remaining = set(graph.keys())
    while remaining:
        progressed = False
        for node_id in list(remaining):
            node = graph[node_id]

            if any(graph[dep].status in ("failed", "blocked") for dep in node.deps):
                node.status = "blocked"
                remaining.remove(node_id)
                progressed = True
                print(f"[phase56-engine] {node.label}: BLOCKED")
                continue

            if not all(graph[dep].status in ("passed", "skipped") for dep in node.deps):
                continue

            if node.command is None:
                node.status = "passed"
                remaining.remove(node_id)
                progressed = True
                print(f"[phase56-engine] {node.label}: PASS (aggregate)")
                continue

            start = time.monotonic()
            print(f"[phase56-engine] {node.label}: RUN")
            return_code, stdout, stderr, fallback_used = _run_command(
                node, allow_offline_cargo=allow_offline_cargo
            )
            node.duration_seconds = time.monotonic() - start
            node.return_code = return_code
            node.stdout_tail = _tail(stdout)
            node.stderr_tail = _tail(stderr)
            node.fallback_used = fallback_used
            node.status = "passed" if return_code == 0 else "failed"

            remaining.remove(node_id)
            progressed = True
            state = "PASS" if node.status == "passed" else "FAIL"
            print(
                f"[phase56-engine] {node.label}: {state} ({node.duration_seconds:.2f}s)"
            )

        if not progressed:
            unresolved = ", ".join(sorted(remaining))
            raise RuntimeError(
                f"Could not resolve graph execution for nodes: {unresolved}"
            )


def calculate_snr(graph: dict[str, CheckNode]) -> tuple[float, float, float]:
    signal = sum(
        node.weight
        for node in graph.values()
        if node.command is not None and node.status == "passed"
    )
    noise = sum(
        node.weight
        for node in graph.values()
        if node.command is not None and node.status in ("failed", "blocked")
    )
    denominator = signal + noise
    snr = signal / denominator if denominator > 0 else 0.0
    return signal, noise, snr


def _render_markdown(
    graph: dict[str, CheckNode],
    signal: float,
    noise: float,
    snr: float,
    threshold: float,
) -> str:
    lines = [
        "# Phase56 Autonomous Security Engine Report",
        "",
        "| Node | Domain | Severity | Weight | Status | Duration(s) |",
        "|------|--------|----------|--------|--------|-------------|",
    ]
    for node in graph.values():
        if node.command is None:
            continue
        lines.append(
            f"| `{node.node_id}` | {node.domain} | {node.severity} | {node.weight:.2f} | {node.status.upper()} | {node.duration_seconds:.2f} |"
        )

    lines.extend(
        [
            "",
            f"- Signal: `{signal:.2f}`",
            f"- Noise: `{noise:.2f}`",
            f"- SNR: `{snr:.4f}`",
            f"- Threshold: `{threshold:.4f}`",
            f"- Verdict: `{'PASS' if snr >= threshold else 'FAIL'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_reports(
    graph: dict[str, CheckNode],
    signal: float,
    noise: float,
    snr: float,
    threshold: float,
    report_json: Path,
    report_md: Path,
) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "signal": signal,
        "noise": noise,
        "snr": snr,
        "threshold": threshold,
        "passed": snr >= threshold,
        "nodes": [
            {
                "node_id": node.node_id,
                "label": node.label,
                "domain": node.domain,
                "severity": node.severity,
                "weight": node.weight,
                "deps": node.deps,
                "status": node.status,
                "duration_seconds": node.duration_seconds,
                "return_code": node.return_code,
                "fallback_used": node.fallback_used,
                "stdout_tail": node.stdout_tail,
                "stderr_tail": node.stderr_tail,
            }
            for node in graph.values()
            if node.command is not None
        ],
    }

    report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        _render_markdown(graph, signal, noise, snr, threshold), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 56 graph-of-thought autonomous security engine."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root path (default: current directory).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum SNR score to pass (default: 0.95).",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/phase56/phase56_engine_report.json",
        help="Path to JSON report output.",
    )
    parser.add_argument(
        "--report-md",
        default="artifacts/phase56/phase56_engine_report.md",
        help="Path to Markdown report output.",
    )
    parser.add_argument(
        "--allow-offline-cargo",
        action="store_true",
        default=False,
        help="Allow fallback to cargo --offline for the Rust node.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root).resolve()

    graph = build_graph(root_dir)
    validate_graph(graph)
    run_graph(graph, allow_offline_cargo=args.allow_offline_cargo)
    signal, noise, snr = calculate_snr(graph)

    report_json = (root_dir / args.report_json).resolve()
    report_md = (root_dir / args.report_md).resolve()
    _write_reports(
        graph=graph,
        signal=signal,
        noise=noise,
        snr=snr,
        threshold=args.threshold,
        report_json=report_json,
        report_md=report_md,
    )

    summary = _render_markdown(graph, signal, noise, snr, args.threshold)
    print(summary)

    github_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_summary:
        with Path(github_summary).open("a", encoding="utf-8") as file:
            file.write("\n")
            file.write(summary)

    if snr < args.threshold:
        print(
            f"[phase56-engine] FAIL: SNR {snr:.4f} below threshold {args.threshold:.4f}",
            file=sys.stderr,
        )
        return 1

    print(f"[phase56-engine] PASS: SNR {snr:.4f} >= threshold {args.threshold:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
