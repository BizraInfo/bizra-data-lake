#!/usr/bin/env python3
"""
BIZRA Context Injection Hook (UserPromptSubmit Hook)
Adds relevant BIZRA system context based on the user's prompt
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def get_service_status() -> str:
    """Get Docker service status if available"""
    try:
        import subprocess

        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            services = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    service = json.loads(line)
                    services.append(
                        f"  - {service.get('Service', 'unknown')}: {service.get('State', 'unknown')}"
                    )
            if services:
                return "Docker Services:\n" + "\n".join(services)
    except Exception:
        pass
    return ""


def get_recent_receipts() -> str:
    """Get info about recent receipts"""
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if not project_dir:
        return ""

    receipts_dir = Path(project_dir) / "docs" / "evidence" / "receipts"
    if not receipts_dir.exists():
        return ""

    try:
        receipt_files = list(receipts_dir.glob("*.json")) + list(
            receipts_dir.glob("*.jsonl")
        )
        if receipt_files:
            # Sort by modification time
            receipt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            count = len(receipt_files)
            latest = receipt_files[0].name if receipt_files else "none"
            return f"Receipts: {count} total, latest: {latest}"
    except Exception:
        pass
    return ""


def check_prompt_patterns(prompt: str) -> tuple[bool, str, list[str]]:
    """
    Check if prompt matches specific patterns that need context
    Returns: (should_add_context, context_type, context_items)
    """
    prompt_lower = prompt.lower()

    # Architecture/design queries
    if any(
        keyword in prompt_lower
        for keyword in ["architecture", "design", "how does", "explain the"]
    ):
        return True, "architecture", [
            "BIZRA Architecture Overview:",
            "- Dual implementation: Rust (port 8080) + Python (port 8010)",
            "- PAT: 7 specialized execution agents",
            "- SAT: 5 guardian validation agents",
            "- Request flow: User → SAT Validation (3/5 consensus) → PAT Execution → SAT Evaluation → Response",
        ]

    # Build/deployment queries
    if any(
        keyword in prompt_lower
        for keyword in ["build", "compile", "deploy", "docker", "cargo"]
    ):
        return True, "build", [
            "Build Commands:",
            "- Rust: cargo build --release && cargo test",
            "- Python: pip install -r requirements-kernel.txt",
            "- Docker: docker compose up -d",
            "- Full stack: docker compose up -d (starts all 7 services)",
        ]

    # Testing queries
    if any(keyword in prompt_lower for keyword in ["test", "pytest", "cargo test"]):
        return True, "testing", [
            "Testing:",
            "- Rust: cargo test (all) or cargo test --test pat_sat_runtime_tests",
            "- Python: pytest tests/ or pytest tests/test_specific.py",
            "- Integration: docker compose -f docker-compose.test.yml up",
        ]

    # Validation/gates queries
    if any(
        keyword in prompt_lower
        for keyword in ["ihsan", "ihsān", "sape", "fate", "validation", "receipt"]
    ):
        return True, "validation", [
            "BIZRA Validation Gates:",
            "- Ihsān Score: ≥0.95 threshold (constitution/ihsan_v1.yaml)",
            "- SAPE: 9-probe verification system (sequential ~900ms, parallel ~300ms)",
            "- FATE: Fail-Safe Agentic Trust Escalation (Low→Medium→High→Critical)",
            "- Receipts: All decisions emit structured receipts to docs/evidence/receipts/",
            "- Fail-Closed: Never proceed without SAT approval",
        ]

    # Service/infrastructure queries
    if any(
        keyword in prompt_lower
        for keyword in ["service", "redis", "postgres", "neo4j", "port"]
    ):
        return True, "services", [
            "Service Architecture:",
            "- elite (Rust): 8080 - PAT/SAT/SAPE/FATE engine",
            "- kernel (Python): 8010 - SAPE planning/FATE/LLM routing",
            "- postgres: 5432 - Knowledge graph + pgvector",
            "- synapse (Redis): 6379 (TLS) - State/receipts/FATE escalations",
            "- wisdom (Neo4j): 7474/7687 - Graph evidence for SAPE",
            "- vectors (ChromaDB): 8001 - Embeddings",
        ]

    return False, "", []


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    prompt = input_data.get("prompt", "")

    # Check if we should add context
    should_add, context_type, context_items = check_prompt_patterns(prompt)

    if not should_add:
        # No relevant context to add
        sys.exit(0)

    # Build context message
    context_parts = [f"\n--- BIZRA System Context ({context_type}) ---"]
    context_parts.extend(context_items)

    # Add service status if relevant
    if context_type in ["services", "build"]:
        service_status = get_service_status()
        if service_status:
            context_parts.append("")
            context_parts.append(service_status)

    # Add receipt status if relevant
    if context_type == "validation":
        receipt_info = get_recent_receipts()
        if receipt_info:
            context_parts.append("")
            context_parts.append(receipt_info)

    context_parts.append(
        f"\n(Context auto-injected at {datetime.now().strftime('%H:%M:%S')})"
    )
    context_parts.append("---\n")

    # Output context (will be added to conversation)
    print("\n".join(context_parts))

    sys.exit(0)


if __name__ == "__main__":
    main()
