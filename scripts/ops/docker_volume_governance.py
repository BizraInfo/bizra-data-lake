#!/usr/bin/env python3
"""
Docker Volume Governance — BIZRA Node0 DevOps

Purpose:
1) Inventory Docker volumes by owner and category.
2) Identify reclaimable orphan volumes.
3) Reclaim k3d/containerd image cache safely with dry-run + confirmation.
4) Emit JSON evidence reports for every run.

Usage:
    python scripts/ops/docker_volume_governance.py inventory
    python scripts/ops/docker_volume_governance.py orphans
    python scripts/ops/docker_volume_governance.py --dry-run reclaim-k3d
    python scripts/ops/docker_volume_governance.py reclaim-k3d --yes --restart-cluster
    python scripts/ops/docker_volume_governance.py reclaim-all --yes
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class VolumeInfo:
    name: str
    size_bytes: int
    size_human: str
    links: int
    created_at: str
    category: str
    owner_containers: list[str]
    owner_mounts: list[str]
    is_anonymous: bool
    is_k3d: bool


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _run_text(cmd: list[str], check: bool = True) -> str:
    return _run(cmd, check=check).stdout.strip()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _log_dir() -> Path:
    windows = Path("C:/BIZRA-DATA-LAKE/logs")
    if windows.exists() or sys.platform == "win32":
        windows.mkdir(parents=True, exist_ok=True)
        return windows

    wsl = Path("/mnt/c/BIZRA-DATA-LAKE/logs")
    if wsl.exists():
        wsl.mkdir(parents=True, exist_ok=True)
        return wsl

    fallback = _project_root() / "logs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _confirm_or_die(prompt: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    answer = input(f"{prompt} (type YES to continue): ").strip()
    if answer != "YES":
        print("Aborted by operator.")
        raise SystemExit(1)


def _parse_size_bytes(size: str) -> int:
    size = size.strip()
    if size in {"", "0B", "-1B"}:
        return 0

    m = re.match(r"^(-?\d+(?:\.\d+)?)([A-Za-z]+)$", size)
    if not m:
        return 0

    value = float(m.group(1))
    unit = m.group(2).upper()
    mult = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
    }.get(unit)
    if mult is None:
        return 0
    return int(value * mult)


def _human_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} EB"


def _docker_available() -> bool:
    proc = _run(["docker", "info"], check=False)
    return proc.returncode == 0


def _parse_volume_size_table(df_verbose: str) -> dict[str, tuple[int, str, int]]:
    """Parse `docker system df -v` local volume section.

    Returns: volume_name -> (size_bytes, size_raw, links)
    """
    in_local_volumes = False
    out: dict[str, tuple[int, str, int]] = {}

    for raw_line in df_verbose.splitlines():
        line = raw_line.strip()

        if raw_line.startswith("Local Volumes space usage"):
            in_local_volumes = False
            continue
        if raw_line.startswith("VOLUME NAME"):
            in_local_volumes = True
            continue
        if raw_line.startswith("Build cache usage"):
            break
        if not in_local_volumes or not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        name = parts[0]
        try:
            links = int(parts[1])
        except ValueError:
            links = 0
        size_raw = "".join(parts[2:])
        out[name] = (_parse_size_bytes(size_raw), size_raw, links)

    return out


def _container_mount_index() -> dict[str, list[tuple[str, str]]]:
    """Build volume -> [(container_name, destination)] index from one inspect pass."""
    ids_raw = _run_text(["docker", "ps", "-a", "--format", "{{.ID}}"], check=False)
    ids = [c.strip() for c in ids_raw.splitlines() if c.strip()]
    if not ids:
        return {}

    inspect = _run_text(["docker", "inspect", *ids], check=False)
    if not inspect:
        return {}

    try:
        data = json.loads(inspect)
    except json.JSONDecodeError:
        return {}

    out: dict[str, list[tuple[str, str]]] = {}
    for container in data:
        name = str(container.get("Name", "")).lstrip("/") or "unknown"
        mounts = container.get("Mounts", []) or []
        for mount in mounts:
            if mount.get("Type") != "volume":
                continue
            vol_name = mount.get("Name")
            if not vol_name:
                continue
            out.setdefault(vol_name, []).append(
                (name, str(mount.get("Destination", "")))
            )
    return out


def _classify_volume(
    name: str, links: int, owners: list[str], is_anonymous: bool
) -> str:
    owner_blob = " ".join(owners).lower()
    name_l = name.lower()

    if "k3d-" in name_l or "k3d-" in owner_blob:
        return "k3d-cluster"
    if "bizra-dual" in name_l or "bizra-dual" in owner_blob:
        return "bizra-stack"
    if "bizra-node0" in name_l or "bizra-node0" in owner_blob:
        return "bizra-node0"
    if "bizra-unified" in name_l:
        return "bizra-unified"
    if "extension" in name_l or "desktop-extension" in owner_blob:
        return "docker-extension"
    if links == 0:
        return "orphan"
    if is_anonymous:
        return "anonymous-active"
    return "other"


def _collect_volumes() -> list[VolumeInfo]:
    names_raw = _run_text(
        ["docker", "volume", "ls", "--format", "{{.Name}}"], check=False
    )
    names = [v.strip() for v in names_raw.splitlines() if v.strip()]

    if not names:
        return []

    size_map = _parse_volume_size_table(
        _run_text(["docker", "system", "df", "-v"], check=False)
    )
    mount_map = _container_mount_index()

    inspect_raw = _run_text(["docker", "volume", "inspect", *names], check=False)
    inspect_map: dict[str, dict[str, Any]] = {}
    if inspect_raw:
        try:
            for rec in json.loads(inspect_raw):
                inspect_map[str(rec.get("Name", ""))] = rec
        except json.JSONDecodeError:
            pass

    out: list[VolumeInfo] = []
    for name in names:
        size_bytes, _, links = size_map.get(name, (0, "0B", 0))
        owners_and_mounts = mount_map.get(name, [])
        owners = [o for o, _ in owners_and_mounts]
        mounts = [m for _, m in owners_and_mounts]
        created_at = str(inspect_map.get(name, {}).get("CreatedAt", "unknown"))
        is_anonymous = len(name) == 64 and all(
            c in "0123456789abcdef" for c in name.lower()
        )
        is_k3d = any("k3d-" in o.lower() for o in owners) or "k3d-" in name.lower()
        category = _classify_volume(name, links, owners, is_anonymous)

        out.append(
            VolumeInfo(
                name=name,
                size_bytes=size_bytes,
                size_human=_human_size(size_bytes),
                links=links,
                created_at=created_at,
                category=category,
                owner_containers=owners,
                owner_mounts=mounts,
                is_anonymous=is_anonymous,
                is_k3d=is_k3d,
            )
        )

    out.sort(key=lambda v: v.size_bytes, reverse=True)
    return out


def _write_report(command: str, dry_run: bool, payload: dict[str, Any]) -> Path:
    now = datetime.now(timezone.utc)
    report = {
        "timestamp_utc": now.isoformat(),
        "command": command,
        "dry_run": dry_run,
        **payload,
    }

    report_path = (
        _log_dir() / f"docker_volume_governance_{now.strftime('%Y%m%d_%H%M%S')}.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report_path


def cmd_inventory(top: int, dry_run: bool) -> int:
    volumes = _collect_volumes()
    total_bytes = sum(v.size_bytes for v in volumes)

    by_category: dict[str, dict[str, int]] = {}
    for v in volumes:
        stat = by_category.setdefault(v.category, {"count": 0, "bytes": 0})
        stat["count"] += 1
        stat["bytes"] += v.size_bytes

    print("\n=== Docker Volume Inventory ===")
    print(f"Total volumes: {len(volumes)}")
    print(f"Total size: {_human_size(total_bytes)}")

    print("\nBy category:")
    for cat, stat in sorted(
        by_category.items(), key=lambda i: i[1]["bytes"], reverse=True
    ):
        print(f"  - {cat:<18} {stat['count']:>3} volumes  {_human_size(stat['bytes'])}")

    print(f"\nTop {max(top, 1)} volumes by size:")
    for v in volumes[: max(top, 1)]:
        owner = v.owner_containers[0] if v.owner_containers else "<unlinked>"
        print(f"  - {v.name:<64} {v.size_human:>10}  links={v.links:<2} owner={owner}")

    report = _write_report(
        command="inventory",
        dry_run=dry_run,
        payload={
            "total_volumes": len(volumes),
            "total_size_bytes": total_bytes,
            "categories": by_category,
            "top_volumes": [asdict(v) for v in volumes[: max(top, 1)]],
        },
    )
    print(f"\nReport written to: {report}")
    return 0


def cmd_orphans(top: int, dry_run: bool) -> int:
    volumes = _collect_volumes()
    orphans = [v for v in volumes if v.links == 0]
    total = sum(v.size_bytes for v in orphans)

    print("\n=== Orphan Volume Analysis ===")
    print(f"Orphan volumes (links=0): {len(orphans)}")
    print(f"Potential reclaimable: {_human_size(total)}")

    print(f"\nTop {max(top, 1)} orphan volumes:")
    for v in orphans[: max(top, 1)]:
        print(f"  - {v.name:<64} {v.size_human:>10}  category={v.category}")

    report = _write_report(
        command="orphans",
        dry_run=dry_run,
        payload={
            "orphans": len(orphans),
            "reclaimable_bytes": total,
            "top_orphans": [asdict(v) for v in orphans[: max(top, 1)]],
        },
    )
    print(f"\nReport written to: {report}")
    return 0


def _k3d_clusters() -> list[dict[str, Any]]:
    out = _run_text(["k3d", "cluster", "list", "-o", "json"], check=False)
    if not out:
        return []
    try:
        parsed = json.loads(out)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


def _running_container_names() -> set[str]:
    out = _run_text(["docker", "ps", "--format", "{{.Names}}"], check=False)
    return {line.strip() for line in out.splitlines() if line.strip()}


def cmd_reclaim_k3d(dry_run: bool, assume_yes: bool, restart_cluster: bool) -> int:
    clusters = _k3d_clusters()
    if not clusters:
        print("No k3d clusters found or k3d unavailable.")
        return 1

    running = _running_container_names()
    executed_steps: list[str] = []
    failures: list[dict[str, Any]] = []

    print("\n=== k3d Cache Reclaim Plan ===")
    for c in clusters:
        name = str(c.get("name", "unknown"))
        print(f"Cluster: {name}")
        node_candidates = [
            f"k3d-{name}-server-0",
            f"k3d-{name}-agent-0",
            f"k3d-{name}-agent-1",
        ]
        nodes = [n for n in node_candidates if n in running]
        if not nodes:
            print(
                "  - No running nodes detected; start cluster first if reclaim needed."
            )
            continue

        if dry_run:
            for node in nodes:
                print(
                    f'  - [DRY-RUN] docker exec {node} sh -lc "crictl rmi --prune (with k3s fallback)"'
                )
            if restart_cluster:
                print(f"  - [DRY-RUN] k3d cluster stop {name}")
                print(f"  - [DRY-RUN] k3d cluster start {name}")
            continue

        _confirm_or_die(
            f"This will prune containerd image cache for cluster '{name}'",
            assume_yes,
        )

        for node in nodes:
            prune_script = (
                "if command -v crictl >/dev/null 2>&1; then "
                "crictl rmi --prune; "
                "elif [ -x /bin/crictl ]; then "
                "/bin/crictl rmi --prune; "
                "elif command -v k3s >/dev/null 2>&1; then "
                "k3s crictl rmi --prune; "
                "else "
                "echo 'crictl not found in node runtime' >&2; "
                "exit 127; "
                "fi"
            )
            cmd = [
                "docker",
                "exec",
                node,
                "sh",
                "-lc",
                prune_script,
            ]
            proc = _run(cmd, check=False)
            executed_steps.append(" ".join(cmd))
            if proc.returncode == 0:
                print(f"  - Pruned image cache on {node}")
            else:
                print(f"  - FAILED prune on {node} (exit={proc.returncode})")
                if proc.stderr.strip():
                    print(f"    stderr: {proc.stderr.strip()}")
                failures.append(
                    {
                        "node": node,
                        "exit_code": proc.returncode,
                        "stderr": proc.stderr,
                        "stdout": proc.stdout,
                    }
                )

        if restart_cluster:
            for cmd in (
                ["k3d", "cluster", "stop", name],
                ["k3d", "cluster", "start", name],
            ):
                _run(cmd, check=False)
                executed_steps.append(" ".join(cmd))
            print("  - Cluster restart sequence completed")

    report = _write_report(
        command="reclaim-k3d",
        dry_run=dry_run,
        payload={
            "clusters": [c.get("name", "unknown") for c in clusters],
            "restart_cluster": restart_cluster,
            "executed_steps": executed_steps,
            "failures": failures,
        },
    )
    print(f"\nReport written to: {report}")
    return 0 if not failures else 1


def cmd_reclaim_all(dry_run: bool, assume_yes: bool) -> int:
    dangling_raw = _run_text(
        ["docker", "volume", "ls", "-q", "--filter", "dangling=true"],
        check=False,
    )
    dangling = [v.strip() for v in dangling_raw.splitlines() if v.strip()]

    print("\n=== Reclaim All Orphan Volumes ===")
    print(f"Dangling volumes: {len(dangling)}")

    if dry_run:
        for v in dangling:
            print(f"  - [DRY-RUN] would remove {v}")
        report = _write_report(
            command="reclaim-all",
            dry_run=True,
            payload={"dangling_count": len(dangling), "dangling_volumes": dangling},
        )
        print(f"\nReport written to: {report}")
        return 0

    if not dangling:
        print("No dangling volumes found.")
        return 0

    _confirm_or_die(
        "This will run 'docker volume prune -f' and remove all dangling volumes",
        assume_yes,
    )
    proc = _run(["docker", "volume", "prune", "-f"], check=False)
    print(proc.stdout.strip())

    report = _write_report(
        command="reclaim-all",
        dry_run=False,
        payload={
            "dangling_count": len(dangling),
            "dangling_volumes": dangling,
            "prune_stdout": proc.stdout,
            "prune_stderr": proc.stderr,
            "prune_exit_code": proc.returncode,
        },
    )
    print(f"\nReport written to: {report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BIZRA Docker Volume Governance")
    parser.add_argument(
        "command",
        choices=["inventory", "orphans", "reclaim-k3d", "reclaim-all"],
        help="Governance action",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview actions only")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Top-N items shown for inventory/orphans reports",
    )
    parser.add_argument(
        "--restart-cluster",
        action="store_true",
        help="Restart k3d cluster after cache reclaim (reclaim-k3d only)",
    )
    args = parser.parse_args()

    if not _docker_available():
        print(
            "Docker daemon is not reachable. Start Docker Desktop / daemon and retry."
        )
        return 2

    if args.command == "inventory":
        return cmd_inventory(top=args.top, dry_run=args.dry_run)
    if args.command == "orphans":
        return cmd_orphans(top=args.top, dry_run=args.dry_run)
    if args.command == "reclaim-k3d":
        return cmd_reclaim_k3d(
            dry_run=args.dry_run,
            assume_yes=args.yes,
            restart_cluster=args.restart_cluster,
        )
    return cmd_reclaim_all(dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
