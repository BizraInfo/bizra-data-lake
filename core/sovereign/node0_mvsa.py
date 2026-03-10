"""
Node0 MVSA Proof — Python wrapper for the Rust proof binary.

Binary resolution order (§3 — fail-closed):
  1. BIZRA_NODE0_MVSA_BIN env var
  2. bizra-omega/target/release/node0-mvsa
  3. bizra-omega/target/debug/node0-mvsa
  4. cargo run -p bizra-resourcepool --bin node0-mvsa -- ...
  5. fail closed: RUST_MVSA_BINARY_UNAVAILABLE

Standing on Giants:
- Boyd (1976): OODA loop — observe genesis, orient proof, decide, act
- Deming (1950): PDCA — plan (resolve), do (run), check (validate), act (persist)
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from .atomic_io import read_json

logger = logging.getLogger(__name__)

REASON_BINARY_UNAVAILABLE = "RUST_MVSA_BINARY_UNAVAILABLE"
REASON_BINARY_FAILED = "RUST_MVSA_BINARY_FAILED"
PROOF_FILE = "node0_mvsa_proof.json"


def _resolve_binary(project_root: Path) -> Optional[Path]:
    """Resolve the Rust MVSA binary following strict precedence."""

    def _existing_binary(path: Path) -> Optional[Path]:
        candidates = [path]
        if os.name == "nt":
            candidates.append(path.with_suffix(".exe"))
        for candidate in candidates:
            if candidate.exists() and os.access(str(candidate), os.X_OK):
                return candidate
        return None

    # 1. Environment variable
    env_bin = os.environ.get("BIZRA_NODE0_MVSA_BIN")
    if env_bin:
        p = Path(env_bin)
        resolved = _existing_binary(p)
        if resolved is not None:
            return resolved
        logger.warning("BIZRA_NODE0_MVSA_BIN=%s not found or not executable", env_bin)

    omega_dir = project_root / "bizra-omega"

    # 2. Release binary
    release = omega_dir / "target" / "release" / "node0-mvsa"
    resolved = _existing_binary(release)
    if resolved is not None:
        return resolved

    # 3. Debug binary
    debug = omega_dir / "target" / "debug" / "node0-mvsa"
    resolved = _existing_binary(debug)
    if resolved is not None:
        return resolved

    return None


def _to_wsl_path(path: Path) -> str:
    """Convert a Windows path into its WSL /mnt form."""
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if drive:
        suffix = resolved.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{suffix}"
    return resolved.as_posix()


def _run_binary(
    binary: Path,
    state_dir: Path,
    out_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the resolved binary with state-dir and out args."""
    cmd = [str(binary), "--state-dir", str(state_dir), "--out", str(out_path)]
    logger.info("Running MVSA binary: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_cargo(
    project_root: Path,
    state_dir: Path,
    out_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Fallback: run via cargo."""
    omega_dir = project_root / "bizra-omega"
    cmd = [
        "cargo",
        "run",
        "-p",
        "bizra-resourcepool",
        "--bin",
        "node0-mvsa",
        "--",
        "--state-dir",
        str(state_dir),
        "--out",
        str(out_path),
    ]
    logger.info("Running MVSA via cargo: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(omega_dir),
    )


def _run_cargo_wsl(
    project_root: Path,
    state_dir: Path,
    out_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Fallback cargo execution through WSL for Windows-hosted repos."""
    omega_dir = _to_wsl_path(project_root / "bizra-omega")
    state_dir_wsl = _to_wsl_path(state_dir)
    out_path_wsl = _to_wsl_path(out_path)
    inner = (
        f"cd {omega_dir} && "
        "cargo run -p bizra-resourcepool --bin node0-mvsa -- "
        f"--state-dir {state_dir_wsl} --out {out_path_wsl}"
    )
    cmd = ["wsl.exe", "bash", "-lc", inner]
    logger.info("Running MVSA via WSL cargo: %s", inner)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def run_mvsa_proof(
    state_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    """
    Execute the Rust MVSA proof binary and return the proof artifact.

    Returns the parsed proof JSON on success.
    Raises RuntimeError on any failure.
    """
    out_path = state_dir / PROOF_FILE

    binary = _resolve_binary(project_root)

    if binary is not None:
        result = _run_binary(binary, state_dir, out_path)
    else:
        # Step 4: cargo run fallback
        cargo_toml = project_root / "bizra-omega" / "bizra-resourcepool" / "Cargo.toml"
        if not cargo_toml.exists():
            raise RuntimeError(
                f"{REASON_BINARY_UNAVAILABLE}: no binary found and "
                f"Cargo.toml missing at {cargo_toml}"
            )
        try:
            result = _run_cargo(project_root, state_dir, out_path)
        except FileNotFoundError:
            if os.name == "nt":
                try:
                    result = _run_cargo_wsl(project_root, state_dir, out_path)
                except FileNotFoundError:
                    raise RuntimeError(
                        f"{REASON_BINARY_UNAVAILABLE}: cargo not found in PATH and wsl.exe unavailable"
                    )
            else:
                raise RuntimeError(
                    f"{REASON_BINARY_UNAVAILABLE}: cargo not found in PATH"
                )

    logger.info("MVSA binary stderr:\n%s", result.stderr)

    # Parse output regardless of exit code (binary writes proof even on failure)
    proof = read_json(out_path)
    if proof is None:
        raise RuntimeError(
            f"{REASON_BINARY_FAILED}: exit_code={result.returncode}, "
            f"stderr={result.stderr[:500]}"
        )

    if result.returncode != 0:
        logger.warning(
            "MVSA binary exited with code %d (status=%s)",
            result.returncode,
            proof.get("status", "unknown"),
        )

    return proof


def read_mvsa_proof(state_dir: Path) -> Optional[dict[str, Any]]:
    """Read the persisted MVSA proof artifact, or None if absent."""
    return read_json(state_dir / PROOF_FILE)


__all__ = [
    "run_mvsa_proof",
    "read_mvsa_proof",
    "PROOF_FILE",
    "REASON_BINARY_UNAVAILABLE",
    "REASON_BINARY_FAILED",
]
