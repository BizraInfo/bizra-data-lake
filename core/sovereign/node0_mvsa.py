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
    # 1. Environment variable
    env_bin = os.environ.get("BIZRA_NODE0_MVSA_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.exists() and os.access(str(p), os.X_OK):
            return p
        logger.warning("BIZRA_NODE0_MVSA_BIN=%s not found or not executable", env_bin)

    omega_dir = project_root / "bizra-omega"

    # 2. Release binary
    release = omega_dir / "target" / "release" / "node0-mvsa"
    if release.exists() and os.access(str(release), os.X_OK):
        return release

    # 3. Debug binary
    debug = omega_dir / "target" / "debug" / "node0-mvsa"
    if debug.exists() and os.access(str(debug), os.X_OK):
        return debug

    return None


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
        "cargo", "run",
        "-p", "bizra-resourcepool",
        "--bin", "node0-mvsa",
        "--",
        "--state-dir", str(state_dir),
        "--out", str(out_path),
    ]
    logger.info("Running MVSA via cargo: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(omega_dir),
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
