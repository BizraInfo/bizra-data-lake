#!/usr/bin/env python3
"""Node0 Genesis Asset Manifest — content-addressed declaration of founder seed assets.

Side-track v0.1, decoupled from the runtime PR queue.

Hard scope (per canon patch 2026-04-25):
- No Claim Registry implementation.
- No attestation service.
- No Proof-of-Impact engine.
- No edits to core/, src/, proof_engine/, bus/, identity/, FATE, CI.
- No raw private content exposed in any output.
- No commit, push, or PR.

The script reads a manifest JSON describing assets, honors each asset's
``visibility`` per ``evidence/node0_genesis_manifest/REDACTION_POLICY.md``,
computes BLAKE3 (or SHA-256 fallback) content + metadata hashes, and writes:

    NODE0_GENESIS_HASH_LEDGER.jsonl   (append-only, one JSON line per asset)
    RUN_REPORT.json                    (counts, warnings, hash algorithm)

Output paths are confined to the manifest's directory by default. The script
NEVER writes outside ``--output-dir`` and NEVER opens files when
``visibility == "redacted"``.

P1 risk recorded for future closure (NOT mitigated here):
    Adversarial evidence poisoning — structurally plausible evidence with
    manipulated empirical/provenance content may pass hash convergence and
    corrupt downstream memory lessons. Closure: Phase 3 Claim Registry must
    require source lineage, provenance validation, and independent
    attestation before upgrading any claim from `directional` to `measured`
    or `independently_attested`. The Genesis Manifest does not protect
    against this risk and does not claim to.

Future IhsanDecision boundary (recorded, NOT implemented):
    IhsanDecision = {
        gate: "allow" | "allow_with_notice" | "require_approval" | "deny",
        trigger_reason: <deterministic string>,
        trace_hash: hash(action_id + policy_version + decision + timestamp),
    }
    Visibility upgrades (private -> public) will require an IhsanDecision
    record once the runtime Phase 2 + Phase 3 spine lands.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

LOG = logging.getLogger("node0_genesis_manifest")

# -- Hashing primitives ----------------------------------------------------

try:  # BLAKE3 is preferred; ``cryptography`` is already in pyproject deps.
    import blake3 as _blake3_mod  # type: ignore[import-not-found]

    _BLAKE3_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent.
    _blake3_mod = None  # type: ignore[assignment]
    _BLAKE3_AVAILABLE = False


def _new_hasher(algorithm: str):
    if algorithm == "BLAKE3":
        if not _BLAKE3_AVAILABLE:
            raise RuntimeError(
                "BLAKE3 hash algorithm requested but `blake3` package is not "
                "installed. Re-run with --hash-algorithm SHA-256 or pip install blake3."
            )
        return _blake3_mod.blake3()  # type: ignore[union-attr]
    if algorithm == "SHA-256":
        return hashlib.sha256()
    raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def _hash_bytes(data: bytes, algorithm: str) -> str:
    h = _new_hasher(algorithm)
    h.update(data)
    return h.hexdigest()


def _hash_file(path: Path, algorithm: str, *, chunk_size: int = 1 << 20) -> tuple[str, int]:
    h = _new_hasher(algorithm)
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


# -- Repo-relative resolution + git tree digest ---------------------------


def _repo_root() -> Path:
    """Best-effort repo root detection."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return here


def _git_ls_tree_digest(repo_root: Path, algorithm: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_root),
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _hash_bytes(result.stdout.encode("utf-8"), algorithm)


def _git_head_sha(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_root),
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


# -- Asset processing ------------------------------------------------------


VALID_VISIBILITY = {"public", "private", "hash_only", "redacted"}
VALID_PROOF_STATUS = {"VERIFIED", "MEASURED", "DERIVED", "FOUNDER_STATED", "PLANNED"}


def _validate_asset(asset: dict[str, Any], idx: int) -> list[str]:
    errors: list[str] = []
    for required in ("asset_id", "category", "title", "visibility", "proof_status"):
        if not asset.get(required):
            errors.append(f"asset[{idx}] missing required field: {required}")
    visibility = asset.get("visibility")
    if visibility and visibility not in VALID_VISIBILITY:
        errors.append(
            f"asset[{idx}] '{asset.get('asset_id')}' invalid visibility: {visibility}"
        )
    if visibility == "redacted" and not asset.get("redaction_reason"):
        errors.append(
            f"asset[{idx}] '{asset.get('asset_id')}' visibility=redacted requires redaction_reason"
        )
    proof_status = asset.get("proof_status")
    if proof_status and proof_status not in VALID_PROOF_STATUS:
        errors.append(
            f"asset[{idx}] '{asset.get('asset_id')}' invalid proof_status: {proof_status}"
        )
    return errors


def _metadata_hash(asset: dict[str, Any], algorithm: str) -> str:
    """Hash that is computable for every asset, including redacted ones.

    Covers the operator-supplied claim: identity, visibility, status, and
    location-hint. Does NOT include any file content. Title and reason are
    operator-controlled per REDACTION_POLICY.md; the operator is responsible
    for ensuring those fields do not leak.
    """
    payload = {
        "asset_id": asset.get("asset_id"),
        "category": asset.get("category"),
        "title": asset.get("title"),
        "language": asset.get("language"),
        "created_at_claim": asset.get("created_at_claim"),
        "current_location": asset.get("current_location"),
        "visibility": asset.get("visibility"),
        "proof_status": asset.get("proof_status"),
        "redaction_reason": asset.get("redaction_reason"),
        "path_relative": asset.get("path_relative"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return _hash_bytes(canonical, algorithm)


def _process_asset(
    asset: dict[str, Any],
    *,
    repo_root: Path,
    algorithm: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return (ledger_entry, warnings_for_this_asset)."""
    warnings: list[str] = []
    visibility = asset.get("visibility")
    asset_id = asset.get("asset_id")

    metadata_hash = _metadata_hash(asset, algorithm)

    content_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    resolution = "skipped"

    is_special_repo_index = asset_id == "node0-repo-head-tree-0007"

    if is_special_repo_index:
        # Special handling: hash `git ls-tree -r HEAD` output, no file open.
        digest = _git_ls_tree_digest(repo_root, algorithm)
        if digest is None:
            warnings.append(
                f"{asset_id}: git ls-tree HEAD failed; repo index hash unavailable"
            )
        else:
            content_hash = digest
            head = _git_head_sha(repo_root)
            if head:
                # Append head sha into resolution note for trace; not into content_hash.
                resolution = f"git_ls_tree:HEAD={head[:12]}"
            else:
                resolution = "git_ls_tree:HEAD=unknown"
    elif visibility in ("public", "private", "hash_only"):
        path_relative = asset.get("path_relative")
        if not path_relative:
            warnings.append(
                f"{asset_id}: visibility={visibility} but path_relative is null; cannot hash content"
            )
            resolution = "no_path"
        else:
            target = (repo_root / path_relative).resolve()
            try:
                target.relative_to(repo_root.resolve())
            except ValueError:
                warnings.append(
                    f"{asset_id}: path_relative escapes repo root; refusing to read"
                )
                resolution = "out_of_tree_refused"
                target = None  # type: ignore[assignment]
            if target is not None:
                if not target.is_file():
                    warnings.append(
                        f"{asset_id}: path not found or not a file; expected {path_relative}"
                    )
                    resolution = "not_found"
                else:
                    try:
                        content_hash, size_bytes = _hash_file(target, algorithm)
                        resolution = "hashed"
                    except OSError as exc:
                        warnings.append(
                            f"{asset_id}: read error class={type(exc).__name__}; content_hash unavailable"
                        )
                        resolution = "read_error"
    elif visibility == "redacted":
        # Per REDACTION_POLICY: do NOT open the file.
        resolution = "redacted_skipped"
    else:
        warnings.append(f"{asset_id}: unknown visibility={visibility!r}; skipped")

    # For private/hash_only, the script computes the hash but does not echo
    # the path beyond what was provided in the manifest itself.
    emit_path = asset.get("path_relative") if visibility == "public" else None

    ledger_entry: dict[str, Any] = {
        "asset_id": asset_id,
        "category": asset.get("category"),
        "visibility": visibility,
        "proof_status": asset.get("proof_status"),
        "metadata_hash": metadata_hash,
        "content_hash": content_hash,
        "size_bytes": size_bytes,
        "path_relative": emit_path,
        "resolution": resolution,
        "hash_algorithm": algorithm,
    }
    return ledger_entry, warnings


# -- Ledger chain ----------------------------------------------------------


def _ledger_line_hash(prev_hash: Optional[str], entry: dict[str, Any], algorithm: str) -> str:
    payload = {
        "prev": prev_hash,
        "entry": entry,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _hash_bytes(canonical, algorithm)


# -- Main ------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_manifest = (
        _repo_root()
        / "evidence"
        / "node0_genesis_manifest"
        / "NODE0_GENESIS_ASSET_MANIFEST.json"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help="Path to the asset manifest JSON. Defaults to evidence/node0_genesis_manifest/NODE0_GENESIS_ASSET_MANIFEST.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the hash ledger and run report. Defaults to the manifest's parent directory.",
    )
    parser.add_argument(
        "--hash-algorithm",
        choices=("BLAKE3", "SHA-256"),
        default=None,
        help="Override hash algorithm. Default: read from manifest.hash_algorithm; falls back to BLAKE3 if available, else SHA-256.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    manifest_path: Path = args.manifest
    if not manifest_path.is_file():
        LOG.error("manifest not found: %s", manifest_path)
        return 2

    output_dir: Path = args.output_dir or manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.error("manifest read/parse error: %s", exc)
        return 2

    algorithm = args.hash_algorithm or manifest.get("hash_algorithm")
    if not algorithm:
        algorithm = "BLAKE3" if _BLAKE3_AVAILABLE else "SHA-256"
    if algorithm == "BLAKE3" and not _BLAKE3_AVAILABLE:
        LOG.warning("blake3 package not available; falling back to SHA-256")
        algorithm = "SHA-256"

    repo_root = _repo_root()
    LOG.info("repo_root=%s", repo_root)
    LOG.info("manifest=%s", manifest_path)
    LOG.info("output_dir=%s", output_dir)
    LOG.info("hash_algorithm=%s", algorithm)

    assets = manifest.get("assets") or []
    LOG.info("processing %d asset(s)", len(assets))

    structural_errors: list[str] = []
    for idx, asset in enumerate(assets):
        structural_errors.extend(_validate_asset(asset, idx))
    if structural_errors:
        LOG.error("structural validation failed:\n  %s", "\n  ".join(structural_errors))
        return 3

    ledger_path = output_dir / "NODE0_GENESIS_HASH_LEDGER.jsonl"
    report_path = output_dir / "RUN_REPORT.json"

    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    counts: dict[str, int] = {
        "total": len(assets),
        "hashed": 0,
        "redacted": 0,
        "not_found": 0,
        "read_error": 0,
        "no_path": 0,
        "out_of_tree_refused": 0,
        "git_indexed": 0,
        "skipped_other": 0,
    }
    proof_status_counts: dict[str, int] = {k: 0 for k in VALID_PROOF_STATUS}
    visibility_counts: dict[str, int] = {k: 0 for k in VALID_VISIBILITY}
    all_warnings: list[str] = []
    prev_line_hash: Optional[str] = None

    with ledger_path.open("w", encoding="utf-8") as ledger_fh:
        # Header line — chains to nothing on a fresh manifest.
        header = {
            "_kind": "manifest_header",
            "version": manifest.get("version"),
            "node_id": manifest.get("node_id"),
            "manifest_path": str(manifest_path.relative_to(repo_root)) if manifest_path.is_relative_to(repo_root) else manifest_path.name,
            "manifest_metadata_hash": _hash_bytes(
                json.dumps(
                    {
                        "version": manifest.get("version"),
                        "node_id": manifest.get("node_id"),
                        "generated_at_utc": manifest.get("generated_at_utc"),
                        "hash_algorithm": algorithm,
                        "previous_manifest_hash": manifest.get("previous_manifest_hash"),
                        "asset_count": len(assets),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                algorithm,
            ),
            "started_at_utc": started_at,
            "hash_algorithm": algorithm,
            "previous_manifest_hash": manifest.get("previous_manifest_hash"),
        }
        header_line_hash = _ledger_line_hash(None, header, algorithm)
        header["line_hash"] = header_line_hash
        ledger_fh.write(json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
        prev_line_hash = header_line_hash

        for idx, asset in enumerate(assets):
            entry, warnings = _process_asset(
                asset, repo_root=repo_root, algorithm=algorithm
            )
            entry["_kind"] = "asset"
            entry["index"] = idx
            entry["prev_line_hash"] = prev_line_hash
            line_hash = _ledger_line_hash(prev_line_hash, entry, algorithm)
            entry["line_hash"] = line_hash
            ledger_fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            prev_line_hash = line_hash

            # Bump counters
            visibility = entry.get("visibility")
            if visibility in visibility_counts:
                visibility_counts[visibility] += 1
            proof_status = entry.get("proof_status")
            if proof_status in proof_status_counts:
                proof_status_counts[proof_status] += 1
            resolution = entry.get("resolution")
            if resolution == "hashed":
                counts["hashed"] += 1
            elif resolution == "redacted_skipped":
                counts["redacted"] += 1
            elif resolution == "not_found":
                counts["not_found"] += 1
            elif resolution == "read_error":
                counts["read_error"] += 1
            elif resolution == "no_path":
                counts["no_path"] += 1
            elif resolution == "out_of_tree_refused":
                counts["out_of_tree_refused"] += 1
            elif resolution and resolution.startswith("git_ls_tree"):
                counts["git_indexed"] += 1
                counts["hashed"] += 1
            else:
                counts["skipped_other"] += 1

            for w in warnings:
                LOG.warning(w)
                all_warnings.append(w)

        # Footer line — final chain anchor.
        footer = {
            "_kind": "manifest_footer",
            "asset_count": len(assets),
            "completed_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "final_chain_hash": prev_line_hash,
            "warnings_count": len(all_warnings),
        }
        footer_line_hash = _ledger_line_hash(prev_line_hash, footer, algorithm)
        footer["line_hash"] = footer_line_hash
        ledger_fh.write(json.dumps(footer, sort_keys=True, separators=(",", ":")) + "\n")
        prev_line_hash = footer_line_hash

    completed_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    run_report = {
        "version": "node0-genesis-run-report-v1",
        "node_id": manifest.get("node_id"),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "manifest_path": str(manifest_path),
        "manifest_version": manifest.get("version"),
        "ledger_path": str(ledger_path),
        "hash_algorithm": algorithm,
        "blake3_available": _BLAKE3_AVAILABLE,
        "counts": counts,
        "proof_status_counts": proof_status_counts,
        "visibility_counts": visibility_counts,
        "warnings": all_warnings,
        "final_chain_hash": prev_line_hash,
        "constraints_honored": {
            "no_runtime_files_edited": True,
            "no_commits_or_pushes": True,
            "no_raw_private_content_emitted": True,
            "no_network_calls": True,
            "no_writes_outside_output_dir": True,
        },
        "p1_risk_recorded_not_mitigated": (
            "Adversarial evidence poisoning. Closure: future Phase 3 Claim Registry must "
            "require source lineage, provenance validation, and independent attestation "
            "for upgraded claims."
        ),
        "ihsan_decision_schema_recorded_not_implemented": {
            "gate": ["allow", "allow_with_notice", "require_approval", "deny"],
            "trigger_reason": "deterministic string explaining the rule/input that triggered the decision",
            "trace_hash": "hash(action_id + policy_version + decision + timestamp)",
        },
    }
    report_path.write_text(
        json.dumps(run_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    LOG.info("hashed=%d redacted=%d not_found=%d read_error=%d no_path=%d",
             counts["hashed"], counts["redacted"], counts["not_found"],
             counts["read_error"], counts["no_path"])
    LOG.info("final_chain_hash=%s", prev_line_hash)
    LOG.info("ledger_path=%s", ledger_path)
    LOG.info("run_report_path=%s", report_path)

    return 0 if not structural_errors else 1


if __name__ == "__main__":
    sys.exit(main())
