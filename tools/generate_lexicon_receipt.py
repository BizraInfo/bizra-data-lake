#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"lexicon receipt generator failed: PyYAML is required ({exc})")
    raise SystemExit(2)


ALLOWED_TRUTH = {"VERIFIED", "MEASURED", "TARGET", "DERIVED"}
REQUIRED_ADAPTER_MODE_FIELDS = {"pat", "sat", "mcp", "a2a", "reasoning"}
ALLOWED_ADAPTER_MODES = {"simulated", "real"}
GIT_SHA_RE = re.compile(r"^[a-f0-9]{40}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def load_yaml_id(path: Path) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at top-level: {path}")
    value = data.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty 'id' in {path}")
    return value.strip()


def load_yaml_map(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at top-level: {path}")
    return data


def parse_adapter_modes(raw: str | None) -> dict[str, str]:
    if raw is None:
        raw = os.getenv("BIZRA_ADAPTER_MODES_JSON")

    if raw is None:
        return {k: "simulated" for k in sorted(REQUIRED_ADAPTER_MODE_FIELDS)}

    try:
        parsed = json.loads(raw)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"adapter modes must be valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("adapter modes JSON must be an object")

    normalized: dict[str, str] = {}
    for k in REQUIRED_ADAPTER_MODE_FIELDS:
        if k not in parsed:
            raise ValueError(f"adapter modes missing required key: {k}")
        v = parsed[k]
        if not isinstance(v, str):
            raise ValueError(f"adapter mode for {k} must be a string")
        val = v.strip().lower()
        if val not in ALLOWED_ADAPTER_MODES:
            raise ValueError(f"adapter mode for {k} must be one of {sorted(ALLOWED_ADAPTER_MODES)}")
        normalized[k] = val

    extra = set(parsed.keys()) - REQUIRED_ADAPTER_MODE_FIELDS
    if extra:
        raise ValueError(f"adapter modes contains unknown keys: {sorted(extra)}")

    return normalized


def repo_commit(repo_root: Path) -> str | None:
    env_sha = os.getenv("GITHUB_SHA")
    if env_sha and GIT_SHA_RE.match(env_sha.strip()):
        return env_sha.strip()

    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None

    sha = res.stdout.strip()
    return sha if GIT_SHA_RE.match(sha) else None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Generate a BIZRA LexiconReceipt v1")
    parser.add_argument(
        "--truth-label",
        required=True,
        help=f"Truth label ({', '.join(sorted(ALLOWED_TRUTH))})",
    )
    parser.add_argument(
        "--adapter-modes-json",
        help=(
            "JSON object with keys pat/sat/mcp/a2a/reasoning and values simulated|real. "
            "If omitted, defaults to all simulated (or uses BIZRA_ADAPTER_MODES_JSON)."
        ),
    )
    parser.add_argument(
        "--lexicon",
        default=str(repo_root / "constitution" / "lexicon_v1.yaml"),
        help="Path to lexicon YAML (default: constitution/lexicon_v1.yaml)",
    )
    parser.add_argument(
        "--ihsan",
        default=str(repo_root / "constitution" / "ihsan_v1.yaml"),
        help="Path to Ihsan constitution YAML (default: constitution/ihsan_v1.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Write receipt JSON to this path (default: stdout)",
    )

    args = parser.parse_args()

    truth_label = args.truth_label.strip().upper()
    if truth_label not in ALLOWED_TRUTH:
        print(f"Invalid truth label '{truth_label}'. Allowed: {sorted(ALLOWED_TRUTH)}")
        return 2

    lexicon_path = Path(args.lexicon)
    ihsan_path = Path(args.ihsan)
    if not lexicon_path.exists():
        print(f"Lexicon file not found: {lexicon_path}")
        return 2
    if not ihsan_path.exists():
        print(f"Ihsan constitution file not found: {ihsan_path}")
        return 2

    try:
        adapter_modes = parse_adapter_modes(args.adapter_modes_json)
        lexicon_data = load_yaml_map(lexicon_path)
        lexicon_id = str(lexicon_data.get("id", "")).strip()
        if not lexicon_id:
            raise ValueError(f"missing non-empty 'id' in {lexicon_path}")
        ihsan_id = load_yaml_id(ihsan_path)
    except Exception as exc:
        print(f"Receipt generation failed: {exc}")
        return 2

    commit = repo_commit(repo_root)
    if commit is None:
        print("Receipt generation failed: could not determine repo commit (git rev-parse HEAD)")
        return 2

    contract_rel = lexicon_data.get("contract")
    schema_rel = lexicon_data.get("schema")
    receipt_schema_rel = lexicon_data.get("receipt_schema")
    if not all(isinstance(v, str) and v.strip() for v in [contract_rel, schema_rel, receipt_schema_rel]):
        print(
            "Receipt generation failed: lexicon must define non-empty contract/schema/receipt_schema paths"
        )
        return 2

    contract_path = repo_root / str(contract_rel)
    schema_path = repo_root / str(schema_rel)
    receipt_schema_path = repo_root / str(receipt_schema_rel)
    for p in [contract_path, schema_path, receipt_schema_path]:
        if not p.exists():
            print(f"Receipt generation failed: referenced file not found: {p}")
            return 2

    lexicon_sha = sha256_file(lexicon_path)
    ihsan_sha = sha256_file(ihsan_path)
    contract_sha = sha256_file(contract_path)
    lexicon_schema_sha = sha256_file(schema_path)
    lexicon_receipt_schema_sha = sha256_file(receipt_schema_path)

    policy_material = {
        "ihsan_constitution_sha256": ihsan_sha,
        "lexicon_contract_sha256": contract_sha,
        "lexicon_receipt_schema_sha256": lexicon_receipt_schema_sha,
        "lexicon_schema_sha256": lexicon_schema_sha,
        "lexicon_sha256": lexicon_sha,
    }
    policy_sha = sha256_text(json.dumps(policy_material, sort_keys=True))

    receipt = {
        "type": "LexiconReceipt",
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "truth_label": truth_label,
        "lexicon_id": lexicon_id,
        "lexicon_sha256": lexicon_sha,
        "ihsan_constitution_id": ihsan_id,
        "ihsan_constitution_sha256": ihsan_sha,
        "lexicon_contract_sha256": contract_sha,
        "lexicon_schema_sha256": lexicon_schema_sha,
        "lexicon_receipt_schema_sha256": lexicon_receipt_schema_sha,
        "policy_sha256": policy_sha,
        "repo_commit": commit,
        "adapter_modes": adapter_modes,
        "generator": {
            "name": "tools/generate_lexicon_receipt.py",
            "version": "1.0.1",
        },
    }

    payload = json.dumps(receipt, indent=2, sort_keys=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
