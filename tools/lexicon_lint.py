#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"lexicon lint failed: PyYAML is required ({exc})")
    raise SystemExit(2)


ALLOWED_TRUTH = {"VERIFIED", "MEASURED", "TARGET", "DERIVED"}
ALLOWED_TERM_KEYS = {
    "expansion",
    "role",
    "notes",
    "required_fields",
    "references",
    "examples",
    "variants",
    "invariants",
}
REQUIRED_ADAPTER_MODE_FIELDS = {"pat", "sat", "mcp", "a2a", "reasoning"}

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at top-level: {path}")
    return data


def parse_semver(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    m = SEMVER_RE.match(value.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def expected_lexicon_id(semver: tuple[int, int, int]) -> str:
    return f"bizra_lexicon_v{semver[0]}_{semver[1]}_{semver[2]}"


def is_git_repo(repo_root: Path) -> bool:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False

    return res.returncode == 0 and res.stdout.strip().lower() == "true"


def git_show_text(repo_root: Path, ref: str, relpath: str) -> str | None:
    res = subprocess.run(
        ["git", "show", f"{ref}:{relpath}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    return res.stdout


def load_baseline_from_git(repo_root: Path, ref: str, relpath: str) -> dict | None:
    text = git_show_text(repo_root, ref, relpath)
    if text is None:
        return None
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return None
    return data


def find_baseline_lexicon(
    repo_root: Path,
    relpath: str,
    baseline_path: Path | None,
    baseline_ref: str | None,
) -> dict | None:
    if baseline_path is not None:
        try:
            return load_yaml(baseline_path)
        except Exception:
            return None

    if baseline_ref is not None and is_git_repo(repo_root):
        return load_baseline_from_git(repo_root, baseline_ref, relpath)

    if not is_git_repo(repo_root):
        return None

    for ref in ["HEAD^1", "HEAD~1"]:
        baseline = load_baseline_from_git(repo_root, ref, relpath)
        if baseline is not None:
            return baseline

    return None


def enforce_append_only(
    *,
    current: dict,
    baseline: dict,
    failures: list[str],
    lexicon_path: Path,
) -> None:
    current_terms = current.get("terms")
    baseline_terms = baseline.get("terms")
    if not isinstance(current_terms, dict) or not isinstance(baseline_terms, dict):
        return

    for term, baseline_spec in baseline_terms.items():
        if term not in current_terms:
            failures.append(f"append-only violation: removed term '{term}' ({lexicon_path})")
            continue
        if current_terms.get(term) != baseline_spec:
            failures.append(f"append-only violation: modified term '{term}' ({lexicon_path})")

    current_truth = current.get("truth_labels")
    baseline_truth = baseline.get("truth_labels")
    if isinstance(current_truth, dict) and isinstance(baseline_truth, dict):
        if current_truth != baseline_truth:
            failures.append(f"append-only violation: modified truth_labels ({lexicon_path})")

    baseline_semver = parse_semver(baseline.get("semver"))
    current_semver = parse_semver(current.get("semver"))
    if baseline_semver is None or current_semver is None:
        return

    if current_semver < baseline_semver:
        failures.append(
            f"append-only violation: semver regressed from {baseline.get('semver')} to {current.get('semver')} ({lexicon_path})"
        )

    added_terms = sorted(set(current_terms.keys()) - set(baseline_terms.keys()), key=str.casefold)
    if added_terms and current_semver <= baseline_semver:
        failures.append(
            f"append-only violation: added terms {added_terms} without semver bump (baseline {baseline.get('semver')} current {current.get('semver')}) ({lexicon_path})"
        )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Validate the BIZRA Lexicon Ledger")
    parser.add_argument(
        "--lexicon",
        default=str(repo_root / "constitution" / "lexicon_v1.yaml"),
        help="Path to lexicon YAML (default: constitution/lexicon_v1.yaml)",
    )
    parser.add_argument(
        "--baseline",
        help="Optional baseline lexicon YAML; enables append-only checks against it",
    )
    parser.add_argument(
        "--baseline-ref",
        help="Optional git ref to load baseline lexicon from (e.g., HEAD^1, origin/main)",
    )
    args = parser.parse_args()

    lexicon_path = Path(args.lexicon)
    contract_path = repo_root / "constitution" / "lexicon_ledger_contract_v1.yaml"
    schema_path = repo_root / "schemas" / "lexicon_v1.schema.json"
    receipt_schema_path = repo_root / "schemas" / "lexicon_receipt_v1.schema.json"
    baseline_path = Path(args.baseline) if args.baseline else None

    failures: list[str] = []

    for required in [lexicon_path, contract_path, schema_path, receipt_schema_path]:
        if not required.exists():
            failures.append(f"missing required file: {required}")

    if failures:
        print("Lexicon lint failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    try:
        lexicon = load_yaml(lexicon_path)
        contract = load_yaml(contract_path)
    except Exception as exc:
        print(f"Lexicon lint failed: {exc}")
        return 1

    if lexicon.get("status") != "canonical":
        failures.append(f"lexicon status must be 'canonical' ({lexicon_path})")

    if lexicon.get("append_only") is not True:
        failures.append(f"lexicon append_only must be true ({lexicon_path})")

    semver = parse_semver(lexicon.get("semver"))
    if semver is None:
        failures.append(f"lexicon semver must be X.Y.Z ({lexicon_path})")
    else:
        expected_id = expected_lexicon_id(semver)
        if lexicon.get("id") != expected_id:
            failures.append(
                f"lexicon id must be '{expected_id}' for semver {lexicon.get('semver')} ({lexicon_path})"
            )

    lexicon_schema_ref = lexicon.get("schema")
    if lexicon_schema_ref != "schemas/lexicon_v1.schema.json":
        failures.append(f"lexicon schema must be schemas/lexicon_v1.schema.json ({lexicon_path})")

    receipt_schema_ref = lexicon.get("receipt_schema")
    if receipt_schema_ref != "schemas/lexicon_receipt_v1.schema.json":
        failures.append(
            f"lexicon receipt_schema must be schemas/lexicon_receipt_v1.schema.json ({lexicon_path})"
        )

    contract_ref = lexicon.get("contract")
    if contract_ref != "constitution/lexicon_ledger_contract_v1.yaml":
        failures.append(
            f"lexicon contract must be constitution/lexicon_ledger_contract_v1.yaml ({lexicon_path})"
        )

    purpose = lexicon.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        failures.append(f"lexicon purpose must be a non-empty string ({lexicon_path})")

    truth_labels = lexicon.get("truth_labels")
    if not isinstance(truth_labels, dict):
        failures.append(f"lexicon truth_labels must be a mapping ({lexicon_path})")
    else:
        missing = ALLOWED_TRUTH - set(str(k).upper() for k in truth_labels.keys())
        if missing:
            failures.append(f"lexicon truth_labels missing: {sorted(missing)} ({lexicon_path})")
        for key, value in truth_labels.items():
            if str(key).upper() not in ALLOWED_TRUTH:
                failures.append(f"lexicon truth_labels contains unknown key '{key}' ({lexicon_path})")
            if not isinstance(value, str) or not value.strip():
                failures.append(f"lexicon truth_labels['{key}'] must be a non-empty string ({lexicon_path})")

    terms = lexicon.get("terms")
    if not isinstance(terms, dict) or not terms:
        failures.append(f"lexicon terms must be a non-empty mapping ({lexicon_path})")
    else:
        keys = list(terms.keys())
        expected = sorted(keys, key=lambda s: str(s).casefold())
        if keys != expected:
            failures.append(f"lexicon terms keys must be sorted (case-insensitive) ({lexicon_path})")

        for term_name, term_spec in terms.items():
            if not isinstance(term_spec, dict):
                failures.append(f"term '{term_name}' must be a mapping ({lexicon_path})")
                continue

            missing_fields = [k for k in ("expansion", "role") if k not in term_spec]
            if missing_fields:
                failures.append(
                    f"term '{term_name}' missing required fields {missing_fields} ({lexicon_path})"
                )

            extra_keys = set(term_spec.keys()) - ALLOWED_TERM_KEYS
            if extra_keys:
                failures.append(
                    f"term '{term_name}' has unknown keys {sorted(extra_keys)} ({lexicon_path})"
                )

            for key in ["notes", "required_fields", "references", "examples", "invariants"]:
                if key in term_spec and not (
                    isinstance(term_spec[key], list)
                    and all(isinstance(item, str) and item.strip() for item in term_spec[key])
                ):
                    failures.append(
                        f"term '{term_name}' field '{key}' must be a list of non-empty strings ({lexicon_path})"
                    )

            if "variants" in term_spec and not (
                isinstance(term_spec["variants"], dict)
                and all(
                    isinstance(k, str)
                    and isinstance(v, str)
                    and k.strip()
                    and v.strip()
                    for k, v in term_spec["variants"].items()
                )
            ):
                failures.append(
                    f"term '{term_name}' field 'variants' must be a mapping of non-empty strings ({lexicon_path})"
                )

        adapter_modes = terms.get("AdapterModes")
        if isinstance(adapter_modes, dict):
            required = adapter_modes.get("required_fields")
            if isinstance(required, list):
                got = {str(s) for s in required}
                missing = REQUIRED_ADAPTER_MODE_FIELDS - got
                if missing:
                    failures.append(
                        f"AdapterModes.required_fields missing {sorted(missing)} ({lexicon_path})"
                    )

        lexicon_receipt_term = terms.get("LexiconReceipt")
        contract_required = (
            contract.get("receipt_requirements", {}).get("required_fields")  # type: ignore[union-attr]
            if isinstance(contract, dict)
            else None
        )
        if not isinstance(lexicon_receipt_term, dict):
            failures.append(f"term 'LexiconReceipt' must exist and be a mapping ({lexicon_path})")
        else:
            rr = lexicon_receipt_term.get("required_fields")
            if not isinstance(rr, list) or not rr:
                failures.append(
                    f"LexiconReceipt.required_fields must be a non-empty list ({lexicon_path})"
                )
            elif isinstance(contract_required, list) and contract_required:
                if set(rr) != set(contract_required):
                    failures.append(
                        "LexiconReceipt.required_fields must match contract receipt_requirements.required_fields "
                        f"({lexicon_path})"
                    )

    baseline = find_baseline_lexicon(
        repo_root,
        relpath="constitution/lexicon_v1.yaml",
        baseline_path=baseline_path,
        baseline_ref=args.baseline_ref,
    )
    if baseline is not None:
        enforce_append_only(current=lexicon, baseline=baseline, failures=failures, lexicon_path=lexicon_path)

    if failures:
        print("Lexicon lint failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
