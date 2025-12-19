#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"model family lint failed: PyYAML is required ({exc})")
    raise SystemExit(2)


PROVIDERS = {"ollama", "lmstudio"}
SHA256_PREFIXED_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at top-level: {path}")
    return data


def is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Validate sealed model family manifest")
    parser.add_argument(
        "--manifest",
        default=str(repo_root / "model-family-genesis-v1-SEALED.yaml"),
        help="Path to model family YAML (default: model-family-genesis-v1-SEALED.yaml)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"model family lint skipped: missing {manifest_path}")
        return 0

    try:
        mf = load_yaml(manifest_path)
    except Exception as exc:
        print(f"Model family lint failed: {exc}")
        return 2

    failures: list[str] = []

    version = mf.get("version")
    if not isinstance(version, int) or version < 1:
        failures.append(f"version must be an int >= 1 ({manifest_path})")

    sealed = mf.get("sealed")
    if sealed is not True:
        failures.append(f"sealed must be true ({manifest_path})")

    sealed_at = mf.get("sealed_at_utc")
    if not is_nonempty_str(sealed_at):
        failures.append(f"sealed_at_utc must be a non-empty string ({manifest_path})")

    sealed_by = mf.get("sealed_by")
    if not (isinstance(sealed_by, list) and all(is_nonempty_str(s) for s in sealed_by)):
        failures.append(f"sealed_by must be a non-empty list of strings ({manifest_path})")

    capability_slots = mf.get("capability_slots")
    if not isinstance(capability_slots, dict) or not capability_slots:
        failures.append(f"capability_slots must be a non-empty mapping ({manifest_path})")

    pinned = mf.get("pinned_artifacts")
    if not isinstance(pinned, dict) or not pinned:
        failures.append(f"pinned_artifacts must be a non-empty mapping ({manifest_path})")

    if isinstance(pinned, dict):
        for artifact_id, spec in pinned.items():
            if not is_nonempty_str(artifact_id):
                failures.append(f"pinned_artifacts contains non-string key ({manifest_path})")
                continue
            if not isinstance(spec, dict):
                failures.append(f"pinned_artifacts['{artifact_id}'] must be a mapping ({manifest_path})")
                continue

            provider = spec.get("provider")
            if not is_nonempty_str(provider) or provider.strip().lower() not in PROVIDERS:
                failures.append(
                    f"pinned_artifacts['{artifact_id}'].provider must be one of {sorted(PROVIDERS)} ({manifest_path})"
                )
                continue

            provider_norm = provider.strip().lower()
            if provider_norm == "ollama":
                digest = spec.get("digest")
                if not is_nonempty_str(digest) or not SHA256_PREFIXED_RE.match(digest.strip().lower()):
                    failures.append(
                        f"pinned_artifacts['{artifact_id}'].digest must be sha256:<64hex> ({manifest_path})"
                    )

                modelfile_sha = spec.get("modelfile_sha256")
                if not is_nonempty_str(modelfile_sha) or not SHA256_HEX_RE.match(
                    modelfile_sha.strip().lower()
                ):
                    failures.append(
                        f"pinned_artifacts['{artifact_id}'].modelfile_sha256 must be 64 hex chars ({manifest_path})"
                    )

            if provider_norm == "lmstudio":
                model_id = spec.get("model_id")
                if not is_nonempty_str(model_id):
                    failures.append(
                        f"pinned_artifacts['{artifact_id}'].model_id must be a non-empty string ({manifest_path})"
                    )

                files = spec.get("files")
                if not (isinstance(files, list) and files):
                    failures.append(
                        f"pinned_artifacts['{artifact_id}'].files must be a non-empty list ({manifest_path})"
                    )
                else:
                    for idx, item in enumerate(files, start=1):
                        if not isinstance(item, dict):
                            failures.append(
                                f"pinned_artifacts['{artifact_id}'].files[{idx}] must be a mapping ({manifest_path})"
                            )
                            continue
                        name = item.get("name")
                        if not is_nonempty_str(name):
                            failures.append(
                                f"pinned_artifacts['{artifact_id}'].files[{idx}].name must be a non-empty string ({manifest_path})"
                            )
                        sha = item.get("sha256")
                        if not is_nonempty_str(sha) or not SHA256_HEX_RE.match(sha.strip().lower()):
                            failures.append(
                                f"pinned_artifacts['{artifact_id}'].files[{idx}].sha256 must be 64 hex chars ({manifest_path})"
                            )
                        size_bytes = item.get("size_bytes")
                        if not isinstance(size_bytes, int) or size_bytes <= 0:
                            failures.append(
                                f"pinned_artifacts['{artifact_id}'].files[{idx}].size_bytes must be a positive int ({manifest_path})"
                            )

    if isinstance(capability_slots, dict) and isinstance(pinned, dict):
        for slot_name, slot_spec in capability_slots.items():
            if not is_nonempty_str(slot_name):
                failures.append(f"capability_slots contains non-string key ({manifest_path})")
                continue
            if not isinstance(slot_spec, dict):
                failures.append(f"capability_slots['{slot_name}'] must be a mapping ({manifest_path})")
                continue

            allowed = slot_spec.get("allowed_models")
            if not (isinstance(allowed, list) and allowed and all(is_nonempty_str(s) for s in allowed)):
                failures.append(
                    f"capability_slots['{slot_name}'].allowed_models must be a non-empty list of strings ({manifest_path})"
                )
                continue

            allowed_set = {s.strip() for s in allowed if isinstance(s, str)}
            missing = sorted([a for a in allowed_set if a not in pinned], key=str.casefold)
            if missing:
                failures.append(
                    f"capability_slots['{slot_name}'] references unpinned models {missing} ({manifest_path})"
                )

            routing = slot_spec.get("routing")
            if not isinstance(routing, dict):
                failures.append(
                    f"capability_slots['{slot_name}'].routing must be a mapping ({manifest_path})"
                )
                continue

            primary = routing.get("primary")
            if not is_nonempty_str(primary) or primary.strip() not in allowed_set:
                failures.append(
                    f"capability_slots['{slot_name}'].routing.primary must be in allowed_models ({manifest_path})"
                )

            fallback = routing.get("fallback")
            if fallback is not None:
                if not is_nonempty_str(fallback) or fallback.strip() not in allowed_set:
                    failures.append(
                        f"capability_slots['{slot_name}'].routing.fallback must be in allowed_models ({manifest_path})"
                    )

    resource_policy = mf.get("resource_policy")
    if isinstance(resource_policy, dict):
        global_policy = resource_policy.get("global")
        if isinstance(global_policy, dict):
            total = global_policy.get("max_loaded_models_total")
            if total != 1:
                failures.append(
                    f"resource_policy.global.max_loaded_models_total must be 1 (single-resident policy) ({manifest_path})"
                )

    if failures:
        print("Model family lint failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

