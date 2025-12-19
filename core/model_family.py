from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


PROVIDERS = {"ollama", "lmstudio"}


def _normalize_key(raw: str) -> str:
    return raw.strip()


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def default_manifest_path() -> Path:
    configured = os.getenv("BIZRA_MODEL_FAMILY_MANIFEST", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (_repo_root_from_here() / "model-family-genesis-v1-SEALED.yaml").resolve()


@dataclass(frozen=True)
class SlotRouting:
    name: str
    allowed_models: List[str]
    primary: str
    fallback: Optional[str]


@dataclass(frozen=True)
class PinnedArtifact:
    name: str
    provider: str
    digest: Optional[str]
    modelfile_sha256: Optional[str]
    model_id: Optional[str]
    files: List[Dict[str, Any]]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class ModelFamily:
    path: Path
    sealed: bool
    sealed_at_utc: Optional[str]
    sealed_by: List[str]
    capability_slots: Dict[str, SlotRouting]
    pinned_artifacts: Dict[str, PinnedArtifact]
    raw: Dict[str, Any]

    def slot(self, name: str) -> SlotRouting:
        if name not in self.capability_slots:
            raise KeyError(f"unknown slot: {name}")
        return self.capability_slots[name]

    def artifact(self, model_name: str) -> PinnedArtifact:
        if model_name not in self.pinned_artifacts:
            raise KeyError(f"unpinned model: {model_name}")
        return self.pinned_artifacts[model_name]

    def route_models(self, slot_name: str) -> List[str]:
        slot = self.slot(slot_name)
        ordered: List[str] = []
        if slot.primary:
            ordered.append(slot.primary)
        if slot.fallback and slot.fallback not in ordered:
            ordered.append(slot.fallback)
        # Optional: include remaining allowed models as tertiary fallbacks (stable order)
        for m in slot.allowed_models:
            if m not in ordered:
                ordered.append(m)
        return ordered


def load_model_family(path: Optional[Path] = None) -> ModelFamily:
    mf_path = (path or default_manifest_path()).expanduser().resolve()
    if not mf_path.exists():
        raise FileNotFoundError(f"model family manifest not found: {mf_path}")

    data = yaml.safe_load(mf_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"model family manifest must be a mapping: {mf_path}")

    sealed = bool(data.get("sealed") is True)
    allow_unsealed = os.getenv("BIZRA_ALLOW_UNSEALED_MODEL_FAMILY", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if not sealed and not allow_unsealed:
        raise ValueError(f"model family manifest must be sealed (set BIZRA_ALLOW_UNSEALED_MODEL_FAMILY=1 to override): {mf_path}")

    sealed_at = data.get("sealed_at_utc")
    sealed_at_utc = str(sealed_at) if isinstance(sealed_at, str) and sealed_at.strip() else None

    sealed_by_raw = data.get("sealed_by")
    sealed_by: List[str] = []
    if isinstance(sealed_by_raw, list):
        sealed_by = [str(x).strip() for x in sealed_by_raw if isinstance(x, str) and x.strip()]

    slots_raw = data.get("capability_slots")
    if not isinstance(slots_raw, dict) or not slots_raw:
        raise ValueError(f"capability_slots must be a non-empty mapping: {mf_path}")

    pinned_raw = data.get("pinned_artifacts")
    if not isinstance(pinned_raw, dict) or not pinned_raw:
        raise ValueError(f"pinned_artifacts must be a non-empty mapping: {mf_path}")

    pinned: Dict[str, PinnedArtifact] = {}
    for name, spec in pinned_raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(spec, dict):
            continue
        provider = str(spec.get("provider") or "").strip().lower()
        if provider not in PROVIDERS:
            continue
        digest = spec.get("digest")
        digest_norm = str(digest).strip() if isinstance(digest, str) and digest.strip() else None
        modelfile_sha = spec.get("modelfile_sha256")
        modelfile_sha_norm = str(modelfile_sha).strip().lower() if isinstance(modelfile_sha, str) and modelfile_sha.strip() else None
        model_id = spec.get("model_id")
        model_id_norm = str(model_id).strip() if isinstance(model_id, str) and model_id.strip() else None
        files_raw = spec.get("files")
        files: List[Dict[str, Any]] = files_raw if isinstance(files_raw, list) else []
        pinned[name.strip()] = PinnedArtifact(
            name=name.strip(),
            provider=provider,
            digest=digest_norm,
            modelfile_sha256=modelfile_sha_norm,
            model_id=model_id_norm,
            files=files,
            raw=spec,
        )

    slots: Dict[str, SlotRouting] = {}
    for slot_name, spec in slots_raw.items():
        if not isinstance(slot_name, str) or not slot_name.strip():
            continue
        if not isinstance(spec, dict):
            continue

        allowed_raw = spec.get("allowed_models")
        if not (isinstance(allowed_raw, list) and allowed_raw and all(isinstance(s, str) and s.strip() for s in allowed_raw)):
            raise ValueError(f"capability_slots['{slot_name}'].allowed_models must be a non-empty list: {mf_path}")
        allowed = [str(s).strip() for s in allowed_raw if isinstance(s, str) and s.strip()]

        routing = spec.get("routing")
        if not isinstance(routing, dict):
            raise ValueError(f"capability_slots['{slot_name}'].routing must be a mapping: {mf_path}")
        primary = routing.get("primary")
        if not isinstance(primary, str) or not primary.strip():
            raise ValueError(f"capability_slots['{slot_name}'].routing.primary must be a string: {mf_path}")
        primary = primary.strip()

        fallback = routing.get("fallback")
        fallback_norm = fallback.strip() if isinstance(fallback, str) and fallback.strip() else None

        # Validate that allowed models are pinned (fail closed)
        missing = [m for m in set(allowed + [primary] + ([fallback_norm] if fallback_norm else [])) if m and m not in pinned]
        if missing:
            raise ValueError(f"capability_slots['{slot_name}'] references unpinned models {sorted(missing)}: {mf_path}")

        slots[slot_name.strip()] = SlotRouting(
            name=slot_name.strip(),
            allowed_models=allowed,
            primary=primary,
            fallback=fallback_norm,
        )

    return ModelFamily(
        path=mf_path,
        sealed=sealed,
        sealed_at_utc=sealed_at_utc,
        sealed_by=sealed_by,
        capability_slots=slots,
        pinned_artifacts=pinned,
        raw=data,
    )


def llm_endpoints() -> Tuple[str, str]:
    """
    Returns (ollama_base_url, lmstudio_base_url).

    - Ollama expects: http://host:11434
    - LM Studio expects: http://host:1234/v1 (OpenAI compatible)
    """
    ollama = (
        os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_URL")
        or os.getenv("BIZRA_OLLAMA_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).strip()
    lmstudio = (
        os.getenv("LMSTUDIO_BASE_URL")
        or os.getenv("LMSTUDIO_URL")
        or os.getenv("BIZRA_LMSTUDIO_URL")
        or "http://127.0.0.1:1234/v1"
    ).strip()
    return ollama, lmstudio
