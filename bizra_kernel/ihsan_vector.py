from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml


def _normalize_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


ThresholdCombine = Literal["max", "min"]


# ==============================================================================
# IhsanDimension Enum (for backward compatibility with kernel.py)
# ==============================================================================
class IhsanDimension(Enum):
    """The 8 dimensions of the Ihsān Vector."""
    CORRECTNESS = "correctness"
    SAFETY = "safety"
    USER_BENEFIT = "user_benefit"
    EFFICIENCY = "efficiency"
    AUDITABILITY = "auditability"
    ANTI_CENTRALIZATION = "anti_centralization"
    ROBUSTNESS = "robustness"
    ADL_FAIRNESS = "adl_fairness"


# ==============================================================================
# IhsanConstitution - YAML-loaded constitution (single source of truth)
# ==============================================================================
@dataclass(frozen=True)
class IhsanConstitution:
    id: str
    version: int
    threshold: float
    score_min: float
    score_max: float
    weights: Dict[str, float]
    combine: ThresholdCombine
    default_env: str
    thresholds_by_env: Dict[str, float]
    thresholds_by_artifact_class: Dict[str, float]
    env_aliases: Dict[str, str]
    artifact_class_aliases: Dict[str, str]

    def canonical_env(self, env_name: str) -> str:
        key = _normalize_key(env_name) or _normalize_key(self.default_env)
        return self.env_aliases.get(key, key)

    def canonical_artifact_class(self, artifact_class: str) -> str:
        key = _normalize_key(artifact_class)
        return self.artifact_class_aliases.get(key, key)

    def threshold_for(self, env_name: str, artifact_class: str) -> float:
        env_key = self.canonical_env(env_name)
        artifact_key = self.canonical_artifact_class(artifact_class)

        candidates: list[float] = []
        if env_key in self.thresholds_by_env:
            candidates.append(float(self.thresholds_by_env[env_key]))
        if artifact_key in self.thresholds_by_artifact_class:
            candidates.append(float(self.thresholds_by_artifact_class[artifact_key]))

        if not candidates:
            return float(self.threshold)
        return min(candidates) if self.combine == "min" else max(candidates)


_CACHED: Optional[IhsanConstitution] = None


def _load_constitution() -> IhsanConstitution:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "constitution" / "ihsan_v1.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"ihsan constitution must be a mapping: {path}")

    cid = str(data.get("id") or "ihsan_v1")
    version = int(data.get("version") or 1)

    units = data.get("units") or {}
    if not isinstance(units, dict):
        raise ValueError(f"ihsan constitution units must be a mapping: {path}")
    score_range = units.get("score_range") or [0.0, 1.0]
    if not (isinstance(score_range, list) and len(score_range) == 2):
        raise ValueError(f"ihsan units.score_range must be [min,max]: {path}")
    score_min = float(score_range[0])
    score_max = float(score_range[1])
    threshold = float(units.get("threshold"))

    dims = data.get("dimensions") or {}
    if not isinstance(dims, dict) or not dims:
        raise ValueError(f"ihsan dimensions must be a non-empty mapping: {path}")
    weights: Dict[str, float] = {}
    for k, v in dims.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        if "weight" not in v:
            continue
        weights[k] = float(v["weight"])

    inv = data.get("invariants") or {}
    expected_sum = 1.0
    if isinstance(inv, dict):
        try:
            if inv.get("weights_sum") is not None:
                expected_sum = float(inv["weights_sum"])
        except Exception:
            expected_sum = 1.0
    if abs(sum(weights.values()) - expected_sum) > 1e-9:
        raise ValueError(f"ihsan weights must sum to {expected_sum} (got {sum(weights.values())})")

    policy = data.get("threshold_policy") or {}
    if not isinstance(policy, dict):
        policy = {}

    combine_raw = str(policy.get("combine") or "max").strip().lower()
    combine: ThresholdCombine = "min" if combine_raw == "min" else "max"
    default_env = str(policy.get("default_env") or "development").strip() or "development"

    thresholds_by_env_raw = policy.get("thresholds_by_env") or {}
    thresholds_by_artifact_raw = policy.get("thresholds_by_artifact_class") or {}

    thresholds_by_env: Dict[str, float] = {}
    if isinstance(thresholds_by_env_raw, dict):
        for k, v in thresholds_by_env_raw.items():
            if not isinstance(k, str):
                continue
            thresholds_by_env[_normalize_key(k)] = float(v)

    thresholds_by_artifact: Dict[str, float] = {}
    if isinstance(thresholds_by_artifact_raw, dict):
        for k, v in thresholds_by_artifact_raw.items():
            if not isinstance(k, str):
                continue
            thresholds_by_artifact[_normalize_key(k)] = float(v)

    normalization = policy.get("normalization") or {}
    if not isinstance(normalization, dict):
        normalization = {}

    env_aliases_raw = normalization.get("env_aliases") or {}
    artifact_aliases_raw = normalization.get("artifact_class_aliases") or {}

    env_aliases: Dict[str, str] = {}
    if isinstance(env_aliases_raw, dict):
        for k, v in env_aliases_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                env_aliases[_normalize_key(k)] = _normalize_key(v)

    artifact_aliases: Dict[str, str] = {}
    if isinstance(artifact_aliases_raw, dict):
        for k, v in artifact_aliases_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                artifact_aliases[_normalize_key(k)] = _normalize_key(v)

    if not (score_min <= threshold <= score_max):
        raise ValueError(f"ihsan threshold {threshold} outside score_range [{score_min},{score_max}]")

    return IhsanConstitution(
        id=cid,
        version=version,
        threshold=threshold,
        score_min=score_min,
        score_max=score_max,
        weights=weights,
        combine=combine,
        default_env=default_env,
        thresholds_by_env=thresholds_by_env,
        thresholds_by_artifact_class=thresholds_by_artifact,
        env_aliases=env_aliases,
        artifact_class_aliases=artifact_aliases,
    )


def constitution() -> IhsanConstitution:
    global _CACHED
    if _CACHED is None:
        _CACHED = _load_constitution()
    return _CACHED


def constitution_snapshot() -> Dict[str, object]:
    c = constitution()
    return {
        "id": c.id,
        "version": c.version,
        "threshold": c.threshold,
        "weights": dict(c.weights),
        "default_env": c.default_env,
        "combine": c.combine,
    }


def score_plain(scores: Dict[str, float]) -> float:
    c = constitution()
    total = 0.0
    for dim, weight in c.weights.items():
        total += float(weight) * float(scores.get(dim, 0.0))
    return float(total)


def threshold_for(env_name: str, artifact_class: str) -> float:
    return float(constitution().threshold_for(env_name, artifact_class))


# ==============================================================================
# IHSAN_WEIGHTS and IHSAN_THRESHOLD (backward compatibility)
# ==============================================================================
def _get_ihsan_weights() -> Dict[IhsanDimension, float]:
    """Load weights from constitution and map to IhsanDimension enum."""
    c = constitution()
    weights: Dict[IhsanDimension, float] = {}
    for dim in IhsanDimension:
        weights[dim] = c.weights.get(dim.value, 0.0)
    return weights


# Lazy-loaded weights (populated on first access)
IHSAN_WEIGHTS: Dict[IhsanDimension, float] = {}

def _ensure_weights():
    global IHSAN_WEIGHTS
    if not IHSAN_WEIGHTS:
        IHSAN_WEIGHTS.update(_get_ihsan_weights())


# Threshold from constitution (0.70 default, may differ by env/artifact)
IHSAN_THRESHOLD = 0.70  # Will be overridden by constitution on load


# ==============================================================================
# IhsanVector class (backward compatibility with kernel.py)
# ==============================================================================
@dataclass
class IhsanVector:
    """
    Ihsān Vector - 8-dimensional ethical scoring.
    
    Wraps the YAML-loaded constitution for backward compatibility with
    code that expects the class-based API.
    """
    scores: Dict[IhsanDimension, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    
    def __post_init__(self):
        _ensure_weights()
        # Initialize all dimensions to 0.5 (neutral) if not provided
        for dim in IhsanDimension:
            if dim not in self.scores:
                self.scores[dim] = 0.5
    
    def set_dimension(self, dimension: IhsanDimension, score: float) -> None:
        """Set a single dimension score (0.0 to 1.0)."""
        self.scores[dimension] = max(0.0, min(1.0, score))
    
    def get_dimension(self, dimension: IhsanDimension) -> float:
        """Get a single dimension score."""
        return self.scores.get(dimension, 0.5)

    def set_score(self, dimension: IhsanDimension, score: float) -> None:
        """Alias for set_dimension for backward compatibility."""
        self.set_dimension(dimension, score)
    
    @property
    def composite_score(self) -> float:
        """Alias for calculate_score for property-style access."""
        return self.calculate_score()
    
    def calculate_score(self) -> float:
        """Calculate weighted sum: I_vec = Σ(w_i × d_i)."""
        _ensure_weights()
        return sum(
            IHSAN_WEIGHTS.get(dim, 0.0) * score
            for dim, score in self.scores.items()
        )
    
    @property
    def passes_threshold(self) -> bool:
        """Property-style access for passes_threshold (aliased in kernel)."""
        return self.calculate_score() >= constitution().threshold

    def to_dict(self) -> Dict[str, float]:
        """Export scores as string-keyed dict."""
        return {dim.value: score for dim, score in self.scores.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "IhsanVector":
        """Import scores from a string-keyed dict."""
        if not data:
            return cls()
        key_map = {_normalize_key(dim.value): dim for dim in IhsanDimension}
        scores: Dict[IhsanDimension, float] = {}
        for key, value in data.items():
            if isinstance(key, IhsanDimension):
                scores[key] = float(value)
                continue
            if not isinstance(key, str):
                continue
            dim = key_map.get(_normalize_key(key))
            if dim is not None:
                scores[dim] = float(value)
        return cls(scores=scores)

    @classmethod
    def from_agent_response(
        cls, 
        response: str, 
        latency_ms: int = 0, 
        token_count: int = 0, 
        rag_used: bool = False,
        agent_role: str = "generic"
    ) -> "IhsanVector":
        """
        Heuristic-based IhsanVector initialization from agent metadata.
        Used by the SystemProtocolKernel.
        """
        vec = cls()
        # Heuristic 1: Efficiency (latency-based)
        if latency_ms > 0:
            efficiency = 1.0 - min(1.0, latency_ms / 5000.0) # 5s limit for 0 score
            vec.set_dimension(IhsanDimension.EFFICIENCY, efficiency)
        
        # Heuristic 2: Robustness (presence of structure)
        if "{" in response and "}" in response: # JSON-like
            vec.set_dimension(IhsanDimension.ROBUSTNESS, 0.9)
            
        # Heuristic 3: User Benefit (RAG used usually improves context)
        if rag_used:
            vec.set_dimension(IhsanDimension.USER_BENEFIT, 0.95)
            
        return vec
    
    def snapshot(self) -> None:
        """Save current state to history."""
        self.history.append(self.to_dict())
    
    def __repr__(self) -> str:
        score = self.calculate_score()
        return f"IhsanVector(score={score:.4f}, passes={self.passes_threshold()})"

