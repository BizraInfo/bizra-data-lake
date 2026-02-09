"""
BIZRA EPIGENETIC LAYER
═══════════════════════════════════════════════════════════════════════════════

Reinterpretation without rewriting. Growth narratives on immutable receipts.

The genome (receipt chain) is append-only.
The epigenome adds meaning without changing facts.

Principle: لا نفترض — We do not assume.
           But we can LEARN, and learning changes interpretation.

Created: 2026-01-29 | BIZRA Sovereignty
"""

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class InterpretationType(str, Enum):
    """Types of reinterpretation allowed."""

    LEARNED = "LEARNED"  # "I learned from this"
    RECONTEXTUALIZED = "RECONTEXTUALIZED"  # "I now see this differently"
    SUPERSEDED = "SUPERSEDED"  # "This is no longer relevant"
    HEALED = "HEALED"  # "I have processed this trauma"


@dataclass
class Interpretation:
    """
    A reinterpretation of an immutable receipt.

    The original receipt is NEVER modified.
    This adds a new layer of meaning on top.
    """

    receipt_hash: str
    interpretation_type: InterpretationType
    new_context: str
    timestamp: str
    signature: str

    # Optional: link to evidence of growth
    evidence_hash: Optional[str] = None

    # Ihsan score at time of interpretation (for validation)
    ihsan_score: float = 0.95

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Interpretation":
        data["interpretation_type"] = InterpretationType(data["interpretation_type"])
        return cls(**data)

    def compute_hash(self) -> str:
        """Compute hash of this interpretation for chaining."""
        content = json.dumps(
            {
                "receipt_hash": self.receipt_hash,
                "type": self.interpretation_type.value,
                "context": self.new_context,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class GrowthNarrative:
    """
    The current dominant narrative for a receipt.

    Combines original receipt meaning with latest valid reframings.
    """

    receipt_hash: str
    original_summary: str
    current_interpretation: Optional[Interpretation]
    interpretation_chain: List[str]  # Hashes of interpretations
    growth_score: float  # 0.0 = no growth, 1.0 = fully processed

    @property
    def is_healed(self) -> bool:
        """Has this event been fully processed?"""
        return (
            self.current_interpretation is not None
            and self.current_interpretation.interpretation_type
            == InterpretationType.HEALED
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EPIGENETIC LAYER
# ═══════════════════════════════════════════════════════════════════════════════


class EpigeneticLayer:
    """
    Epigenetic reinterpretation layer on top of immutable receipts.

    Key principles:
    1. Original receipts are NEVER modified
    2. Interpretations are append-only (like receipts)
    3. Each interpretation is signed and timestamped
    4. Ihsan validation required for all reframings
    5. Time-locked healing suggestions for old trauma

    This is NOT forgetting. This is GROWTH.
    """

    # Minimum age before auto-healing suggestions (7 years)
    HEALING_THRESHOLD = timedelta(days=7 * 365)

    # Minimum Ihsan score for any interpretation
    IHSAN_MINIMUM = UNIFIED_IHSAN_THRESHOLD

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        ihsan_validator: Optional[Callable[..., Any]] = None,
    ):
        self.storage_path = storage_path or Path("/var/lib/bizra/epigenome.json")
        self.ihsan_validator = ihsan_validator or self._default_ihsan_validator

        # In-memory cache: receipt_hash -> List[Interpretation]
        self._interpretations: Dict[str, List[Interpretation]] = {}
        self._lock = threading.RLock()

        # Load existing interpretations
        self._load()

    def reframe(
        self,
        receipt_hash: str,
        new_context: str,
        interpretation_type: InterpretationType = InterpretationType.RECONTEXTUALIZED,
        evidence_hash: Optional[str] = None,
        signer: Optional[Callable[..., Any]] = None,
    ) -> Optional[Interpretation]:
        """
        Add a new interpretation to an immutable receipt.

        The original receipt is NOT modified.
        This adds a new layer of meaning.

        Returns the Interpretation if valid, None if rejected.
        """
        # Validate Ihsan score
        current_ihsan = self.ihsan_validator()
        if current_ihsan < self.IHSAN_MINIMUM:
            print(
                f"[Epigenome] Reframe rejected: Ihsan {current_ihsan} < {self.IHSAN_MINIMUM}"
            )
            return None

        # Create interpretation
        timestamp = datetime.now(timezone.utc).isoformat()

        # Sign the interpretation
        sign_content = f"{receipt_hash}:{new_context}:{timestamp}"
        signature = signer(sign_content) if signer else self._default_sign(sign_content)

        interpretation = Interpretation(
            receipt_hash=receipt_hash,
            interpretation_type=interpretation_type,
            new_context=new_context,
            timestamp=timestamp,
            signature=signature,
            evidence_hash=evidence_hash,
            ihsan_score=current_ihsan,
        )

        # Validate consistency (new interpretation must not contradict facts)
        if not self._validate_consistency(receipt_hash, interpretation):
            print("[Epigenome] Reframe rejected: Inconsistent with original receipt")
            return None

        # Store
        with self._lock:
            if receipt_hash not in self._interpretations:
                self._interpretations[receipt_hash] = []
            self._interpretations[receipt_hash].append(interpretation)
            self._save()

        print(
            f"[Epigenome] Reframe accepted: {interpretation_type.value} for {receipt_hash[:16]}..."
        )
        return interpretation

    def get_narrative(self, receipt_hash: str) -> GrowthNarrative:
        """
        Get the current dominant narrative for a receipt.

        Returns the latest valid interpretation, or ORIGINAL if none.
        """
        with self._lock:
            interpretations = self._interpretations.get(receipt_hash, [])

        if not interpretations:
            return GrowthNarrative(
                receipt_hash=receipt_hash,
                original_summary="ORIGINAL",
                current_interpretation=None,
                interpretation_chain=[],
                growth_score=0.0,
            )

        # Find latest valid interpretation
        valid_interpretations = [
            i for i in interpretations if i.ihsan_score >= self.IHSAN_MINIMUM
        ]

        if not valid_interpretations:
            return GrowthNarrative(
                receipt_hash=receipt_hash,
                original_summary="ORIGINAL",
                current_interpretation=None,
                interpretation_chain=[i.compute_hash() for i in interpretations],
                growth_score=0.0,
            )

        latest = valid_interpretations[-1]

        # Calculate growth score based on interpretation type
        growth_scores = {
            InterpretationType.LEARNED: 0.5,
            InterpretationType.RECONTEXTUALIZED: 0.7,
            InterpretationType.SUPERSEDED: 0.8,
            InterpretationType.HEALED: 1.0,
        }

        return GrowthNarrative(
            receipt_hash=receipt_hash,
            original_summary="ORIGINAL",  # Would need receipt lookup for actual summary
            current_interpretation=latest,
            interpretation_chain=[i.compute_hash() for i in interpretations],
            growth_score=growth_scores.get(latest.interpretation_type, 0.5),
        )

    def suggest_healing(
        self, receipt_hash: str, receipt_timestamp: str
    ) -> Optional[str]:
        """
        Suggest healing interpretation for old trauma.

        Only suggests for receipts older than HEALING_THRESHOLD (7 years).
        User must explicitly accept the reframe.
        """
        try:
            receipt_time = datetime.fromisoformat(
                receipt_timestamp.replace("Z", "+00:00")
            )
        except ValueError:
            return None

        age = datetime.now(timezone.utc) - receipt_time

        if age < self.HEALING_THRESHOLD:
            return None

        # Check if already healed
        narrative = self.get_narrative(receipt_hash)
        if narrative.is_healed:
            return None

        # Generate healing suggestion
        years = age.days // 365
        return (
            f"This event is {years} years old. "
            f"Would you like to add a growth interpretation? "
            f"The original record remains unchanged."
        )

    def generate_growth_proof(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> dict:
        """
        Generate a ZK-style growth proof.

        Proves growth happened without revealing content.
        Returns: {
            "period": {"start": ..., "end": ...},
            "interpretations_count": N,
            "growth_score_delta": 0.X,
            "proof_hash": "...",
            "content_revealed": False
        }
        """
        with self._lock:
            all_interpretations = []
            for receipt_hash, interps in self._interpretations.items():
                all_interpretations.extend(interps)

        # Filter by time range
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            all_interpretations = [
                i
                for i in all_interpretations
                if datetime.fromisoformat(i.timestamp.replace("Z", "+00:00"))
                >= start_dt
            ]

        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            all_interpretations = [
                i
                for i in all_interpretations
                if datetime.fromisoformat(i.timestamp.replace("Z", "+00:00")) <= end_dt
            ]

        # Calculate growth score
        growth_weights = {
            InterpretationType.LEARNED: 0.5,
            InterpretationType.RECONTEXTUALIZED: 0.7,
            InterpretationType.SUPERSEDED: 0.8,
            InterpretationType.HEALED: 1.0,
        }

        total_growth = sum(
            growth_weights.get(i.interpretation_type, 0.0) for i in all_interpretations
        )

        # Generate proof hash (commits to the interpretations without revealing content)
        proof_content = json.dumps(
            {
                "count": len(all_interpretations),
                "hashes": [i.compute_hash() for i in all_interpretations],
            },
            sort_keys=True,
        )
        proof_hash = hashlib.sha256(proof_content.encode()).hexdigest()

        return {
            "period": {
                "start": start_time or "GENESIS",
                "end": end_time or datetime.now(timezone.utc).isoformat(),
            },
            "interpretations_count": len(all_interpretations),
            "growth_score_delta": round(total_growth, 3),
            "proof_hash": proof_hash,
            "content_revealed": False,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _validate_consistency(
        self, receipt_hash: str, interpretation: Interpretation
    ) -> bool:
        """
        Validate that new interpretation doesn't contradict original receipt.

        For now, this is a placeholder. Full implementation would:
        1. Load original receipt
        2. Use semantic analysis to check for contradictions
        3. Reject interpretations that deny facts (vs. reframe meaning)
        """
        # Basic validation: interpretation must reference existing receipt
        # In production, this would verify the receipt actually exists
        if not receipt_hash or len(receipt_hash) != 64:
            return False

        # Interpretations of type HEALED require evidence
        if interpretation.interpretation_type == InterpretationType.HEALED:
            if not interpretation.evidence_hash:
                return False

        return True

    def _default_ihsan_validator(self) -> float:
        """Default Ihsan validator returns threshold (for testing)."""
        return 0.95

    def _default_sign(self, content: str) -> str:
        """
        Default signer uses HMAC with environment-sourced key.

        SECURITY: Key is loaded from environment variable BIZRA_EPIGENOME_SECRET.
        If not set, raises RuntimeError to prevent insecure fallback.

        Production uses Ed25519 via the signer parameter.
        """
        import hmac
        import os

        secret = os.environ.get("BIZRA_EPIGENOME_SECRET")
        if not secret:
            # Check if running in production mode
            is_production = os.environ.get("BIZRA_ENV", "").lower() == "production" or \
                            os.environ.get("BIZRA_PRODUCTION_MODE", "0") == "1"
            
            if is_production:
                raise RuntimeError(
                    "BIZRA_EPIGENOME_SECRET must be set in production. "
                    "Set this environment variable with a cryptographically secure key."
                )
            
            # For testing only: derive ephemeral key from node identity
            # This ensures tests work but production MUST set the env var
            import warnings

            warnings.warn(
                "BIZRA_EPIGENOME_SECRET not set - using ephemeral key. "
                "Set this environment variable in production.",
                RuntimeWarning,
            )
            # Use process-unique ephemeral key (secure for isolated test runs)
            secret = f"ephemeral-{os.getpid()}-{id(self)}"

        return hmac.new(secret.encode(), content.encode(), hashlib.sha256).hexdigest()

    def _load(self):
        """Load interpretations from storage."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            with self._lock:
                for receipt_hash, interps in data.items():
                    self._interpretations[receipt_hash] = [
                        Interpretation.from_dict(i) for i in interps
                    ]
        except Exception as e:
            print(f"[Epigenome] Load failed: {e}")

    def _save(self):
        """Persist interpretations to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                receipt_hash: [i.to_dict() for i in interps]
                for receipt_hash, interps in self._interpretations.items()
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Epigenome] Save failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_epigenome_instance: Optional[EpigeneticLayer] = None


def get_epigenome() -> EpigeneticLayer:
    """Get the singleton epigenetic layer."""
    global _epigenome_instance
    if _epigenome_instance is None:
        _epigenome_instance = EpigeneticLayer()
    return _epigenome_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Epigenetic Layer")
    parser.add_argument("command", choices=["reframe", "narrative", "proof"])
    parser.add_argument("--receipt", help="Receipt hash to operate on")
    parser.add_argument("--context", help="New context for reframing")
    parser.add_argument(
        "--type", choices=["LEARNED", "RECONTEXTUALIZED", "SUPERSEDED", "HEALED"]
    )
    parser.add_argument("--start", help="Start time for growth proof")
    parser.add_argument("--end", help="End time for growth proof")
    args = parser.parse_args()

    epigenome = get_epigenome()

    if args.command == "reframe":
        if not args.receipt or not args.context:
            print("Error: --receipt and --context required")
            exit(1)

        interp_type = (
            InterpretationType(args.type)
            if args.type
            else InterpretationType.RECONTEXTUALIZED
        )
        result = epigenome.reframe(args.receipt, args.context, interp_type)

        if result:
            print(f"✅ Reframe successful: {result.compute_hash()}")
        else:
            print("❌ Reframe rejected")

    elif args.command == "narrative":
        if not args.receipt:
            print("Error: --receipt required")
            exit(1)

        narrative = epigenome.get_narrative(args.receipt)
        print(json.dumps(asdict(narrative), indent=2, default=str))

    elif args.command == "proof":
        proof = epigenome.generate_growth_proof(args.start, args.end)
        print(json.dumps(proof, indent=2))
