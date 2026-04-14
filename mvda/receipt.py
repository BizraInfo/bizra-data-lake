"""MVDA Development Receipt — non-canonical, hash-chained."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MvdaReceipt:
    ledger_class: str = "mvda_dev"
    canonical: bool = False
    timestamp: float = field(default_factory=time.time)
    actor: str = ""
    step: str = ""
    status: str = ""
    verdict: str = ""
    reason: str = ""
    ihsan_score: float = 0.0
    evidence_refs: List[str] = field(default_factory=list)
    evidence_sufficient: bool = False
    content_hash: str = ""
    prev_hash: str = ""
    receipt_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        payload = json.dumps(
            {
                "ledger_class": self.ledger_class,
                "timestamp": self.timestamp,
                "actor": self.actor,
                "step": self.step,
                "status": self.status,
                "verdict": self.verdict,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
        ).encode()
        self.receipt_hash = hashlib.blake2b(payload, digest_size=32).hexdigest()
        return self.receipt_hash

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
