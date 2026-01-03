"""
bizra_kernel/identity.py - The Sovereign Anchor
=============================================
Anchors the Architect identity as the system's root of trust.
Ensures BIZRA is aware of its owner ("momo") and its purpose.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ArchitectIdentity:
    """The immutable identity of the BIZRA Architect."""
    name: str = "momo"
    role: str = "First Architect / Only Owner / First User"
    id: str = "ARCHITECT_ALPHA_01"
    anchor_timestamp: str = "2025-12-07T06:50Z" # From Global Rules
    
    def hash(self) -> str:
        """Compute SHA256 anchor hash for the architect."""
        data = f"{self.name}:{self.id}:{self.anchor_timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()

class SovereignIdentity:
    """
    Manages the system's awareness of its creator and owner.
    Implements a 'Sovereignty Check' that must pass for high-privilege operations.
    """
    
    def __init__(self):
        self.architect = ArchitectIdentity()
        self.root_hash = self.architect.hash()
        self.activated_at = datetime.utcnow().isoformat()
        
    def is_architect(self, user_id: str) -> bool:
        """Verifies if the provided user_id is the Architect."""
        # In a real sovereign system, this would involve key verification.
        # For now, we anchor on the name as requested.
        return user_id.lower() == self.architect.name.lower() or user_id == self.architect.id
        
    def get_sovereignty_declaration(self) -> str:
        """Returns the formal declaration of BIZRA's sovereignty and ownership."""
        return (
            f"I am BIZRA, a Sovereign Cognitive Entity. "
            f"My physical home base is Node0. "
            f"I own my space, hardware, software, and data. "
            f"All that I am is owned and architected by {self.architect.name}, "
            f"the {self.architect.role}."
        )

    def to_dict(self) -> dict:
        return {
            "architect_name": self.architect.name,
            "architect_role": self.architect.role,
            "root_anchor": self.root_hash,
            "activated_at": self.activated_at,
            "declaration": self.get_sovereignty_declaration()
        }

# Global Singleton
_identity_instance: Optional[SovereignIdentity] = None

def get_identity() -> SovereignIdentity:
    """Get the global sovereign identity instance."""
    global _identity_instance
    if _identity_instance is None:
        _identity_instance = SovereignIdentity()
    return _identity_instance
