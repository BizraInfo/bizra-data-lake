"""
GOLDEN GEM #4: IHSĀN AS CIRCUIT CONSTRAINT
═══════════════════════════════════════════

Ethics as structural constraint, not advisory.

Like a circuit breaker: it doesn't tell you about overload,
it PREVENTS overload.

The 5 dimensions:
- Correctness (25%): Does it work as specified?
- Safety (25%): Does it protect?
- Beneficence (20%): Does it help?
- Transparency (15%): Can we understand it?
- Sustainability (15%): Can we maintain it?

SNR Score: 0.91
"""

from dataclasses import dataclass
from typing import Dict, Callable, Any, Optional


@dataclass
class IhsanVector:
    """The 5-dimensional ethical state."""
    correctness: float = 0.0
    safety: float = 0.0
    beneficence: float = 0.0
    transparency: float = 0.0
    sustainability: float = 0.0
    
    @property
    def composite(self) -> float:
        """Weighted composite score."""
        weights = {
            "correctness": 0.25,
            "safety": 0.25,
            "beneficence": 0.20,
            "transparency": 0.15,
            "sustainability": 0.15,
        }
        return sum(
            getattr(self, dim) * weight
            for dim, weight in weights.items()
        )
    
    @property
    def minimum(self) -> float:
        """Minimum dimension value."""
        return min(
            self.correctness,
            self.safety,
            self.beneficence,
            self.transparency,
            self.sustainability,
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "correctness": self.correctness,
            "safety": self.safety,
            "beneficence": self.beneficence,
            "transparency": self.transparency,
            "sustainability": self.sustainability,
            "composite": self.composite,
            "minimum": self.minimum,
        }


class IhsanViolation(Exception):
    """Raised when an operation violates Ihsān constraints."""
    pass


class IhsanCircuit:
    """
    The FATE Gate — Ethics as Circuit Breaker.
    
    This is NOT a scoring system. It's a constraint system.
    Operations that violate constraints are not scored low —
    they are IMPOSSIBLE.
    """
    
    def __init__(
        self,
        min_threshold: float = 0.70,
        min_composite: float = 0.80,
    ):
        self.min_threshold = min_threshold
        self.min_composite = min_composite
        self.blocked_count = 0
        self.passed_count = 0
    
    def gate(self, vector: IhsanVector) -> bool:
        """
        The FATE Gate.
        
        Returns True if operation is permitted.
        Returns False if operation is blocked.
        """
        # Check per-dimension minimum
        if vector.minimum < self.min_threshold:
            self.blocked_count += 1
            return False
        
        # Check composite
        if vector.composite < self.min_composite:
            self.blocked_count += 1
            return False
        
        self.passed_count += 1
        return True
    
    def constrain(
        self,
        operation: Callable[..., Any],
        vector: IhsanVector,
        *args,
        **kwargs,
    ) -> Optional[Any]:
        """
        Execute operation only if it passes the gate.
        
        This is the structural constraint — the operation
        literally cannot execute if ethics are violated.
        """
        if not self.gate(vector):
            return None
        
        return operation(*args, **kwargs)
    
    def require(self, vector: IhsanVector) -> None:
        """
        Require gate to pass or raise exception.
        
        Use this for hard failures.
        """
        if not self.gate(vector):
            # Undo the blocked_count increment from gate()
            self.blocked_count -= 1
            raise IhsanViolation(
                f"Ihsān violation: composite={vector.composite:.2f}, "
                f"minimum={vector.minimum:.2f}"
            )
    
    def wrap(self, vector_fn: Callable[..., IhsanVector]):
        """
        Decorator to wrap any function with Ihsān constraint.
        
        Usage:
            @circuit.wrap(compute_ihsan)
            def dangerous_operation():
                ...
        """
        def decorator(fn: Callable):
            def wrapped(*args, **kwargs):
                vector = vector_fn(*args, **kwargs)
                if not self.gate(vector):
                    raise IhsanViolation(f"Blocked: {vector.to_dict()}")
                return fn(*args, **kwargs)
            return wrapped
        return decorator
    
    def stats(self) -> Dict:
        """Circuit statistics."""
        total = self.blocked_count + self.passed_count
        return {
            "blocked": self.blocked_count,
            "passed": self.passed_count,
            "block_rate": self.blocked_count / total if total > 0 else 0,
        }


# === Pre-built vectors for common scenarios ===

def safe_operation() -> IhsanVector:
    """Vector for a typical safe operation."""
    return IhsanVector(
        correctness=0.9,
        safety=0.9,
        beneficence=0.8,
        transparency=0.85,
        sustainability=0.8,
    )


def unsafe_operation() -> IhsanVector:
    """Vector for an unsafe operation (will be blocked)."""
    return IhsanVector(
        correctness=0.9,
        safety=0.5,  # Below threshold
        beneficence=0.8,
        transparency=0.85,
        sustainability=0.8,
    )


def demo():
    """Demonstrate ethics as structural constraint."""
    
    circuit = IhsanCircuit(min_threshold=0.70, min_composite=0.80)
    
    # Good operation
    good = safe_operation()
    print("=== GOOD OPERATION ===")
    print(f"Vector: composite={good.composite:.2f}, minimum={good.minimum:.2f}")
    print(f"Gate result: {circuit.gate(good)}")
    
    # Unsafe operation
    unsafe = unsafe_operation()
    print("\n=== UNSAFE OPERATION ===")
    print(f"Vector: composite={unsafe.composite:.2f}, minimum={unsafe.minimum:.2f}")
    print(f"Gate result: {circuit.gate(unsafe)}")
    
    # Constrained execution
    def send_email(to: str) -> str:
        return f"Email sent to {to}"
    
    print("\n=== CONSTRAINED EXECUTION ===")
    result = circuit.constrain(send_email, good, "user@example.com")
    print(f"With good vector: {result}")
    
    result = circuit.constrain(send_email, unsafe, "user@example.com")
    print(f"With unsafe vector: {result}")
    
    # Stats
    print(f"\n=== STATS ===")
    print(circuit.stats())


if __name__ == "__main__":
    demo()
