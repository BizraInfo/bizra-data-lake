"""
GOLDEN GEM #7: ALGEBRAIC EFFECTS
═════════════════════════════════

Kong Gateway Internalized — Routing and middleware
as composable effect handlers.

Effects = things that happen (logging, auth, retry, rate-limit)
Handlers = how they're handled (can be swapped, composed, mocked)

SNR Score: 0.87
"""

from typing import Any, Callable, Dict, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import asyncio


# === EFFECT DEFINITIONS ===

@dataclass
class Effect(ABC):
    """Base class for all effects."""
    name: str
    
    @abstractmethod
    def describe(self) -> str:
        pass


@dataclass
class LogEffect(Effect):
    """Request to log something."""
    message: str
    level: str = "info"
    
    def describe(self) -> str:
        return f"Log({self.level}): {self.message}"


@dataclass
class AuthEffect(Effect):
    """Request to authenticate."""
    token: str
    required_roles: List[str] = field(default_factory=list)
    
    def describe(self) -> str:
        return f"Auth: token={self.token[:8]}... roles={self.required_roles}"


@dataclass
class RateLimitEffect(Effect):
    """Request to check rate limit."""
    key: str
    limit: int
    window_seconds: int
    
    def describe(self) -> str:
        return f"RateLimit: {self.key} ({self.limit}/{self.window_seconds}s)"


@dataclass
class IhsanEffect(Effect):
    """Request to check Ihsān constraints."""
    vector: Dict[str, float]
    min_threshold: float = 0.7
    
    def describe(self) -> str:
        composite = sum(self.vector.values()) / len(self.vector) if self.vector else 0
        return f"Ihsan: min={self.min_threshold}, composite={composite:.2f}"


# === EFFECT HANDLERS ===

class EffectHandler(ABC):
    """Base class for effect handlers."""
    
    @abstractmethod
    def can_handle(self, effect: Effect) -> bool:
        pass
    
    @abstractmethod
    async def handle(self, effect: Effect) -> Any:
        pass


class LogHandler(EffectHandler):
    """Handles logging effects."""
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, LogEffect)
    
    async def handle(self, effect: LogEffect) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{effect.level.upper()}] {effect.message}")


class AuthHandler(EffectHandler):
    """Handles authentication effects."""
    
    def __init__(self, valid_tokens: Dict[str, List[str]] = None):
        self.valid_tokens = valid_tokens or {
            "bizra_secret_123": ["admin", "user"],
            "user_token_456": ["user"],
        }
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, AuthEffect)
    
    async def handle(self, effect: AuthEffect) -> bool:
        if effect.token not in self.valid_tokens:
            return False
        
        user_roles = self.valid_tokens[effect.token]
        if effect.required_roles:
            return all(role in user_roles for role in effect.required_roles)
        
        return True


class RateLimitHandler(EffectHandler):
    """Handles rate limiting effects."""
    
    def __init__(self):
        self.buckets: Dict[str, List[float]] = {}
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, RateLimitEffect)
    
    async def handle(self, effect: RateLimitEffect) -> bool:
        now = time.time()
        key = effect.key
        
        if key not in self.buckets:
            self.buckets[key] = []
        
        # Remove old entries
        self.buckets[key] = [
            t for t in self.buckets[key]
            if now - t < effect.window_seconds
        ]
        
        if len(self.buckets[key]) >= effect.limit:
            return False
        
        self.buckets[key].append(now)
        return True


class IhsanHandler(EffectHandler):
    """Handles Ihsān constraint effects."""
    
    def can_handle(self, effect: Effect) -> bool:
        return isinstance(effect, IhsanEffect)
    
    async def handle(self, effect: IhsanEffect) -> bool:
        for score in effect.vector.values():
            if score < effect.min_threshold:
                return False
        return True


# === EFFECT RUNTIME ===

class EffectBlocked(Exception):
    """Raised when an effect blocks an operation."""
    pass


class EffectRuntime:
    """
    The algebraic effect runtime.
    
    This replaces external gateways with an internal,
    composable effect system.
    
    Benefits:
    - No external service dependency
    - Composable (effects can trigger other effects)
    - Testable (handlers can be mocked)
    - Type-safe (effects are typed)
    """
    
    def __init__(self):
        self.handlers: List[EffectHandler] = []
    
    def register(self, handler: EffectHandler):
        """Register an effect handler."""
        self.handlers.append(handler)
    
    async def perform(self, effect: Effect) -> Any:
        """
        Perform an effect.
        
        Finds a handler and executes it.
        """
        for handler in self.handlers:
            if handler.can_handle(effect):
                return await handler.handle(effect)
        
        raise ValueError(f"No handler for effect: {effect.name}")
    
    async def run_with_effects(
        self,
        operation: Callable[..., Any],
        effects: List[Effect],
        *args,
        **kwargs,
    ) -> Any:
        """
        Run an operation with a chain of effects.
        
        Each effect must succeed for the operation to proceed.
        """
        for effect in effects:
            result = await self.perform(effect)
            
            # Log effects always succeed
            if isinstance(effect, LogEffect):
                continue
            
            # Other effects can block
            if result is False:
                raise EffectBlocked(f"Blocked by: {effect.describe()}")
        
        # All effects passed
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            return operation(*args, **kwargs)


async def demo():
    """Show the effect system replacing external gateway."""
    
    runtime = EffectRuntime()
    runtime.register(LogHandler())
    runtime.register(AuthHandler())
    runtime.register(RateLimitHandler())
    runtime.register(IhsanHandler())
    
    print("=== ALGEBRAIC EFFECTS DEMO ===\n")
    
    async def process_request(data: str) -> Dict:
        return {"status": "success", "data": data}
    
    # Valid request
    effects = [
        LogEffect(name="log", message="Incoming request", level="info"),
        AuthEffect(name="auth", token="bizra_secret_123", required_roles=["admin"]),
        RateLimitEffect(name="rate", key="api", limit=10, window_seconds=60),
        IhsanEffect(name="ihsan", vector={
            "correctness": 0.9,
            "safety": 0.85,
            "beneficence": 0.8,
        }),
    ]
    
    print("Request 1: Valid admin request")
    try:
        result = await runtime.run_with_effects(process_request, effects, "hello")
        print(f"  Result: {result}")
    except EffectBlocked as e:
        print(f"  Blocked: {e}")
    
    # Invalid token
    print("\nRequest 2: Invalid token")
    bad_effects = [
        AuthEffect(name="auth", token="invalid"),
    ]
    try:
        result = await runtime.run_with_effects(process_request, bad_effects, "hello")
        print(f"  Result: {result}")
    except EffectBlocked as e:
        print(f"  Blocked: {e}")
    
    # Ihsān violation
    print("\nRequest 3: Ihsān violation")
    unsafe_effects = [
        IhsanEffect(name="ihsan", vector={"safety": 0.5}),
    ]
    try:
        result = await runtime.run_with_effects(process_request, unsafe_effects, "hello")
        print(f"  Result: {result}")
    except EffectBlocked as e:
        print(f"  Blocked: {e}")


if __name__ == "__main__":
    asyncio.run(demo())
