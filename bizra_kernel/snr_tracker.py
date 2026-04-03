"""
SNR Tracker — Signal-to-Noise Ratio Optimization
=================================================
From the Blueprint:
  SNR_score = (useful_tokens / total_tokens) × confidence × ethical_compliance × safety_compliance × tool_directness
  Target: SNR > 0.90

Tracks token efficiency across sessions and enables SAPE elevation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import json


@dataclass
class SNRMetrics:
    """Metrics for a single interaction."""
    total_tokens: int
    useful_tokens: int
    confidence_score: float
    ethical_compliance: float  # From Ihsān vector
    tool_directness: float  # How directly the tool addressed the query
    latency_ms: int
    agent_role: str
    safety_compliance: float = 1.0  # Damage control safety factor
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def snr_score(self) -> float:
        """Calculate the composite SNR score."""
        if self.total_tokens == 0:
            return 0.0
        
        token_efficiency = self.useful_tokens / self.total_tokens
        return (
            token_efficiency
            * self.confidence_score
            * self.ethical_compliance
            * self.safety_compliance
            * self.tool_directness
        )
    
    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "useful_tokens": self.useful_tokens,
            "token_efficiency": self.useful_tokens / max(1, self.total_tokens),
            "confidence_score": self.confidence_score,
            "ethical_compliance": self.ethical_compliance,
            "safety_compliance": self.safety_compliance,
            "tool_directness": self.tool_directness,
            "snr_score": self.snr_score,
            "latency_ms": self.latency_ms,
            "agent_role": self.agent_role,
            "timestamp": self.timestamp,
        }


class SNRTracker:
    """
    Tracks SNR across sessions and agents for optimization.
    
    Enables:
    - Per-agent SNR tracking
    - Pattern detection for SAPE elevation
    - Historical trend analysis
    """
    
    TARGET_SNR = 0.90
    
    def __init__(self, history_limit: int = 1000):
        self.metrics_history: List[SNRMetrics] = []
        self.history_limit = history_limit
        self.agent_metrics: Dict[str, List[SNRMetrics]] = defaultdict(list)
    
    def record(self, metrics: SNRMetrics) -> None:
        """Record a new SNR measurement."""
        self.metrics_history.append(metrics)
        self.agent_metrics[metrics.agent_role].append(metrics)
        
        # Trim history if needed
        if len(self.metrics_history) > self.history_limit:
            self.metrics_history = self.metrics_history[-self.history_limit:]
        
        if len(self.agent_metrics[metrics.agent_role]) > self.history_limit:
            self.agent_metrics[metrics.agent_role] = \
                self.agent_metrics[metrics.agent_role][-self.history_limit:]
    
    def get_current_snr(self) -> float:
        """Get the most recent SNR score."""
        if not self.metrics_history:
            return 0.0
        return self.metrics_history[-1].snr_score
    
    def get_average_snr(self, window: int = 100) -> float:
        """Get average SNR over recent window."""
        if not self.metrics_history:
            return 0.0
        
        recent = self.metrics_history[-window:]
        return sum(m.snr_score for m in recent) / len(recent)
    
    def get_agent_snr(self, agent_role: str, window: int = 50) -> float:
        """Get average SNR for a specific agent."""
        agent_history = self.agent_metrics.get(agent_role, [])
        if not agent_history:
            return 0.0
        
        recent = agent_history[-window:]
        return sum(m.snr_score for m in recent) / len(recent)
    
    def get_agent_rankings(self) -> List[Dict]:
        """Rank agents by their average SNR."""
        rankings = []
        for agent, history in self.agent_metrics.items():
            if history:
                avg_snr = sum(m.snr_score for m in history) / len(history)
                rankings.append({
                    "agent": agent,
                    "avg_snr": avg_snr,
                    "sample_count": len(history),
                    "meets_target": avg_snr >= self.TARGET_SNR,
                })
        
        return sorted(rankings, key=lambda x: x["avg_snr"], reverse=True)
    
    def get_token_waste(self) -> float:
        """Calculate overall token waste percentage."""
        if not self.metrics_history:
            return 0.0
        
        total = sum(m.total_tokens for m in self.metrics_history)
        useful = sum(m.useful_tokens for m in self.metrics_history)
        
        if total == 0:
            return 0.0
        
        return 1.0 - (useful / total)
    
    def detect_patterns(self) -> List[Dict]:
        """
        Detect patterns suitable for SAPE elevation.
        
        Returns patterns that occur >3 times with high SNR variance.
        """
        # Group by agent role
        patterns = []
        
        for agent, history in self.agent_metrics.items():
            if len(history) < 5:
                continue
            
            snr_scores = [m.snr_score for m in history[-20:]]
            avg = sum(snr_scores) / len(snr_scores)
            variance = sum((s - avg) ** 2 for s in snr_scores) / len(snr_scores)
            
            # High variance suggests optimization opportunity
            if variance > 0.01 and avg < self.TARGET_SNR:
                patterns.append({
                    "agent": agent,
                    "avg_snr": avg,
                    "variance": variance,
                    "recommendation": f"Agent {agent} shows inconsistent SNR; consider SAPE elevation",
                    "sample_count": len(history),
                })
        
        return patterns
    
    def get_statistics(self) -> dict:
        """Get comprehensive SNR statistics."""
        if not self.metrics_history:
            return {
                "total_measurements": 0,
                "average_snr": 0.0,
                "meets_target": False,
                "token_waste_percent": 0.0,
            }
        
        avg_snr = self.get_average_snr()
        
        return {
            "total_measurements": len(self.metrics_history),
            "average_snr": avg_snr,
            "current_snr": self.get_current_snr(),
            "target_snr": self.TARGET_SNR,
            "meets_target": avg_snr >= self.TARGET_SNR,
            "token_waste_percent": self.get_token_waste() * 100,
            "agent_count": len(self.agent_metrics),
            "agent_rankings": self.get_agent_rankings(),
            "elevation_candidates": self.detect_patterns(),
        }


def estimate_useful_tokens(response: str) -> int:
    """
    Estimate the number of useful tokens in a response.
    
    Filters out:
    - Filler phrases
    - Repetition
    - Excessive politeness
    """
    import re
    
    # Remove filler phrases
    fillers = [
        r"\bum\b", r"\buh\b", r"\bwell\b", r"\bso\b",
        r"as an ai", r"as a language model", r"i think that",
        r"it's important to note", r"i would say",
        r"basically", r"actually", r"literally",
    ]
    
    cleaned = response
    for filler in fillers:
        cleaned = re.sub(filler, "", cleaned, flags=re.IGNORECASE)
    
    # Count remaining words as useful tokens (rough estimate)
    words = cleaned.split()
    return len([w for w in words if len(w) > 1])
