"""
SNR Maximizer — Shannon-Inspired Signal Optimization Engine
═══════════════════════════════════════════════════════════════════════════════

"The fundamental problem of communication is that of reproducing at one point
either exactly or approximately a message selected at another point."
    — Claude Shannon, 1948

This module implements a comprehensive Signal-to-Noise Ratio maximization
engine that optimizes information quality across all BIZRA operations.

Core Principle: Every piece of information has signal (useful content) and
noise (uncertainty, distortion). Maximizing SNR improves decision quality.

SNR Calculation (Shannon-inspired):
    SNR = Signal Power / Noise Power
    SNR_dB = 10 * log10(Signal / Noise)

For BIZRA, we normalize to [0, 1]:
    SNR_normalized = Signal / (Signal + Noise)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        SNR MAXIMIZER                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
    │  │ Signal          │  │ Noise           │  │ Channel                 │ │
    │  │ Extractor       │  │ Estimator       │  │ Capacity                │ │
    │  └────────┬────────┘  └────────┬────────┘  └───────────┬─────────────┘ │
    │           │                    │                       │               │
    │           └────────────────────┼───────────────────────┘               │
    │                                ▼                                       │
    │              ┌──────────────────────────────────┐                      │
    │              │         SNR Calculator           │                      │
    │              │    (Adaptive Thresholding)       │                      │
    │              └──────────────────────────────────┘                      │
    │                                │                                       │
    │                                ▼                                       │
    │              ┌──────────────────────────────────┐                      │
    │              │         Filter Chain             │                      │
    │              │  (Noise reduction, amplification)│                      │
    │              └──────────────────────────────────┘                      │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Standing on Giants: Shannon (1948), Nyquist (1928), Wiener (1949)

Created: 2026-02-04 | BIZRA Sovereign Runtime v1.0
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean, stdev
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from core.integration.constants import SNR_THRESHOLD_T1_HIGH, UNIFIED_SNR_THRESHOLD

logger = logging.getLogger(__name__)


# Shannon-inspired constants (from single source of truth)
SNR_FLOOR: float = UNIFIED_SNR_THRESHOLD  # Minimum acceptable SNR
SNR_EXCELLENT: float = SNR_THRESHOLD_T1_HIGH  # Ihsan-grade SNR
SNR_CHANNEL_CAPACITY: float = 1.0  # Theoretical maximum
NOISE_FLOOR_DB: float = -60.0  # Minimum detectable noise
ADAPTIVE_WINDOW: int = 100  # Samples for adaptive thresholding


class SignalQuality(str, Enum):
    """Qualitative signal quality levels."""

    EXCELLENT = "excellent"  # SNR >= 0.95
    GOOD = "good"  # SNR >= 0.90
    ACCEPTABLE = "acceptable"  # SNR >= 0.85
    POOR = "poor"  # SNR >= 0.70
    NOISE = "noise"  # SNR < 0.70


class NoiseType(str, Enum):
    """Types of noise in information channels."""

    THERMAL = "thermal"  # Random background noise
    QUANTIZATION = "quantization"  # Discretization error
    INTERFERENCE = "interference"  # External signal corruption
    ATTENUATION = "attenuation"  # Signal loss over distance
    DISTORTION = "distortion"  # Non-linear channel effects


@dataclass
class Signal:
    """A signal with SNR metrics."""

    id: str
    content: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Signal metrics
    signal_power: float = 1.0  # Raw signal strength
    noise_power: float = 0.1  # Estimated noise
    snr: float = 0.0  # Calculated SNR [0, 1]
    snr_db: float = 0.0  # SNR in decibels
    quality: SignalQuality = SignalQuality.POOR

    # Provenance
    source: str = ""
    channel: str = ""
    hops: int = 0  # Signal degradation tracking

    def __post_init__(self):
        """Calculate derived metrics."""
        self._calculate_snr()

    def _calculate_snr(self) -> None:
        """Calculate SNR from signal and noise power."""
        if self.noise_power <= 0:
            self.snr = SNR_CHANNEL_CAPACITY
            self.snr_db = float("inf")
        else:
            # Normalized SNR [0, 1]
            self.snr = self.signal_power / (self.signal_power + self.noise_power)
            # SNR in dB
            self.snr_db = 10 * math.log10(self.signal_power / self.noise_power)

        # Assign quality level
        if self.snr >= SNR_EXCELLENT:
            self.quality = SignalQuality.EXCELLENT
        elif self.snr >= 0.90:
            self.quality = SignalQuality.GOOD
        elif self.snr >= SNR_FLOOR:
            self.quality = SignalQuality.ACCEPTABLE
        elif self.snr >= 0.70:
            self.quality = SignalQuality.POOR
        else:
            self.quality = SignalQuality.NOISE


@dataclass
class ChannelMetrics:
    """Metrics for an information channel."""

    channel_id: str
    capacity: float = 1.0  # Shannon capacity
    bandwidth: float = 1.0  # Available bandwidth
    latency_ms: float = 0.0  # Channel latency
    error_rate: float = 0.0  # Bit error rate
    utilization: float = 0.0  # Current utilization
    snr_history: List[float] = field(default_factory=list)

    def average_snr(self) -> float:
        """Get average SNR over history."""
        return mean(self.snr_history) if self.snr_history else 0.0

    def shannon_capacity(self) -> float:
        """
        Calculate Shannon capacity: C = B * log2(1 + SNR)

        This is the theoretical maximum rate of reliable communication.
        """
        avg_snr = self.average_snr()
        if avg_snr <= 0:
            return 0.0
        # Normalized to [0, 1] for our purposes
        return self.bandwidth * math.log2(1 + avg_snr)


class NoiseEstimator:
    """
    Estimates noise in signals using various techniques.

    Standing on Giants:
    - Wiener (1949): Optimal filtering theory
    - Kalman (1960): State estimation under noise
    """

    def __init__(self, window_size: int = ADAPTIVE_WINDOW):
        self.window_size = window_size
        self._noise_history: Deque[float] = deque(maxlen=window_size)
        self._baseline_noise: float = 0.1

    def estimate_noise(self, signal_values: List[float]) -> float:
        """
        Estimate noise power from signal variance.

        Uses high-frequency component estimation assuming
        signal is low-frequency and noise is high-frequency.
        """
        if len(signal_values) < 2:
            return self._baseline_noise

        # High-pass filter approximation: differences
        differences = [
            abs(signal_values[i] - signal_values[i - 1])
            for i in range(1, len(signal_values))
        ]

        # Noise estimate from variance of differences
        noise_estimate = mean(differences) if differences else self._baseline_noise

        # Update history
        self._noise_history.append(noise_estimate)

        return noise_estimate

    def adaptive_threshold(self) -> float:
        """Get adaptive noise threshold from history."""
        if not self._noise_history:
            return self._baseline_noise

        return (
            mean(self._noise_history) + stdev(self._noise_history)
            if len(self._noise_history) > 1
            else mean(self._noise_history)
        )


class SignalExtractor:
    """
    Extracts signal strength from various data types.

    Maps domain-specific metrics to normalized signal power.
    """

    def __init__(self):
        self._extractors: Dict[str, Callable[[Any], float]] = {
            "confidence": self._extract_confidence,
            "probability": self._extract_probability,
            "score": self._extract_score,
            "strength": self._extract_strength,
            "certainty": self._extract_certainty,
        }

    def extract(self, data: Any, metric_type: str = "confidence") -> float:
        """Extract signal power from data."""
        extractor = self._extractors.get(metric_type, self._extract_default)
        return extractor(data)

    def _extract_confidence(self, data: Any) -> float:
        """Extract from confidence score."""
        if isinstance(data, (int, float)):
            return max(0.0, min(1.0, float(data)))
        if hasattr(data, "confidence"):
            return max(0.0, min(1.0, float(data.confidence)))
        return 0.5

    def _extract_probability(self, data: Any) -> float:
        """Extract from probability."""
        if isinstance(data, (int, float)):
            return max(0.0, min(1.0, float(data)))
        if hasattr(data, "probability"):
            return max(0.0, min(1.0, float(data.probability)))
        return 0.5

    def _extract_score(self, data: Any) -> float:
        """Extract from score (may need normalization)."""
        if isinstance(data, (int, float)):
            # Assume [0, 100] scale
            return max(0.0, min(1.0, float(data) / 100))
        if hasattr(data, "score"):
            return max(0.0, min(1.0, float(data.score)))
        return 0.5

    def _extract_strength(self, data: Any) -> float:
        """Extract from strength indicator."""
        if isinstance(data, str):
            strength_map = {
                "strong": 0.9,
                "moderate": 0.7,
                "weak": 0.4,
            }
            return strength_map.get(data.lower(), 0.5)
        return self._extract_default(data)

    def _extract_certainty(self, data: Any) -> float:
        """Extract from certainty measure."""
        return self._extract_confidence(data)

    def _extract_default(self, data: Any) -> float:
        """Default extraction."""
        if isinstance(data, (int, float)):
            return max(0.0, min(1.0, float(data)))
        return 0.5


class SNRCalculator:
    """
    Calculates and tracks SNR across the system.

    Standing on Giants: Shannon (1948) information theory
    """

    def __init__(
        self,
        snr_floor: float = SNR_FLOOR,
        adaptive: bool = True,
    ):
        self.snr_floor = snr_floor
        self.adaptive = adaptive

        self._signal_extractor = SignalExtractor()
        self._noise_estimator = NoiseEstimator()

        # Tracking
        self._signal_history: Dict[str, Deque[Signal]] = {}
        self._channel_metrics: Dict[str, ChannelMetrics] = {}

    def calculate(
        self,
        data: Any,
        source: str = "",
        channel: str = "default",
        metric_type: str = "confidence",
    ) -> Signal:
        """
        Calculate SNR for incoming data.

        Args:
            data: Input data to analyze
            source: Origin of the signal
            channel: Communication channel
            metric_type: Type of metric to extract

        Returns:
            Signal with SNR metrics
        """
        # Extract signal power
        signal_power = self._signal_extractor.extract(data, metric_type)

        # Get noise history for this channel
        channel_history = self._get_channel_history(channel)
        noise_power = self._noise_estimator.estimate_noise(channel_history)

        # Create signal object
        signal = Signal(
            id=f"sig-{int(time.time() * 1000)}",
            content=data,
            signal_power=signal_power,
            noise_power=noise_power,
            source=source,
            channel=channel,
        )

        # Track signal
        self._track_signal(signal)

        return signal

    def _get_channel_history(self, channel: str) -> List[float]:
        """Get recent signal powers for a channel."""
        if channel not in self._signal_history:
            return []
        return [s.signal_power for s in self._signal_history[channel]]

    def _track_signal(self, signal: Signal) -> None:
        """Track signal in history."""
        if signal.channel not in self._signal_history:
            self._signal_history[signal.channel] = deque(maxlen=ADAPTIVE_WINDOW)

        self._signal_history[signal.channel].append(signal)

        # Update channel metrics
        if signal.channel not in self._channel_metrics:
            self._channel_metrics[signal.channel] = ChannelMetrics(
                channel_id=signal.channel
            )

        self._channel_metrics[signal.channel].snr_history.append(signal.snr)
        if len(self._channel_metrics[signal.channel].snr_history) > ADAPTIVE_WINDOW:
            self._channel_metrics[signal.channel].snr_history.pop(0)

    def passes_threshold(self, signal: Signal) -> bool:
        """Check if signal passes SNR threshold."""
        threshold = self.snr_floor

        # Adaptive threshold adjustment
        if self.adaptive and signal.channel in self._channel_metrics:
            avg_snr = self._channel_metrics[signal.channel].average_snr()
            # Adjust threshold based on channel quality
            threshold = max(self.snr_floor, avg_snr * 0.9)

        return signal.snr >= threshold

    def get_channel_capacity(self, channel: str) -> float:
        """Get Shannon capacity for a channel."""
        metrics = self._channel_metrics.get(channel)
        if not metrics:
            return 0.0
        return metrics.shannon_capacity()


class SNRFilter:
    """
    Filter chain for SNR-based signal processing.

    Applies noise reduction and signal amplification.
    """

    def __init__(self, snr_floor: float = SNR_FLOOR):
        self.snr_floor = snr_floor
        self._calculator = SNRCalculator(snr_floor=snr_floor)

    def filter(
        self,
        data: Any,
        source: str = "",
        channel: str = "default",
        metric_type: str = "confidence",
    ) -> Tuple[bool, Signal]:
        """
        Filter data based on SNR.

        Returns:
            Tuple of (passes_filter, signal)
        """
        signal = self._calculator.calculate(data, source, channel, metric_type)
        passes = self._calculator.passes_threshold(signal)

        if not passes:
            logger.debug(
                f"Signal filtered: SNR={signal.snr:.3f} < {self.snr_floor:.3f} "
                f"(source={source}, channel={channel})"
            )

        return passes, signal

    def batch_filter(
        self,
        items: List[Any],
        source: str = "",
        channel: str = "default",
        metric_type: str = "confidence",
    ) -> List[Tuple[Any, Signal]]:
        """Filter a batch, returning only those that pass."""
        results = []
        for item in items:
            passes, signal = self.filter(item, source, channel, metric_type)
            if passes:
                results.append((item, signal))
        return results


class SNRMaximizer:
    """
    The master SNR Maximization Engine.

    Coordinates signal extraction, noise estimation, and filtering
    to maximize information quality across all BIZRA operations.

    Standing on Giants:
    - Shannon (1948): Information theory, channel capacity
    - Nyquist (1928): Sampling theorem, bandwidth limits
    - Wiener (1949): Optimal filtering, noise reduction

    Usage:
        maximizer = SNRMaximizer()

        # Single signal
        signal = maximizer.process(data, source="market", metric_type="confidence")
        if maximizer.is_excellent(signal):
            # High-confidence action

        # Batch filtering
        clean_signals = maximizer.filter_batch(raw_signals, min_snr=0.85)

        # Channel analysis
        capacity = maximizer.get_channel_capacity("inference")
    """

    def __init__(
        self,
        snr_floor: float = SNR_FLOOR,
        snr_excellent: float = SNR_EXCELLENT,
        adaptive: bool = True,
    ):
        """
        Initialize the SNR Maximizer.

        Args:
            snr_floor: Minimum acceptable SNR (default 0.85)
            snr_excellent: Ihsan-grade SNR (default 0.95)
            adaptive: Enable adaptive thresholding
        """
        self.snr_floor = snr_floor
        self.snr_excellent = snr_excellent
        self.adaptive = adaptive

        # Components
        self._calculator = SNRCalculator(snr_floor=snr_floor, adaptive=adaptive)
        self._filter = SNRFilter(snr_floor=snr_floor)

        # Metrics
        self._total_processed: int = 0
        self._total_passed: int = 0
        self._total_excellent: int = 0

        logger.info(
            f"SNRMaximizer initialized: floor={snr_floor}, "
            f"excellent={snr_excellent}, adaptive={adaptive}"
        )

    def process(
        self,
        data: Any,
        source: str = "",
        channel: str = "default",
        metric_type: str = "confidence",
    ) -> Signal:
        """
        Process data and calculate SNR.

        Args:
            data: Input data
            source: Signal source identifier
            channel: Communication channel
            metric_type: Metric extraction type

        Returns:
            Signal with SNR metrics
        """
        signal = self._calculator.calculate(data, source, channel, metric_type)
        self._total_processed += 1

        if signal.snr >= self.snr_floor:
            self._total_passed += 1
        if signal.snr >= self.snr_excellent:
            self._total_excellent += 1

        return signal

    def filter(
        self,
        data: Any,
        source: str = "",
        channel: str = "default",
        metric_type: str = "confidence",
    ) -> Optional[Signal]:
        """
        Filter data, returning Signal only if it passes threshold.

        Returns:
            Signal if passes, None otherwise
        """
        signal = self.process(data, source, channel, metric_type)
        return signal if signal.snr >= self.snr_floor else None

    def filter_batch(
        self,
        items: List[Any],
        source: str = "",
        channel: str = "default",
        metric_type: str = "confidence",
        min_snr: Optional[float] = None,
    ) -> List[Signal]:
        """
        Filter a batch of items by SNR.

        Args:
            items: List of items to process
            source: Signal source
            channel: Communication channel
            metric_type: Metric extraction type
            min_snr: Override minimum SNR (default uses snr_floor)

        Returns:
            List of Signals that passed the filter
        """
        threshold = min_snr if min_snr is not None else self.snr_floor
        results = []

        for item in items:
            signal = self.process(item, source, channel, metric_type)
            if signal.snr >= threshold:
                results.append(signal)

        return results

    def is_excellent(self, signal: Signal) -> bool:
        """Check if signal is Ihsan-grade (excellent quality)."""
        return signal.snr >= self.snr_excellent

    def is_acceptable(self, signal: Signal) -> bool:
        """Check if signal is acceptable quality."""
        return signal.snr >= self.snr_floor

    def get_quality(self, signal: Signal) -> SignalQuality:
        """Get quality classification for a signal."""
        return signal.quality

    def get_channel_capacity(self, channel: str) -> float:
        """Get Shannon capacity for a channel."""
        return self._calculator.get_channel_capacity(channel)

    def get_channel_metrics(self, channel: str) -> Optional[ChannelMetrics]:
        """Get metrics for a channel."""
        return self._calculator._channel_metrics.get(channel)

    def get_statistics(self) -> Dict[str, Any]:
        """Get maximizer statistics."""
        pass_rate = (
            self._total_passed / self._total_processed
            if self._total_processed > 0
            else 0.0
        )
        excellence_rate = (
            self._total_excellent / self._total_processed
            if self._total_processed > 0
            else 0.0
        )

        return {
            "total_processed": self._total_processed,
            "total_passed": self._total_passed,
            "total_excellent": self._total_excellent,
            "pass_rate": pass_rate,
            "excellence_rate": excellence_rate,
            "snr_floor": self.snr_floor,
            "snr_excellent": self.snr_excellent,
            "channels": list(self._calculator._channel_metrics.keys()),
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._total_processed = 0
        self._total_passed = 0
        self._total_excellent = 0


# Global maximizer instance
_maximizer: Optional[SNRMaximizer] = None


def get_snr_maximizer() -> SNRMaximizer:
    """Get the global SNR maximizer."""
    global _maximizer
    if _maximizer is None:
        _maximizer = SNRMaximizer()
    return _maximizer


def snr_filter(min_snr: float = SNR_FLOOR):
    """
    Decorator to filter function outputs by SNR.

    Usage:
        @snr_filter(min_snr=0.90)
        def get_market_signals():
            return [signal1, signal2, ...]
    """

    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            if isinstance(result, list):
                maximizer = get_snr_maximizer()
                return maximizer.filter_batch(result, min_snr=min_snr)
            return result

        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, list):
                maximizer = get_snr_maximizer()
                return maximizer.filter_batch(result, min_snr=min_snr)
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
