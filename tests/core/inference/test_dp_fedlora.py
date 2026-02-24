"""
Tests for DP-FedLoRA — Differential Privacy for Federated LoRA Updates
"""

import math
import random

import pytest

from core.inference.dp_fedlora import (
    DEFAULT_CLIP_THRESHOLD,
    DEFAULT_NOISE_MULTIPLIER,
    DPConfig,
    DPFedLoRAUpdate,
    DPResult,
    GradientClipper,
    LoRAWeightDelta,
    NoiseInjector,
)


class TestDPConfig:
    """Test DP configuration."""

    def test_epsilon_computation(self):
        config = DPConfig(clip_threshold=1.0, noise_multiplier=1.1, delta=1e-5)
        eps = config.epsilon
        # epsilon = C * sqrt(2 * ln(1.25 / delta)) / sigma
        expected = 1.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.1
        assert abs(eps - expected) < 1e-6

    def test_epsilon_infinite_when_no_noise(self):
        config = DPConfig(noise_multiplier=0.0)
        assert config.epsilon == float("inf")

    def test_to_dict(self):
        config = DPConfig()
        d = config.to_dict()
        assert "clip_threshold" in d
        assert "noise_multiplier" in d
        assert "epsilon" in d


class TestGradientClipper:
    """Test gradient clipping."""

    def test_no_clipping_within_threshold(self):
        clipper = GradientClipper(clip_threshold=10.0)
        delta = LoRAWeightDelta(values=[[1.0, 0.0], [0.0, 1.0]])

        clipped, norm, was_clipped = clipper.clip(delta)
        assert not was_clipped
        assert abs(norm - math.sqrt(2.0)) < 1e-6
        assert clipped.values == delta.values

    def test_clipping_above_threshold(self):
        clipper = GradientClipper(clip_threshold=1.0)
        # Create a matrix with L2 norm = 5.0
        delta = LoRAWeightDelta(values=[[3.0, 4.0]])

        clipped, norm, was_clipped = clipper.clip(delta)
        assert was_clipped
        assert abs(norm - 5.0) < 1e-6

        # Clipped norm should equal threshold
        clipped_norm = clipper.compute_l2_norm(clipped.values)
        assert abs(clipped_norm - 1.0) < 1e-6

    def test_l2_norm_identity_matrix(self):
        clipper = GradientClipper()
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        norm = clipper.compute_l2_norm(matrix)
        assert abs(norm - math.sqrt(2.0)) < 1e-6

    def test_l2_norm_clipped_correctly(self):
        """Verify: after clipping, ||B_k||_2 <= C."""
        clipper = GradientClipper(clip_threshold=1.0)
        delta = LoRAWeightDelta(values=[[10.0, 20.0], [30.0, 40.0]])

        clipped, _, _ = clipper.clip(delta)
        clipped_norm = clipper.compute_l2_norm(clipped.values)
        assert clipped_norm <= 1.0 + 1e-9

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            GradientClipper(clip_threshold=-1.0)


class TestNoiseInjector:
    """Test noise injection."""

    def test_noise_std_computation(self):
        injector = NoiseInjector(noise_multiplier=1.1, clip_threshold=2.0)
        assert abs(injector.noise_std - 2.2) < 1e-9

    def test_noise_changes_values(self):
        injector = NoiseInjector(noise_multiplier=1.0, clip_threshold=1.0)
        delta = LoRAWeightDelta(values=[[1.0, 2.0], [3.0, 4.0]])

        rng = random.Random(42)
        noisy = injector.inject(delta, rng=rng)

        # Values should be different after noise injection
        for i in range(2):
            for j in range(2):
                assert noisy.values[i][j] != delta.values[i][j]

    def test_noise_statistical_properties(self):
        """Verify: noise has mean ~0 and std ~sigma."""
        injector = NoiseInjector(noise_multiplier=1.0, clip_threshold=1.0)
        rng = random.Random(42)

        # Collect many noise samples
        n_samples = 10_000
        noise_values = []
        for _ in range(n_samples):
            delta = LoRAWeightDelta(values=[[0.0]])
            noisy = injector.inject(delta, rng=rng)
            noise_values.append(noisy.values[0][0])

        mean = sum(noise_values) / len(noise_values)
        variance = sum((x - mean) ** 2 for x in noise_values) / len(noise_values)
        std = math.sqrt(variance)

        # Mean should be approximately 0
        assert abs(mean) < 0.05, f"Mean {mean} too far from 0"
        # Std should be approximately sigma * C = 1.0
        assert abs(std - 1.0) < 0.1, f"Std {std} too far from 1.0"


class TestDPFedLoRAUpdate:
    """Test the end-to-end DP-FedLoRA pipeline."""

    def test_privatize_success(self):
        dp = DPFedLoRAUpdate()
        delta = LoRAWeightDelta(
            values=[[0.1, 0.2], [0.3, 0.4]],
            layer_name="layer.0.lora_A",
        )
        rng = random.Random(42)

        result = dp.privatize(delta, rng=rng)
        assert result.success
        assert result.privatized_delta is not None
        assert result.epsilon > 0
        assert result.noise_std > 0

    def test_privatize_clips_large_updates(self):
        dp = DPFedLoRAUpdate(config=DPConfig(clip_threshold=1.0))
        delta = LoRAWeightDelta(values=[[100.0, 200.0]])

        result = dp.privatize(delta)
        assert result.success
        assert result.was_clipped
        assert result.clipped_norm <= 1.0 + 0.01  # Within threshold + noise

    def test_update_count_increments(self):
        dp = DPFedLoRAUpdate()
        delta = LoRAWeightDelta(values=[[0.1]])

        assert dp.update_count == 0
        dp.privatize(delta)
        assert dp.update_count == 1
        dp.privatize(delta)
        assert dp.update_count == 2

    def test_compose_privacy(self):
        dp = DPFedLoRAUpdate(config=DPConfig(clip_threshold=1.0, noise_multiplier=1.1))
        composed = dp.compose_privacy(num_updates=100)

        assert composed["num_updates"] == 100
        assert composed["composed_epsilon"] > composed["per_step_epsilon"]

    def test_l2_norm_after_privatization(self):
        """SAPE spec: verify L2 norm <= clip threshold after clipping."""
        config = DPConfig(clip_threshold=0.5)
        dp = DPFedLoRAUpdate(config=config)
        delta = LoRAWeightDelta(values=[[5.0, 5.0, 5.0]])

        result = dp.privatize(delta)
        assert result.was_clipped
        assert result.clipped_norm <= 0.5 + 1e-6

    def test_noise_injection_alters_values(self):
        """SAPE spec: verify noise injection alters values."""
        dp = DPFedLoRAUpdate(config=DPConfig(noise_multiplier=2.0))
        delta = LoRAWeightDelta(values=[[1.0, 2.0, 3.0]])

        result = dp.privatize(delta)
        assert result.success

        # At least one value should differ
        original = delta.values[0]
        noisy = result.privatized_delta.values[0]
        differs = any(abs(o - n) > 1e-10 for o, n in zip(original, noisy))
        assert differs, "Noise injection should alter at least one value"
