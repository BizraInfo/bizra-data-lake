"""
SDPO A/B Testing Framework — Validation and Statistical Analysis
===============================================================================

Provides rigorous A/B testing for SDPO improvements:
- Statistical significance testing
- Quality gate validation
- Multi-arm experiment support

Standing on Giants: Fisher (Statistics) + SDPO Paper
Genesis Strict Synthesis v2.2.2
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import math
import random

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sdpo import SDPO_ADVANTAGE_THRESHOLD


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    min_samples_per_arm: int = 100
    confidence_level: float = 0.95
    power: float = 0.80
    min_effect_size: float = 0.05  # Minimum detectable effect
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD
    advantage_threshold: float = SDPO_ADVANTAGE_THRESHOLD
    max_duration_hours: float = 168.0  # 1 week default


@dataclass
class ExperimentArm:
    """A single arm in an A/B experiment."""
    name: str
    description: str
    is_control: bool = False
    samples: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_sample(self, value: float):
        self.samples.append(value)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def variance(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        mean = self.mean
        return sum((x - mean) ** 2 for x in self.samples) / (len(self.samples) - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "is_control": self.is_control,
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "metadata": self.metadata,
        }


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results."""
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    confidence_interval: Tuple[float, float]
    is_significant: bool
    power_achieved: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval),
            "is_significant": self.is_significant,
            "power_achieved": self.power_achieved,
        }


@dataclass
class ABTestResult:
    """Result from an A/B test."""
    experiment_id: str
    status: ExperimentStatus
    control_arm: ExperimentArm
    treatment_arm: ExperimentArm
    analysis: Optional[StatisticalAnalysis]
    winner: Optional[str]
    ihsan_compliant: bool
    recommendation: str
    start_time: datetime
    end_time: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "control": self.control_arm.to_dict(),
            "treatment": self.treatment_arm.to_dict(),
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "winner": self.winner,
            "ihsan_compliant": self.ihsan_compliant,
            "recommendation": self.recommendation,
            "duration_hours": (
                (self.end_time - self.start_time).total_seconds() / 3600
                if self.end_time else None
            ),
        }


class QualityValidator:
    """
    Validates SDPO improvements against quality gates.

    Ensures that any "winning" variant meets Ihsān and SNR thresholds.
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or ABTestConfig()

    def validate(
        self,
        arm: ExperimentArm,
        metric_type: str = "ihsan",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate an experiment arm against quality gates.

        Args:
            arm: The experiment arm to validate
            metric_type: Type of metric ("ihsan", "snr", "advantage")

        Returns:
            (passes, details)
        """
        if arm.n == 0:
            return False, {"reason": "No samples"}

        mean = arm.mean

        if metric_type == "ihsan":
            threshold = self.config.ihsan_threshold
            passes = mean >= threshold
        elif metric_type == "snr":
            threshold = self.config.snr_threshold
            passes = mean >= threshold
        elif metric_type == "advantage":
            threshold = self.config.advantage_threshold
            passes = mean >= threshold
        else:
            return False, {"reason": f"Unknown metric type: {metric_type}"}

        return passes, {
            "mean": mean,
            "threshold": threshold,
            "margin": mean - threshold,
            "n_samples": arm.n,
        }


class SDPOABTestFramework:
    """
    A/B Testing Framework for SDPO Improvements.

    Provides statistical rigor for comparing SDPO variants:
    - Welch's t-test for unequal variances
    - Effect size (Cohen's d) calculation
    - Power analysis
    - Ihsān-constrained winner selection

    Usage:
        framework = SDPOABTestFramework()

        # Create experiment
        exp_id = framework.create_experiment(
            name="SDPO v1 vs v2",
            control_desc="Baseline SDPO",
            treatment_desc="Enhanced SDPO with PRM",
        )

        # Add samples
        framework.add_sample(exp_id, "control", 0.92)
        framework.add_sample(exp_id, "treatment", 0.95)

        # Analyze
        result = framework.analyze(exp_id)
        print(f"Winner: {result.winner}")
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or ABTestConfig()
        self.validator = QualityValidator(self.config)
        self._experiments: Dict[str, Dict[str, Any]] = {}

    def create_experiment(
        self,
        name: str,
        control_desc: str,
        treatment_desc: str,
        experiment_id: Optional[str] = None,
    ) -> str:
        """Create a new A/B experiment."""
        exp_id = experiment_id or f"exp_{len(self._experiments) + 1}_{int(datetime.now().timestamp())}"

        self._experiments[exp_id] = {
            "name": name,
            "control": ExperimentArm(
                name="control",
                description=control_desc,
                is_control=True,
            ),
            "treatment": ExperimentArm(
                name="treatment",
                description=treatment_desc,
                is_control=False,
            ),
            "status": ExperimentStatus.RUNNING,
            "start_time": datetime.now(timezone.utc),
            "end_time": None,
        }

        return exp_id

    def add_sample(
        self,
        experiment_id: str,
        arm: str,
        value: float,
    ) -> bool:
        """Add a sample to an experiment arm."""
        if experiment_id not in self._experiments:
            return False

        exp = self._experiments[experiment_id]
        if exp["status"] != ExperimentStatus.RUNNING:
            return False

        if arm == "control":
            exp["control"].add_sample(value)
        elif arm == "treatment":
            exp["treatment"].add_sample(value)
        else:
            return False

        return True

    def add_samples_batch(
        self,
        experiment_id: str,
        control_samples: List[float],
        treatment_samples: List[float],
    ):
        """Add multiple samples at once."""
        for sample in control_samples:
            self.add_sample(experiment_id, "control", sample)
        for sample in treatment_samples:
            self.add_sample(experiment_id, "treatment", sample)

    def analyze(
        self,
        experiment_id: str,
        force_complete: bool = False,
    ) -> ABTestResult:
        """
        Analyze an experiment and determine winner.

        Args:
            experiment_id: ID of the experiment
            force_complete: Complete even if min samples not reached

        Returns:
            ABTestResult with statistical analysis
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp = self._experiments[experiment_id]
        control = exp["control"]
        treatment = exp["treatment"]

        # Check sample size
        has_min_samples = (
            control.n >= self.config.min_samples_per_arm and
            treatment.n >= self.config.min_samples_per_arm
        )

        if not has_min_samples and not force_complete:
            return ABTestResult(
                experiment_id=experiment_id,
                status=ExperimentStatus.RUNNING,
                control_arm=control,
                treatment_arm=treatment,
                analysis=None,
                winner=None,
                ihsan_compliant=False,
                recommendation=f"Need more samples (control: {control.n}, treatment: {treatment.n})",
                start_time=exp["start_time"],
                end_time=None,
            )

        # Perform statistical analysis
        analysis = self._perform_analysis(control, treatment)

        # Validate quality gates
        control_valid, _ = self.validator.validate(control)
        treatment_valid, _ = self.validator.validate(treatment)

        # Determine winner
        winner = None
        recommendation = ""

        if analysis.is_significant:
            # Significant difference found
            if treatment.mean > control.mean:
                if treatment_valid:
                    winner = "treatment"
                    recommendation = f"Treatment wins with effect size {analysis.effect_size:.3f}"
                else:
                    recommendation = "Treatment better but fails Ihsān gate"
            else:
                if control_valid:
                    winner = "control"
                    recommendation = "Control remains superior"
                else:
                    recommendation = "Both variants fail Ihsān gate"
        else:
            # No significant difference
            if treatment_valid and control_valid:
                recommendation = "No significant difference; either variant acceptable"
            elif treatment_valid:
                winner = "treatment"
                recommendation = "Treatment meets Ihsān; control does not"
            elif control_valid:
                winner = "control"
                recommendation = "Control meets Ihsān; treatment does not"
            else:
                recommendation = "Neither variant meets Ihsān threshold"

        # Update experiment status
        exp["status"] = ExperimentStatus.COMPLETED
        exp["end_time"] = datetime.now(timezone.utc)

        ihsan_compliant = (winner == "treatment" and treatment_valid) or (winner == "control" and control_valid)

        return ABTestResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.COMPLETED,
            control_arm=control,
            treatment_arm=treatment,
            analysis=analysis,
            winner=winner,
            ihsan_compliant=ihsan_compliant,
            recommendation=recommendation,
            start_time=exp["start_time"],
            end_time=exp["end_time"],
        )

    def _perform_analysis(
        self,
        control: ExperimentArm,
        treatment: ExperimentArm,
    ) -> StatisticalAnalysis:
        """Perform Welch's t-test and compute statistics."""
        n1, n2 = control.n, treatment.n
        m1, m2 = control.mean, treatment.mean
        s1, s2 = control.std, treatment.std

        # Handle edge cases
        if n1 < 2 or n2 < 2 or (s1 == 0 and s2 == 0):
            return StatisticalAnalysis(
                t_statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                power_achieved=0.0,
            )

        # Welch's t-test
        se1 = s1 ** 2 / n1 if n1 > 0 else 0
        se2 = s2 ** 2 / n2 if n2 > 0 else 0
        se_diff = math.sqrt(se1 + se2) if (se1 + se2) > 0 else 1e-10

        t_stat = (m2 - m1) / se_diff

        # Degrees of freedom (Welch-Satterthwaite)
        if se1 + se2 > 0:
            df = ((se1 + se2) ** 2) / (
                (se1 ** 2 / (n1 - 1) if n1 > 1 else 0) +
                (se2 ** 2 / (n2 - 1) if n2 > 1 else 0)
            ) if ((se1 ** 2 / (n1 - 1) if n1 > 1 else 0) + (se2 ** 2 / (n2 - 1) if n2 > 1 else 0)) > 0 else 1
        else:
            df = n1 + n2 - 2

        # Approximate p-value using normal approximation for large df
        p_value = self._approx_p_value(t_stat, df)

        # Cohen's d effect size
        pooled_std = math.sqrt(
            ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
        ) if (n1 + n2 - 2) > 0 else 1.0
        effect_size = (m2 - m1) / pooled_std if pooled_std > 0 else 0.0

        # Confidence interval for difference
        alpha = 1 - self.config.confidence_level
        z_crit = 1.96  # Approximate for 95% CI
        ci_lower = (m2 - m1) - z_crit * se_diff
        ci_upper = (m2 - m1) + z_crit * se_diff

        # Achieved power (simplified)
        power_achieved = min(1.0, abs(effect_size) / self.config.min_effect_size * self.config.power)

        return StatisticalAnalysis(
            t_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < (1 - self.config.confidence_level),
            power_achieved=power_achieved,
        )

    def _approx_p_value(self, t: float, df: float) -> float:
        """Approximate two-tailed p-value from t-distribution."""
        # Using normal approximation for large df
        if df > 30:
            # Normal approximation
            z = abs(t)
            # Approximate using error function
            p = 2 * (1 - self._normal_cdf(z))
            return p
        else:
            # Crude approximation for small df
            # More accurate would use scipy.stats.t.sf
            z = abs(t) * math.sqrt(df / (df + t ** 2))
            return 2 * (1 - self._normal_cdf(z))

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        # Using error function approximation
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment."""
        if experiment_id not in self._experiments:
            return {"error": "Experiment not found"}

        exp = self._experiments[experiment_id]
        return {
            "id": experiment_id,
            "name": exp.get("name", ""),
            "status": exp["status"].value,
            "control_samples": exp["control"].n,
            "treatment_samples": exp["treatment"].n,
            "control_mean": exp["control"].mean,
            "treatment_mean": exp["treatment"].mean,
            "start_time": exp["start_time"].isoformat(),
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        return [
            self.get_experiment_status(exp_id)
            for exp_id in self._experiments
        ]

    def simulate_experiment(
        self,
        control_mean: float,
        treatment_mean: float,
        std: float,
        n_samples: int,
    ) -> ABTestResult:
        """
        Simulate an A/B experiment for testing.

        Useful for power analysis and framework validation.
        """
        exp_id = self.create_experiment(
            name="Simulated Experiment",
            control_desc=f"Control (μ={control_mean})",
            treatment_desc=f"Treatment (μ={treatment_mean})",
        )

        # Generate samples
        control_samples = [random.gauss(control_mean, std) for _ in range(n_samples)]
        treatment_samples = [random.gauss(treatment_mean, std) for _ in range(n_samples)]

        self.add_samples_batch(exp_id, control_samples, treatment_samples)

        return self.analyze(exp_id)
