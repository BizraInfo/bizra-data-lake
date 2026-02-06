"""
BIZRA-SDPO Training Loop — Self-Distillation Training for BIZRA Agents
===============================================================================

Implements the full SDPO training loop for BIZRA's cognitive architecture:
- Rich feedback integration from quality gates
- Token-level advantage optimization
- Checkpoint management and resumption

Standing on Giants: Shannon + SDPO Paper + Distributed Training
Genesis Strict Synthesis v2.2.2
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import json

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sdpo import (
    SDPO_LEARNING_RATE,
    SDPO_MAX_ITERATIONS,
    SDPO_ADVANTAGE_THRESHOLD,
)
from core.sdpo.optimization import (
    SDPOAdvantage,
    SDPOAdvantageCalculator,
    SDPOFeedback,
    BIZRAFeedbackGenerator,
)


@dataclass
class TrainingConfig:
    """Configuration for BIZRA-SDPO training."""
    learning_rate: float = SDPO_LEARNING_RATE
    batch_size: int = 8
    max_epochs: int = 10
    max_iterations_per_epoch: int = SDPO_MAX_ITERATIONS
    advantage_threshold: float = SDPO_ADVANTAGE_THRESHOLD
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD
    checkpoint_interval: int = 100  # Steps between checkpoints
    checkpoint_dir: str = "checkpoints/sdpo"
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0


@dataclass
class TrainingState:
    """State tracking for training."""
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    best_ihsan_score: float = 0.0
    total_samples_processed: int = 0
    accumulated_advantage: float = 0.0
    learning_rate_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    ihsan_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "best_ihsan_score": self.best_ihsan_score,
            "total_samples_processed": self.total_samples_processed,
            "accumulated_advantage": self.accumulated_advantage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        return cls(
            epoch=data.get("epoch", 0),
            global_step=data.get("global_step", 0),
            best_loss=data.get("best_loss", float("inf")),
            best_ihsan_score=data.get("best_ihsan_score", 0.0),
            total_samples_processed=data.get("total_samples_processed", 0),
            accumulated_advantage=data.get("accumulated_advantage", 0.0),
        )


@dataclass
class TrainingBatch:
    """A batch of training data for SDPO."""
    questions: List[str]
    failed_attempts: List[str]
    feedbacks: List[str]
    corrected_attempts: List[str]
    quality_scores: List[float]

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        for i in range(len(self)):
            yield (
                self.questions[i],
                self.failed_attempts[i],
                self.feedbacks[i],
                self.corrected_attempts[i],
                self.quality_scores[i],
            )


@dataclass
class TrainingResult:
    """Result from a training run."""
    final_state: TrainingState
    total_epochs_completed: int
    total_steps: int
    final_loss: float
    final_ihsan_score: float
    training_time_seconds: float
    checkpoints_saved: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_state": self.final_state.to_dict(),
            "total_epochs_completed": self.total_epochs_completed,
            "total_steps": self.total_steps,
            "final_loss": self.final_loss,
            "final_ihsan_score": self.final_ihsan_score,
            "training_time_seconds": self.training_time_seconds,
            "checkpoints_saved": self.checkpoints_saved,
        }


class CheckpointManager:
    """Manages training checkpoints for resumption."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state: TrainingState,
        step: int,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint and return path."""
        checkpoint = {
            "state": state.to_dict(),
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "additional_data": additional_data or {},
        }

        path = self.checkpoint_dir / f"checkpoint_{step}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        return str(path)

    def load_latest(self) -> Optional[Tuple[TrainingState, int, Dict[str, Any]]]:
        """Load the latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None

        # Find latest by step number
        latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))

        with open(latest, "r") as f:
            data = json.load(f)

        state = TrainingState.from_dict(data["state"])
        step = data["step"]
        additional = data.get("additional_data", {})

        return state, step, additional

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return sorted(
            [str(p) for p in self.checkpoint_dir.glob("checkpoint_*.json")]
        )


class BIZRASDPOTrainer:
    """
    BIZRA-SDPO Training Loop.

    Implements full SDPO training with:
    - Rich feedback from BIZRA quality gates
    - Token-level advantage optimization
    - Checkpoint management
    - Ihsān-constrained learning

    Usage:
        trainer = BIZRASDPOTrainer(config=TrainingConfig())

        # Prepare data
        batch = TrainingBatch(
            questions=["What is sovereignty?"],
            failed_attempts=["Sovereignty is control..."],
            feedbacks=["Be more specific about..."],
            corrected_attempts=["Sovereignty is supreme authority..."],
            quality_scores=[0.95],
        )

        # Train
        result = await trainer.train([batch])
        print(f"Final Ihsān: {result.final_ihsan_score}")
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model_update_callback: Optional[Callable[[SDPOAdvantage, float], None]] = None,
    ):
        self.config = config or TrainingConfig()
        self.model_update_callback = model_update_callback
        self.advantage_calculator = SDPOAdvantageCalculator()
        self.feedback_generator = BIZRAFeedbackGenerator()
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

        self.state = TrainingState()
        self._accumulated_gradients: List[SDPOAdvantage] = []

    async def train(
        self,
        batches: List[TrainingBatch],
        resume_from_checkpoint: bool = True,
    ) -> TrainingResult:
        """
        Run SDPO training on provided batches.

        Args:
            batches: List of training batches
            resume_from_checkpoint: Whether to resume from latest checkpoint

        Returns:
            TrainingResult with final metrics
        """
        start_time = datetime.now(timezone.utc)
        checkpoints_saved = 0

        # Resume from checkpoint if available
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                self.state, start_step, _ = checkpoint
                print(f"Resumed from checkpoint at step {start_step}")

        # Training loop
        for epoch in range(self.state.epoch, self.config.max_epochs):
            self.state.epoch = epoch
            epoch_loss = 0.0
            epoch_samples = 0

            for batch_idx, batch in enumerate(batches):
                batch_loss, batch_ihsan = await self._train_batch(batch)

                epoch_loss += batch_loss * len(batch)
                epoch_samples += len(batch)

                # Update state
                self.state.global_step += 1
                self.state.total_samples_processed += len(batch)
                self.state.loss_history.append(batch_loss)
                self.state.ihsan_history.append(batch_ihsan)

                # Checkpoint
                if self.state.global_step % self.config.checkpoint_interval == 0:
                    self.checkpoint_manager.save(self.state, self.state.global_step)
                    checkpoints_saved += 1

                # Early stopping check
                if batch_ihsan >= self.config.ihsan_threshold:
                    if batch_ihsan > self.state.best_ihsan_score:
                        self.state.best_ihsan_score = batch_ihsan

            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
            if avg_epoch_loss < self.state.best_loss:
                self.state.best_loss = avg_epoch_loss

        # Final checkpoint
        self.checkpoint_manager.save(self.state, self.state.global_step)
        checkpoints_saved += 1

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        return TrainingResult(
            final_state=self.state,
            total_epochs_completed=self.config.max_epochs,
            total_steps=self.state.global_step,
            final_loss=self.state.best_loss,
            final_ihsan_score=self.state.best_ihsan_score,
            training_time_seconds=elapsed,
            checkpoints_saved=checkpoints_saved,
        )

    async def _train_batch(self, batch: TrainingBatch) -> Tuple[float, float]:
        """Train on a single batch."""
        batch_advantages: List[SDPOAdvantage] = []
        batch_losses: List[float] = []

        for question, failed, feedback, corrected, quality in batch:
            # Calculate SDPO advantage
            advantage = await self.advantage_calculator.calculate_advantages(
                question=question,
                failed_attempt=failed,
                feedback=feedback,
                corrected_attempt=corrected,
            )

            batch_advantages.append(advantage)

            # Calculate loss (negative advantage for minimization)
            if advantage.is_beneficial:
                loss = -advantage.overall_advantage
            else:
                loss = abs(advantage.overall_advantage) + 1.0  # Penalty for non-beneficial

            batch_losses.append(loss)

            # Accumulate gradients
            self._accumulated_gradients.append(advantage)

            # Apply gradient update if accumulation complete
            if len(self._accumulated_gradients) >= self.config.gradient_accumulation_steps:
                await self._apply_accumulated_gradients()
                self._accumulated_gradients = []

            # Track advantage
            self.state.accumulated_advantage += advantage.overall_advantage

        # Calculate batch metrics
        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        avg_ihsan = sum(batch.quality_scores) / len(batch.quality_scores)

        return avg_loss, avg_ihsan

    async def _apply_accumulated_gradients(self):
        """Apply accumulated gradients to model."""
        if not self._accumulated_gradients or not self.model_update_callback:
            return

        # Average advantages
        avg_overall = sum(a.overall_advantage for a in self._accumulated_gradients) / len(self._accumulated_gradients)

        # Create combined advantage
        combined = SDPOAdvantage(
            token_advantages=[],
            overall_advantage=avg_overall,
            advantage_variance=0.0,
            max_advantage=max(a.max_advantage for a in self._accumulated_gradients),
            min_advantage=min(a.min_advantage for a in self._accumulated_gradients),
            positive_ratio=sum(a.positive_ratio for a in self._accumulated_gradients) / len(self._accumulated_gradients),
        )

        # Get current learning rate (with warmup)
        lr = self._get_learning_rate()
        self.state.learning_rate_history.append(lr)

        # Apply update via callback
        self.model_update_callback(combined, lr)

    def _get_learning_rate(self) -> float:
        """Get current learning rate with warmup."""
        if self.state.global_step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (self.state.global_step / self.config.warmup_steps)
        return self.config.learning_rate

    async def evaluate(self, batches: List[TrainingBatch]) -> Dict[str, float]:
        """Evaluate on validation batches."""
        total_loss = 0.0
        total_ihsan = 0.0
        total_samples = 0

        for batch in batches:
            for question, failed, feedback, corrected, quality in batch:
                advantage = await self.advantage_calculator.calculate_advantages(
                    question=question,
                    failed_attempt=failed,
                    feedback=feedback,
                    corrected_attempt=corrected,
                )

                loss = -advantage.overall_advantage if advantage.is_beneficial else 1.0
                total_loss += loss
                total_ihsan += quality
                total_samples += 1

        return {
            "eval_loss": total_loss / total_samples if total_samples > 0 else 0,
            "eval_ihsan": total_ihsan / total_samples if total_samples > 0 else 0,
            "eval_samples": total_samples,
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            **self.state.to_dict(),
            "current_learning_rate": self._get_learning_rate(),
            "accumulated_gradients": len(self._accumulated_gradients),
            "checkpoints": self.checkpoint_manager.list_checkpoints(),
        }
