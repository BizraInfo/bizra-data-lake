"""Tests for BIZRASDPOTrainer — the SDPO training loop.

Covers:
- TrainingConfig defaults and constant imports
- TrainingState construction, serialization, deserialization
- TrainingBatch construction, iteration, length
- TrainingResult construction and serialization
- CheckpointManager save, load_latest, list_checkpoints
- BIZRASDPOTrainer: constructor, training loop, batch processing,
  gradient accumulation, learning rate warmup, evaluation, stats
- Checkpoint resume across training runs
- Edge cases: empty batches, zero-length data, multi-epoch training

Blueprint Reference: Elite Implementation Blueprint v1.0 — P0 Learning Loop
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sdpo import (
    SDPO_ADVANTAGE_THRESHOLD,
    SDPO_LEARNING_RATE,
    SDPO_MAX_ITERATIONS,
)
from core.sdpo.training.bizra_sdpo_trainer import (
    BIZRASDPOTrainer,
    CheckpointManager,
    TrainingBatch,
    TrainingConfig,
    TrainingResult,
    TrainingState,
)

# ═══════════════════════════════════════════════════════════════════════════
# TrainingConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingConfig:

    def test_defaults_from_constants(self):
        """Config defaults must import from constants — not hardcoded."""
        config = TrainingConfig()
        assert config.learning_rate == SDPO_LEARNING_RATE
        assert config.max_iterations_per_epoch == SDPO_MAX_ITERATIONS
        assert config.advantage_threshold == SDPO_ADVANTAGE_THRESHOLD
        assert config.ihsan_threshold == UNIFIED_IHSAN_THRESHOLD
        assert config.snr_threshold == UNIFIED_SNR_THRESHOLD

    def test_override_defaults(self):
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=16,
            max_epochs=5,
        )
        assert config.learning_rate == 0.001
        assert config.batch_size == 16
        assert config.max_epochs == 5

    def test_checkpoint_dir_is_string(self):
        config = TrainingConfig()
        assert isinstance(config.checkpoint_dir, str)


# ═══════════════════════════════════════════════════════════════════════════
# TrainingState
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingState:

    def test_default_construction(self):
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_loss == float("inf")
        assert state.best_ihsan_score == 0.0
        assert state.total_samples_processed == 0

    def test_to_dict(self):
        state = TrainingState(epoch=3, global_step=150, best_loss=0.42)
        d = state.to_dict()
        assert d["epoch"] == 3
        assert d["global_step"] == 150
        assert d["best_loss"] == 0.42

    def test_from_dict_roundtrip(self):
        original = TrainingState(
            epoch=5,
            global_step=200,
            best_loss=0.35,
            best_ihsan_score=0.97,
            total_samples_processed=500,
            accumulated_advantage=12.5,
        )
        d = original.to_dict()
        restored = TrainingState.from_dict(d)
        assert restored.epoch == original.epoch
        assert restored.global_step == original.global_step
        assert restored.best_loss == original.best_loss
        assert restored.best_ihsan_score == original.best_ihsan_score

    def test_from_dict_with_missing_keys(self):
        """Partial dict should use defaults for missing keys."""
        state = TrainingState.from_dict({"epoch": 7})
        assert state.epoch == 7
        assert state.global_step == 0  # default
        assert state.best_loss == float("inf")  # default

    def test_history_lists_start_empty(self):
        state = TrainingState()
        assert state.learning_rate_history == []
        assert state.loss_history == []
        assert state.ihsan_history == []


# ═══════════════════════════════════════════════════════════════════════════
# TrainingBatch
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingBatch:

    def _make_batch(self, n=3):
        return TrainingBatch(
            questions=[f"Question {i}" for i in range(n)],
            failed_attempts=[f"Wrong answer {i}" for i in range(n)],
            feedbacks=[f"Improve {i}" for i in range(n)],
            corrected_attempts=[f"Right answer {i}" for i in range(n)],
            quality_scores=[0.90 + i * 0.02 for i in range(n)],
        )

    def test_length(self):
        batch = self._make_batch(5)
        assert len(batch) == 5

    def test_iteration_yields_tuples(self):
        batch = self._make_batch(2)
        items = list(batch)
        assert len(items) == 2
        # Each item is (question, failed, feedback, corrected, score)
        assert len(items[0]) == 5
        assert items[0][0] == "Question 0"
        assert items[0][4] == 0.90

    def test_empty_batch(self):
        batch = TrainingBatch(
            questions=[],
            failed_attempts=[],
            feedbacks=[],
            corrected_attempts=[],
            quality_scores=[],
        )
        assert len(batch) == 0
        assert list(batch) == []

    def test_single_item_batch(self):
        batch = TrainingBatch(
            questions=["Q"],
            failed_attempts=["F"],
            feedbacks=["FB"],
            corrected_attempts=["C"],
            quality_scores=[0.99],
        )
        assert len(batch) == 1
        q, f, fb, c, s = next(iter(batch))
        assert q == "Q"
        assert s == 0.99


# ═══════════════════════════════════════════════════════════════════════════
# TrainingResult
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingResult:

    def test_to_dict(self):
        state = TrainingState(epoch=2, global_step=50)
        result = TrainingResult(
            final_state=state,
            total_epochs_completed=2,
            total_steps=50,
            final_loss=0.25,
            final_ihsan_score=0.97,
            training_time_seconds=12.5,
            checkpoints_saved=3,
        )
        d = result.to_dict()
        assert d["total_epochs_completed"] == 2
        assert d["final_loss"] == 0.25
        assert d["final_ihsan_score"] == 0.97
        assert d["training_time_seconds"] == 12.5
        assert "final_state" in d
        assert d["final_state"]["global_step"] == 50


# ═══════════════════════════════════════════════════════════════════════════
# CheckpointManager
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpointManager:

    def test_save_creates_file(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "ckpt"))
        state = TrainingState(epoch=1, global_step=10)
        path = mgr.save(state, step=10)
        assert Path(path).exists()

    def test_save_content_is_valid_json(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "ckpt"))
        state = TrainingState(epoch=2, global_step=20, best_loss=0.5)
        path = mgr.save(state, step=20, additional_data={"note": "test"})
        with open(path) as f:
            data = json.load(f)
        assert data["step"] == 20
        assert data["state"]["epoch"] == 2
        assert data["additional_data"]["note"] == "test"
        assert "timestamp" in data

    def test_load_latest_returns_none_when_empty(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "ckpt"))
        assert mgr.load_latest() is None

    def test_load_latest_returns_highest_step(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "ckpt"))
        mgr.save(TrainingState(epoch=0), step=5)
        mgr.save(TrainingState(epoch=1), step=15)
        mgr.save(TrainingState(epoch=2, best_loss=0.3), step=25)

        result = mgr.load_latest()
        assert result is not None
        state, step, additional = result
        assert step == 25
        assert state.epoch == 2
        assert state.best_loss == 0.3

    def test_list_checkpoints(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "ckpt"))
        mgr.save(TrainingState(), step=10)
        mgr.save(TrainingState(), step=20)
        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 2

    def test_checkpoint_dir_created_automatically(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c"
        CheckpointManager(str(deep_path))
        assert deep_path.exists()


# ═══════════════════════════════════════════════════════════════════════════
# BIZRASDPOTrainer
# ═══════════════════════════════════════════════════════════════════════════


class TestBIZRASDPOTrainer:

    @pytest.fixture
    def trainer(self, tmp_path):
        return BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=1,
                checkpoint_dir=str(tmp_path / "ckpt"),
                checkpoint_interval=1000,  # Don't checkpoint mid-test
            )
        )

    def _make_batch(self, n=2):
        return TrainingBatch(
            questions=[f"What is {i}+{i}?" for i in range(n)],
            failed_attempts=[f"{i}+{i} equals {i*3}" for i in range(n)],
            feedbacks=[f"Incorrect. {i}+{i}={i*2}." for i in range(n)],
            corrected_attempts=[f"{i}+{i} equals {i*2}" for i in range(n)],
            quality_scores=[0.95] * n,
        )

    def test_default_construction(self, tmp_path):
        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(checkpoint_dir=str(tmp_path / "ckpt"))
        )
        assert trainer.state.epoch == 0
        assert trainer.state.global_step == 0
        assert trainer.model_update_callback is None

    def test_construction_with_callback(self, tmp_path):
        cb = MagicMock()
        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(checkpoint_dir=str(tmp_path / "ckpt")),
            model_update_callback=cb,
        )
        assert trainer.model_update_callback is cb

    @pytest.mark.asyncio
    async def test_train_single_batch(self, trainer):
        """Train on one batch, verify result structure."""
        batch = self._make_batch(2)
        result = await trainer.train([batch])

        assert isinstance(result, TrainingResult)
        assert result.total_epochs_completed == 1
        assert result.total_steps >= 1
        assert result.training_time_seconds >= 0
        assert result.checkpoints_saved >= 1  # final checkpoint

    @pytest.mark.asyncio
    async def test_train_updates_state(self, trainer):
        """Training updates internal state correctly."""
        batch = self._make_batch(3)
        await trainer.train([batch])

        assert trainer.state.global_step >= 1
        assert trainer.state.total_samples_processed >= 3
        assert len(trainer.state.loss_history) >= 1
        assert len(trainer.state.ihsan_history) >= 1

    @pytest.mark.asyncio
    async def test_train_multi_epoch(self, tmp_path):
        """Multi-epoch training processes all epochs."""
        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=3,
                checkpoint_dir=str(tmp_path / "ckpt"),
                checkpoint_interval=1000,
            )
        )
        batch = self._make_batch(2)
        result = await trainer.train([batch])

        assert result.total_epochs_completed == 3
        # 3 epochs * 1 batch = 3 steps
        assert result.total_steps == 3

    @pytest.mark.asyncio
    async def test_train_multiple_batches(self, tmp_path):
        """Training with multiple batches per epoch."""
        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=1,
                checkpoint_dir=str(tmp_path / "ckpt"),
                checkpoint_interval=1000,
            )
        )
        batches = [self._make_batch(2), self._make_batch(3)]
        result = await trainer.train(batches)

        assert result.total_steps == 2  # 2 batches in 1 epoch
        assert trainer.state.total_samples_processed == 5  # 2 + 3

    @pytest.mark.asyncio
    async def test_best_loss_tracked(self, trainer):
        """Best loss should be tracked across batches."""
        batch = self._make_batch(2)
        result = await trainer.train([batch])

        assert result.final_loss < float("inf")
        assert trainer.state.best_loss == result.final_loss

    @pytest.mark.asyncio
    async def test_best_ihsan_tracked(self, tmp_path):
        """Best ihsan tracks when score exceeds threshold."""
        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=1,
                ihsan_threshold=0.90,  # lower threshold to trigger tracking
                checkpoint_dir=str(tmp_path / "ckpt"),
                checkpoint_interval=1000,
            )
        )
        batch = TrainingBatch(
            questions=["Q1"],
            failed_attempts=["Bad"],
            feedbacks=["Fix"],
            corrected_attempts=["Good"],
            quality_scores=[0.96],
        )
        await trainer.train([batch])
        # Quality score 0.96 > threshold 0.90 → should update best
        assert trainer.state.best_ihsan_score == 0.96

    @pytest.mark.asyncio
    async def test_learning_rate_warmup(self, tmp_path):
        """Learning rate starts low during warmup period."""
        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(
                warmup_steps=100,
                checkpoint_dir=str(tmp_path / "ckpt"),
                checkpoint_interval=1000,
            )
        )
        # At step 0, warmup LR should be 0
        lr = trainer._get_learning_rate()
        assert lr == 0.0

        # Simulate advancing steps
        trainer.state.global_step = 50
        lr = trainer._get_learning_rate()
        assert lr == trainer.config.learning_rate * 0.5  # 50/100

        # After warmup complete
        trainer.state.global_step = 100
        lr = trainer._get_learning_rate()
        assert lr == trainer.config.learning_rate

        # Well past warmup
        trainer.state.global_step = 500
        lr = trainer._get_learning_rate()
        assert lr == trainer.config.learning_rate

    @pytest.mark.asyncio
    async def test_gradient_accumulation_with_callback(self, tmp_path):
        """Callback fires after accumulation_steps gradients."""
        cb = MagicMock()
        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=1,
                gradient_accumulation_steps=2,
                warmup_steps=0,  # No warmup to ensure callback fires
                checkpoint_dir=str(tmp_path / "ckpt"),
                checkpoint_interval=1000,
            ),
            model_update_callback=cb,
        )
        # Need enough samples to trigger accumulation
        batch = self._make_batch(4)
        await trainer.train([batch])

        # 4 samples / 2 accumulation = 2 callback calls
        assert cb.call_count == 2

    @pytest.mark.asyncio
    async def test_no_callback_without_model_update(self, trainer):
        """Without callback, gradient accumulation is silent."""
        batch = self._make_batch(4)
        # Should not raise even without callback
        await trainer.train([batch])

    @pytest.mark.asyncio
    async def test_checkpoint_resume(self, tmp_path):
        """Training resumes from checkpoint correctly."""
        ckpt_dir = str(tmp_path / "ckpt")

        # Train first time
        trainer1 = BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=2,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=1,  # Checkpoint every step
            )
        )
        batch = self._make_batch(2)
        await trainer1.train([batch])

        # Create second trainer — should resume
        trainer2 = BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=4,  # More epochs
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=1000,
            )
        )
        result2 = await trainer2.train([batch], resume_from_checkpoint=True)

        # Second trainer should have started from checkpoint state
        assert result2.total_epochs_completed == 4

    @pytest.mark.asyncio
    async def test_no_resume_when_disabled(self, tmp_path):
        """Training from scratch when resume_from_checkpoint=False."""
        ckpt_dir = str(tmp_path / "ckpt")

        # Save a checkpoint
        mgr = CheckpointManager(ckpt_dir)
        mgr.save(TrainingState(epoch=5, global_step=100), step=100)

        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=1,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=1000,
            )
        )
        await trainer.train([self._make_batch(1)], resume_from_checkpoint=False)

        # Should start from epoch 0, not 5
        assert trainer.state.epoch == 0

    @pytest.mark.asyncio
    async def test_evaluate(self, trainer):
        """Evaluate returns correct metric keys."""
        batch = self._make_batch(3)
        result = await trainer.evaluate([batch])

        assert "eval_loss" in result
        assert "eval_ihsan" in result
        assert "eval_samples" in result
        assert result["eval_samples"] == 3

    @pytest.mark.asyncio
    async def test_evaluate_empty_batches(self, trainer):
        """Evaluate on empty data returns zeros."""
        batch = TrainingBatch(
            questions=[],
            failed_attempts=[],
            feedbacks=[],
            corrected_attempts=[],
            quality_scores=[],
        )
        result = await trainer.evaluate([batch])
        assert result["eval_loss"] == 0
        assert result["eval_ihsan"] == 0
        assert result["eval_samples"] == 0

    def test_get_training_stats(self, trainer):
        """Stats include state + runtime info."""
        stats = trainer.get_training_stats()
        assert "epoch" in stats
        assert "global_step" in stats
        assert "current_learning_rate" in stats
        assert "accumulated_gradients" in stats
        assert "checkpoints" in stats
        assert isinstance(stats["checkpoints"], list)

    @pytest.mark.asyncio
    async def test_accumulated_advantage_grows(self, trainer):
        """Accumulated advantage tracks total advantage across training."""
        batch = self._make_batch(3)
        await trainer.train([batch])
        # Some advantage should have accumulated (positive or negative)
        assert trainer.state.accumulated_advantage != 0.0

    @pytest.mark.asyncio
    async def test_checkpoint_saved_at_end(self, tmp_path):
        """Final checkpoint is always saved."""
        ckpt_dir = str(tmp_path / "ckpt")
        trainer = BIZRASDPOTrainer(
            config=TrainingConfig(
                max_epochs=1,
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=999999,  # Never checkpoint mid-training
            )
        )
        result = await trainer.train([self._make_batch(1)])

        # Final checkpoint always saved
        assert result.checkpoints_saved >= 1
        checkpoints = CheckpointManager(ckpt_dir).list_checkpoints()
        assert len(checkpoints) >= 1
