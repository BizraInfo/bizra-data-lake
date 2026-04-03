"""
Test suite for BIZRA Autonomous Forest Growth Protocol - Phase 1

Tests:
1. Seed Creation - Node0 can create child seeds with unique IDs
2. Lifecycle Progression - Seeds advance through 6 states
3. Reproduction Gating - Only Ihsān >= 0.95 + PoI >= 10 can reproduce
4. Covenant Inheritance - Children inherit parent covenant
5. Receipt Generation - All lifecycle events emit receipts
6. Token Tracking - Off-chain BZR balances maintained
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from bizra_kernel.seed_manager import (
    SeedManager,
    Seed,
    SeedState,
    REPRODUCTION_REQUIREMENTS,
)
from bizra_kernel.autonomous_growth import (
    AutonomousGrowthEngine,
    GrowthTrigger,
    GrowthCycle,
)
from bizra_kernel.recursive_node import RecursiveNode, SeedState as NodeSeedState


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def seed_manager(temp_storage):
    """Create SeedManager with temp storage."""
    return SeedManager(storage_dir=temp_storage)


@pytest.fixture
def growth_engine(seed_manager):
    """Create AutonomousGrowthEngine with seed manager."""
    return AutonomousGrowthEngine(seed_manager)


class TestSeedCreation:
    """Test 1: Node0 can create child seeds with unique IDs."""

    def test_create_node0_seed(self, seed_manager):
        """Node0 (root seed) can be created with no parent."""
        seed = seed_manager.create_seed(parent_id=None)

        assert seed is not None
        assert seed.seed_id.startswith("SEED-")
        assert seed.parent_id is None
        assert seed.state == SeedState.DORMANT
        assert seed.depth == 0

    def test_create_child_seed(self, seed_manager):
        """Child seeds can be created from parent."""
        parent = seed_manager.create_seed(parent_id=None)
        child = seed_manager.create_seed(parent_id=parent.seed_id)

        assert child is not None
        assert child.parent_id == parent.seed_id
        assert child.depth == 1
        assert child.seed_id in parent.children

    def test_unique_seed_ids(self, seed_manager):
        """All seed IDs are unique."""
        seeds = [seed_manager.create_seed(parent_id=None) for _ in range(10)]
        seed_ids = [s.seed_id for s in seeds]

        assert len(seed_ids) == len(set(seed_ids))  # All unique

    def test_seed_persistence(self, temp_storage):
        """Seeds persist across manager instances."""
        # Create seed with first manager
        mgr1 = SeedManager(storage_dir=temp_storage)
        seed = mgr1.create_seed(parent_id=None)
        seed_id = seed.seed_id

        # Load with second manager
        mgr2 = SeedManager(storage_dir=temp_storage)
        loaded = mgr2.get_seed(seed_id)

        assert loaded is not None
        assert loaded.seed_id == seed_id


class TestLifecycleProgression:
    """Test 2: Seeds advance through 6 states."""

    def test_germination(self, seed_manager):
        """DORMANT seeds can germinate."""
        seed = seed_manager.create_seed(parent_id=None)
        assert seed.state == SeedState.DORMANT

        result = seed_manager.germinate(seed.seed_id)

        assert result is True
        assert seed.state == SeedState.GERMINATING
        assert seed.germinated_at is not None

    def test_cannot_germinate_twice(self, seed_manager):
        """Already germinated seeds cannot germinate again."""
        seed = seed_manager.create_seed(parent_id=None)
        seed_manager.germinate(seed.seed_id)

        result = seed_manager.germinate(seed.seed_id)

        assert result is False

    def test_lifecycle_advancement(self, seed_manager):
        """Seeds advance through lifecycle with sufficient requirements."""
        seed = seed_manager.create_seed(parent_id=None)
        seed_manager.germinate(seed.seed_id)

        # Add PoI events to meet requirements
        for i in range(15):
            seed_manager.record_poi_event(seed.seed_id, {
                "type": f"test_event_{i}",
                "impact_score": 0.1,
            })

        seed_manager.accept_covenant(seed.seed_id)

        # Advance through states
        states_seen = [seed.state]
        while True:
            new_state = seed_manager.advance_lifecycle(seed.seed_id)
            if new_state is None:
                break
            states_seen.append(new_state)

        # Should progress through multiple states
        assert len(states_seen) > 1
        assert SeedState.SAPLING in states_seen or SeedState.MATURE in states_seen

    def test_all_six_states_exist(self):
        """All 6 lifecycle states are defined."""
        states = list(SeedState)
        assert len(states) == 6
        assert SeedState.DORMANT in states
        assert SeedState.GERMINATING in states
        assert SeedState.SAPLING in states
        assert SeedState.MATURE in states
        assert SeedState.FRUITING in states
        assert SeedState.ELDER in states


class TestReproductionGating:
    """Test 3: Only Ihsān >= 0.95 + PoI >= 10 can reproduce."""

    def test_cannot_reproduce_without_ihsan(self, seed_manager):
        """Seeds with low Ihsān cannot reproduce."""
        seed = seed_manager.create_seed(parent_id=None)
        seed.ihsan_score = 0.5  # Below threshold
        seed.poi_events = 15
        seed.covenant_accepted = True
        seed.state = SeedState.FRUITING

        assert seed_manager.can_reproduce(seed.seed_id) is False

    def test_cannot_reproduce_without_poi(self, seed_manager):
        """Seeds with low PoI cannot reproduce."""
        seed = seed_manager.create_seed(parent_id=None)
        seed.ihsan_score = 0.98
        seed.poi_events = 3  # Below threshold
        seed.covenant_accepted = True
        seed.state = SeedState.FRUITING

        assert seed_manager.can_reproduce(seed.seed_id) is False

    def test_cannot_reproduce_without_covenant(self, seed_manager):
        """Seeds without covenant cannot reproduce."""
        seed = seed_manager.create_seed(parent_id=None)
        seed.ihsan_score = 0.98
        seed.poi_events = 15
        seed.covenant_accepted = False  # Not accepted
        seed.state = SeedState.FRUITING

        assert seed_manager.can_reproduce(seed.seed_id) is False

    def test_cannot_reproduce_wrong_state(self, seed_manager):
        """Seeds not in FRUITING/ELDER cannot reproduce."""
        seed = seed_manager.create_seed(parent_id=None)
        seed.ihsan_score = 0.98
        seed.poi_events = 15
        seed.covenant_accepted = True
        seed.state = SeedState.MATURE  # Wrong state

        assert seed_manager.can_reproduce(seed.seed_id) is False

    def test_can_reproduce_with_all_requirements(self, seed_manager):
        """Seeds meeting all requirements can reproduce."""
        seed = seed_manager.create_seed(parent_id=None)
        seed_manager.germinate(seed.seed_id)

        # Set to 8 days ago to meet age requirement
        seed.germinated_at = datetime.utcnow() - timedelta(days=8)

        seed.ihsan_score = 0.98
        seed.poi_events = 15
        seed.covenant_accepted = True
        seed.state = SeedState.FRUITING

        assert seed_manager.can_reproduce(seed.seed_id) is True

    def test_max_children_limit(self, seed_manager):
        """Seeds cannot exceed max children (7)."""
        seed = seed_manager.create_seed(parent_id=None)
        seed.germinated_at = datetime.utcnow() - timedelta(days=8)
        seed.ihsan_score = 0.98
        seed.poi_events = 15
        seed.covenant_accepted = True
        seed.state = SeedState.FRUITING

        # Add 7 children (max)
        for i in range(7):
            child = seed_manager.create_seed(parent_id=seed.seed_id)

        assert seed_manager.can_reproduce(seed.seed_id) is False


class TestCovenantInheritance:
    """Test 4: Children inherit parent covenant."""

    def test_covenant_inherited_on_creation(self, seed_manager):
        """Child inherits covenant from parent at creation."""
        parent = seed_manager.create_seed(parent_id=None)
        seed_manager.accept_covenant(parent.seed_id)

        child = seed_manager.create_seed(parent_id=parent.seed_id)

        assert child.covenant_accepted is True

    def test_no_inheritance_if_parent_no_covenant(self, seed_manager):
        """Child does not inherit if parent has no covenant."""
        parent = seed_manager.create_seed(parent_id=None)
        # Parent has not accepted covenant

        child = seed_manager.create_seed(parent_id=parent.seed_id)

        assert child.covenant_accepted is False

    def test_recursive_node_covenant_inheritance(self):
        """RecursiveNode.inherit_covenant works."""
        parent = RecursiveNode()
        parent.covenant_accepted = True

        child = RecursiveNode()
        child.inherit_covenant(parent)

        assert child.covenant_accepted is True


class TestReceiptGeneration:
    """Test 5: All lifecycle events emit receipts."""

    def test_creation_emits_receipt(self, temp_storage):
        """Seed creation emits receipt."""
        mgr = SeedManager(storage_dir=temp_storage)
        mgr.create_seed(parent_id=None)

        receipts_file = Path("docs/evidence/receipts/seed_lifecycle_receipts.jsonl")
        assert receipts_file.exists()

        with open(receipts_file) as f:
            lines = f.readlines()
            assert len(lines) >= 1

            import json
            receipt = json.loads(lines[-1])
            assert "receipt_id" in receipt
            assert "timestamp" in receipt
            assert receipt["event_type"] == "seed_created"
            assert "integrity_hash" in receipt

    def test_germination_emits_receipt(self, temp_storage):
        """Germination emits receipt."""
        mgr = SeedManager(storage_dir=temp_storage)
        seed = mgr.create_seed(parent_id=None)
        mgr.germinate(seed.seed_id)

        receipts_file = Path("docs/evidence/receipts/seed_lifecycle_receipts.jsonl")

        import json
        with open(receipts_file) as f:
            receipts = [json.loads(line) for line in f]

        germination_receipts = [r for r in receipts if r["event_type"] == "seed_germinated"]
        assert len(germination_receipts) >= 1

    def test_poi_event_emits_receipt(self, temp_storage):
        """PoI events emit receipts."""
        mgr = SeedManager(storage_dir=temp_storage)
        seed = mgr.create_seed(parent_id=None)
        mgr.record_poi_event(seed.seed_id, {"type": "test", "impact_score": 0.1})

        receipts_file = Path("docs/evidence/receipts/seed_lifecycle_receipts.jsonl")

        import json
        with open(receipts_file) as f:
            receipts = [json.loads(line) for line in f]

        poi_receipts = [r for r in receipts if r["event_type"] == "poi_event"]
        assert len(poi_receipts) >= 1


class TestTokenTracking:
    """Test 6: Off-chain BZR balances maintained."""

    def test_initial_balance_zero(self, seed_manager):
        """New seeds start with zero balance."""
        seed = seed_manager.create_seed(parent_id=None)
        assert seed.token_balance == 0.0

    def test_add_tokens(self, seed_manager):
        """Tokens can be added to seed."""
        seed = seed_manager.create_seed(parent_id=None)
        result = seed_manager.update_token_balance(seed.seed_id, 100.0)

        assert result is True
        assert seed.token_balance == 100.0

    def test_subtract_tokens(self, seed_manager):
        """Tokens can be subtracted from seed."""
        seed = seed_manager.create_seed(parent_id=None)
        seed_manager.update_token_balance(seed.seed_id, 100.0)
        result = seed_manager.update_token_balance(seed.seed_id, -30.0)

        assert result is True
        assert seed.token_balance == 70.0

    def test_cannot_go_negative(self, seed_manager):
        """Balance cannot go below zero."""
        seed = seed_manager.create_seed(parent_id=None)
        seed_manager.update_token_balance(seed.seed_id, 50.0)
        result = seed_manager.update_token_balance(seed.seed_id, -100.0)

        assert result is False
        assert seed.token_balance == 50.0  # Unchanged

    def test_token_persistence(self, temp_storage):
        """Token balance persists."""
        mgr1 = SeedManager(storage_dir=temp_storage)
        seed = mgr1.create_seed(parent_id=None)
        mgr1.update_token_balance(seed.seed_id, 250.0)

        mgr2 = SeedManager(storage_dir=temp_storage)
        loaded = mgr2.get_seed(seed.seed_id)

        assert loaded.token_balance == 250.0


class TestAutonomousGrowthEngine:
    """Integration tests for the growth engine."""

    def test_initialize_node0(self, growth_engine):
        """Node0 can be initialized."""
        node0 = growth_engine.initialize_node0()

        assert node0 is not None
        assert node0.parent_id is None
        assert node0.ihsan_score >= 0.95
        assert node0.poi_events >= 10
        assert node0.covenant_accepted is True

    def test_seed_first_generation(self, growth_engine):
        """First 7 seeds can be created from Node0."""
        growth_engine.initialize_node0()
        first_gen = growth_engine.seed_first_generation()

        assert len(first_gen) == 7
        for seed in first_gen:
            assert seed.depth == 1
            assert seed.covenant_accepted is True  # Inherited

    def test_run_growth_cycle(self, growth_engine):
        """Growth cycle runs successfully."""
        growth_engine.initialize_node0()
        cycle = growth_engine.run_growth_cycle()

        assert cycle is not None
        assert cycle.cycle_id.startswith("CYCLE-")
        assert cycle.seeds_evaluated >= 1
        assert cycle.completed_at is not None

    def test_forest_metrics(self, growth_engine):
        """Forest metrics are calculated."""
        growth_engine.initialize_node0()
        growth_engine.seed_first_generation()

        metrics = growth_engine.get_forest_metrics()

        assert "total_seeds" in metrics
        assert metrics["total_seeds"] == 8  # Node0 + 7 first gen
        assert "network_effect" in metrics
        assert "average_ihsan_score" in metrics

    def test_network_effect_calculation(self, growth_engine):
        """Network effect score is calculated."""
        growth_engine.initialize_node0()
        growth_engine.seed_first_generation()

        effect = growth_engine.calculate_network_effect()

        assert 0.0 <= effect <= 1.0

    def test_evaluate_seed_triggers(self, growth_engine, seed_manager):
        """Seed triggers are correctly evaluated."""
        node0 = growth_engine.initialize_node0()
        triggers = growth_engine.evaluate_seed(node0.seed_id)

        assert GrowthTrigger.POI_THRESHOLD in triggers
        assert GrowthTrigger.IHSAN_THRESHOLD in triggers
        assert GrowthTrigger.COVENANT_ACCEPTED in triggers


class TestRecursiveNodeSeedIntegration:
    """Test RecursiveNode seed lifecycle integration."""

    def test_node_initial_seed_state(self):
        """RecursiveNode starts in DORMANT state."""
        node = RecursiveNode()
        assert node.seed_state == NodeSeedState.DORMANT
        assert node.ihsan_score == 0.0
        assert node.poi_events == 0

    def test_node_germinate(self):
        """RecursiveNode can germinate."""
        node = RecursiveNode()
        result = node.germinate()

        assert result is True
        assert node.seed_state == NodeSeedState.GERMINATING
        assert node.germinated_at is not None

    def test_node_record_poi(self):
        """RecursiveNode can record PoI events."""
        node = RecursiveNode()
        node.record_poi({"type": "test", "impact_score": 0.1})

        assert node.poi_events == 1
        assert node.ihsan_score > 0

    def test_node_can_fruit(self):
        """RecursiveNode.can_fruit works correctly."""
        node = RecursiveNode()
        node.seed_state = NodeSeedState.FRUITING
        node.ihsan_score = 0.98
        node.poi_events = 15
        node.covenant_accepted = True
        node.germinated_at = datetime.utcnow() - timedelta(days=8)

        assert node.can_fruit() is True

    def test_node_advance_seed_state(self):
        """RecursiveNode can advance seed state."""
        node = RecursiveNode()
        node.germinate()
        node.record_poi({"type": "test", "impact_score": 0.1})

        new_state = node.advance_seed_state()

        assert new_state == NodeSeedState.SAPLING


class TestLineageTracking:
    """Test lineage and ancestry features."""

    def test_get_lineage_single(self, seed_manager):
        """Single seed has self in lineage."""
        seed = seed_manager.create_seed(parent_id=None)
        lineage = seed_manager.get_lineage(seed.seed_id)

        assert len(lineage) == 1
        assert lineage[0].seed_id == seed.seed_id

    def test_get_lineage_multi_generation(self, seed_manager):
        """Multi-generation lineage is correct."""
        gen0 = seed_manager.create_seed(parent_id=None)
        gen1 = seed_manager.create_seed(parent_id=gen0.seed_id)
        gen2 = seed_manager.create_seed(parent_id=gen1.seed_id)

        lineage = seed_manager.get_lineage(gen2.seed_id)

        assert len(lineage) == 3
        assert lineage[0].seed_id == gen0.seed_id
        assert lineage[1].seed_id == gen1.seed_id
        assert lineage[2].seed_id == gen2.seed_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
