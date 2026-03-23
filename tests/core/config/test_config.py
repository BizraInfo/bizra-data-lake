"""
Config System Tests — Phase 68.03
═════════════════════════════════

TDD anchors for 3-scope YAML config: loading, merging, SSoT validation,
federation signature check, reload + watchers.

Standing on Giants:
- Beck (2002): TDD by Example
- 12-Factor App (2011): Config in the environment
"""

from __future__ import annotations


import pytest

from core.config.loader import ConfigLoader, ConfigViolation
from core.config.schema import BizraConfig
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

# ═══════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════


class TestConfigLoading:
    """Loading from various sources."""

    def test_load_defaults_when_no_files(self, tmp_path) -> None:
        """Missing YAML files produce valid defaults."""
        loader = ConfigLoader(
            node_path=tmp_path / "missing.yaml",
            operator_path=tmp_path / "also_missing.yaml",
        )
        config = loader.load()
        assert isinstance(config, BizraConfig)
        assert config.policy.ihsan_floor == UNIFIED_IHSAN_THRESHOLD

    def test_load_local_only(self, tmp_path) -> None:
        """Local YAML is loaded and parsed."""
        pytest.importorskip("yaml")
        import yaml

        node_file = tmp_path / "bizra.node.yaml"
        node_file.write_text(
            yaml.dump(
                {
                    "orchestrator": {"max_workers": 16},
                }
            )
        )
        loader = ConfigLoader(
            node_path=node_file,
            operator_path=tmp_path / "nope.yaml",
        )
        config = loader.load()
        assert config.orchestrator.max_workers == 16

    def test_load_with_operator_merge(self, tmp_path) -> None:
        """Operator + local configs merge correctly."""
        pytest.importorskip("yaml")
        import yaml

        op_file = tmp_path / "operator.yaml"
        op_file.write_text(
            yaml.dump(
                {
                    "inference": {"primary": "lmstudio"},
                    "orchestrator": {"max_workers": 12},
                }
            )
        )
        node_file = tmp_path / "bizra.node.yaml"
        node_file.write_text(
            yaml.dump(
                {
                    "orchestrator": {"max_workers": 8},
                }
            )
        )
        loader = ConfigLoader(
            node_path=node_file,
            operator_path=op_file,
        )
        config = loader.load()
        assert config.orchestrator.max_workers == 8  # local wins
        assert config.inference.primary == "lmstudio"  # inherited from operator

    def test_local_overrides_operator(self, tmp_path) -> None:
        """Local scope takes precedence over operator scope."""
        pytest.importorskip("yaml")
        import yaml

        op_file = tmp_path / "operator.yaml"
        op_file.write_text(yaml.dump({"inference": {"primary": "ollama"}}))
        node_file = tmp_path / "bizra.node.yaml"
        node_file.write_text(yaml.dump({"inference": {"primary": "cloud"}}))
        loader = ConfigLoader(node_path=node_file, operator_path=op_file)
        config = loader.load()
        assert config.inference.primary == "cloud"


# ═══════════════════════════════════════════════════════════
# Constitutional SSoT validation
# ═══════════════════════════════════════════════════════════


class TestConstitutionalValidation:
    """Config cannot weaken SSoT thresholds."""

    def test_ihsan_below_ssot_rejected(self) -> None:
        loader = ConfigLoader()
        with pytest.raises(ConfigViolation, match="ihsan_floor"):
            loader.load_from_dict(
                {
                    "policy": {"ihsan_floor": 0.50},
                }
            )

    def test_gini_above_ssot_rejected(self) -> None:
        loader = ConfigLoader()
        with pytest.raises(ConfigViolation, match="gini_target"):
            loader.load_from_dict(
                {
                    "policy": {"gini_target": 0.90},
                }
            )

    def test_stricter_than_ssot_accepted(self) -> None:
        loader = ConfigLoader()
        config = loader.load_from_dict(
            {
                "policy": {"ihsan_floor": 0.99, "gini_target": 0.20},
            }
        )
        assert config.policy.ihsan_floor == 0.99
        assert config.policy.gini_target == 0.20


# ═══════════════════════════════════════════════════════════
# Federation signature
# ═══════════════════════════════════════════════════════════


class TestFederationSignature:
    """Federation configs must be signed."""

    def test_unsigned_federation_rejected(self, tmp_path) -> None:
        """Federation config without signature is ignored."""
        pytest.importorskip("yaml")
        import yaml

        fed_file = tmp_path / "bizra.fed.yaml"
        fed_file.write_text(
            yaml.dump(
                {
                    "policy": {"ihsan_floor": 0.99},
                }
            )
        )
        loader = ConfigLoader(
            node_path=tmp_path / "nope.yaml",
            federation_path=fed_file,
        )
        config = loader.load()
        # Federation was rejected, so default ihsan_floor
        assert config.policy.ihsan_floor == UNIFIED_IHSAN_THRESHOLD

    def test_signed_federation_accepted(self, tmp_path) -> None:
        """Federation config with signature fields is accepted."""
        pytest.importorskip("yaml")
        import yaml

        fed_file = tmp_path / "bizra.fed.yaml"
        fed_file.write_text(
            yaml.dump(
                {
                    "_signature": "ed25519:fakesig",
                    "_signed_by": "node:ed25519:fakekey",
                    "orchestrator": {"max_workers": 4},
                }
            )
        )
        loader = ConfigLoader(
            node_path=tmp_path / "nope.yaml",
            federation_path=fed_file,
        )
        config = loader.load()
        assert config.orchestrator.max_workers == 4


# ═══════════════════════════════════════════════════════════
# Reload and watchers
# ═══════════════════════════════════════════════════════════


class TestConfigReload:
    """Cache invalidation and watcher notifications."""

    def test_reload_invalidates_cache(self, tmp_path) -> None:
        loader = ConfigLoader(
            node_path=tmp_path / "nope.yaml",
            operator_path=tmp_path / "nope2.yaml",
        )
        c1 = loader.load()
        c2 = loader.load()
        assert c1 is c2  # cached
        c3 = loader.reload()
        assert c3 is not c1  # fresh

    def test_watchers_notified_on_reload(self, tmp_path) -> None:
        loader = ConfigLoader(
            node_path=tmp_path / "nope.yaml",
            operator_path=tmp_path / "nope2.yaml",
        )
        received = []
        loader.watch(lambda cfg: received.append(cfg))
        loader.reload()
        assert len(received) == 1
        assert isinstance(received[0], BizraConfig)

    def test_load_from_dict(self) -> None:
        """Programmatic config loading."""
        loader = ConfigLoader()
        config = loader.load_from_dict(
            {
                "node": {"id": "test-node"},
                "orchestrator": {"routing_model": "keyword"},
            }
        )
        assert config.node.id == "test-node"
        assert config.orchestrator.routing_model == "keyword"
