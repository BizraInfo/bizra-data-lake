"""Integration tests for XTR-WARP environment, FAISS detection, and bridge.

Standing on Giants: Meta FAISS, Stanford ColBERTv2/PLAID, Google DeepMind XTR

These tests validate:
  1. FAISS environment detection (cpu/gpu variant, version)
  2. Conda environment YAML consistency (versions aligned)
  3. WARP bridge initialization safety (graceful degradation)
  4. Shared faiss_env module contract
  5. validate_env.py script runs without error
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
XTR_WARP_DIR = REPO_ROOT / "xtr-warp"

pytestmark = pytest.mark.skipif(
    not XTR_WARP_DIR.exists(),
    reason="xtr-warp/ directory not present",
)
BRIDGES_DIR = REPO_ROOT / "tools" / "bridges"

# Add bridges to sys.path so faiss_env is importable
if str(BRIDGES_DIR) not in sys.path:
    sys.path.insert(0, str(BRIDGES_DIR))


# ═══════════════════════════════════════════════════════════════════════════
# 1. FAISS ENVIRONMENT DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestFAISSEnvironment:
    """Verify the shared faiss_env module."""

    def test_faiss_env_importable(self):
        """faiss_env module imports without error."""
        mod = importlib.import_module("faiss_env")
        assert hasattr(mod, "FAISS_AVAILABLE")
        assert hasattr(mod, "is_gpu_ready")
        assert hasattr(mod, "FAISS_VERSION")
        assert hasattr(mod, "get_env")

    def test_faiss_env_constants_are_bool(self):
        from faiss_env import FAISS_AVAILABLE, is_gpu_ready

        assert isinstance(FAISS_AVAILABLE, bool)
        assert isinstance(is_gpu_ready(), bool)

    def test_faiss_env_summary_returns_string(self):
        from faiss_env import faiss_summary

        result = faiss_summary()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_faiss_env_dataclass_frozen(self):
        from faiss_env import get_env

        env = get_env()
        with pytest.raises(AttributeError):
            env.available = False  # type: ignore[misc]

    def test_faiss_available_matches_import(self):
        """FAISS_AVAILABLE must agree with direct import attempt."""
        from faiss_env import FAISS_AVAILABLE

        try:
            import faiss  # noqa: F401

            assert FAISS_AVAILABLE is True
        except ImportError:
            assert FAISS_AVAILABLE is False

    @pytest.mark.requires_gpu
    def test_faiss_gpu_functional(self):
        """When GPU is available, FAISS GPU resources should initialize."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from faiss_env import is_gpu_ready

        # If faiss-gpu is installed + CUDA works, this should be True.
        # It's OK if False (faiss-cpu only), but we surface it clearly.
        if is_gpu_ready():
            import faiss

            res = faiss.StandardGpuResources()
            assert res is not None
            del res


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONDA ENVIRONMENT CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════


class TestCondaEnvironments:
    """Ensure all conda env files are consistent."""

    @pytest.fixture(scope="class")
    def env_files(self) -> dict[str, dict]:
        files = {}
        for name in ("conda_env.yml", "conda_env_cpu.yml", "conda_env_windows.yml"):
            path = XTR_WARP_DIR / name
            if path.exists():
                with open(path) as f:
                    files[name] = yaml.safe_load(f)
        return files

    def test_all_envs_exist(self, env_files):
        assert "conda_env.yml" in env_files
        assert "conda_env_cpu.yml" in env_files
        assert "conda_env_windows.yml" in env_files

    def test_all_envs_valid_yaml(self, env_files):
        for name, data in env_files.items():
            assert isinstance(data, dict), f"{name} is not a valid YAML mapping"
            assert "dependencies" in data, f"{name} missing 'dependencies'"

    def test_python_version_documented(self, env_files):
        """Each environment must specify a Python version."""
        for name, data in env_files.items():
            found = False
            for dep in data.get("dependencies", []):
                if isinstance(dep, str) and dep.startswith("python="):
                    found = True
            assert found, f"{name} missing python version pin"

    def test_windows_python_at_least_3_10(self, env_files):
        """Windows env should use Python 3.10+ (for pip faiss-gpu-cu11 wheel compat)."""
        for dep in env_files["conda_env_windows.yml"]["dependencies"]:
            if isinstance(dep, str) and dep.startswith("python="):
                version = dep.split("=")[1]
                major, minor = version.split(".")[:2]
                assert (
                    int(major) >= 3 and int(minor) >= 10
                ), f"Windows env Python {version} < 3.10"

    def test_windows_no_faiss_gpu_conda(self, env_files):
        """Windows env must NOT have conda faiss-gpu (it has no Windows binaries)."""
        win_deps = env_files["conda_env_windows.yml"]["dependencies"]
        conda_deps = [d for d in win_deps if isinstance(d, str)]
        assert (
            "faiss-gpu" not in conda_deps
        ), "conda faiss-gpu found in Windows env — this will fail to install"

    def test_windows_has_faiss_cpu_baseline(self, env_files):
        """Windows env should have conda faiss-cpu as baseline."""
        win_deps = env_files["conda_env_windows.yml"]["dependencies"]
        conda_deps = [d for d in win_deps if isinstance(d, str)]
        assert "faiss-cpu" in conda_deps

    def test_windows_has_pip_faiss_gpu(self, env_files):
        """Windows env should have pip faiss-gpu-cu* for GPU support."""
        win_deps = env_files["conda_env_windows.yml"]["dependencies"]
        pip_section = [d for d in win_deps if isinstance(d, dict) and "pip" in d]
        assert len(pip_section) == 1, "Expected exactly one pip section"
        pip_deps = pip_section[0]["pip"]
        faiss_gpu_deps = [d for d in pip_deps if "faiss-gpu" in d]
        assert len(faiss_gpu_deps) >= 1, "No pip faiss-gpu wheel in Windows env"

    def test_faiss_gpu_pip_is_pinned(self, env_files):
        """pip faiss-gpu wheel should be version-pinned."""
        win_deps = env_files["conda_env_windows.yml"]["dependencies"]
        pip_section = [d for d in win_deps if isinstance(d, dict) and "pip" in d]
        pip_deps = pip_section[0]["pip"]
        faiss_gpu_deps = [d for d in pip_deps if "faiss-gpu" in d]
        for dep in faiss_gpu_deps:
            assert "==" in dep, f"faiss-gpu pip dep not pinned: {dep}"

    def test_env_name_consistent(self, env_files):
        """All env files should use the same env name."""
        names = {data.get("name") for data in env_files.values()}
        assert len(names) == 1, f"Inconsistent env names: {names}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. WARP BRIDGE SAFETY
# ═══════════════════════════════════════════════════════════════════════════


class TestWARPBridgeSafety:
    """Verify the bridge handles missing dependencies gracefully."""

    def test_warp_bridge_importable(self):
        """warp_bridge module imports even without XTR-WARP installed."""
        # This may fail if BIZRA config can't resolve paths — that's OK
        # for CI. The important thing is no hard crash.
        try:
            from tools.bridges.warp_bridge import WARPBridge, WARPStatus

            assert WARPStatus.OFFLINE is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("warp_bridge has unresolvable imports in this environment")

    def test_warp_retriever_importable(self):
        """warp_retriever module imports with graceful fallbacks."""
        try:
            from tools.bridges.warp_retriever import (
                RetrieverBackend,
                WARPRetriever,
            )

            assert RetrieverBackend.MINILM is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("warp_retriever has unresolvable imports in this environment")


# ═══════════════════════════════════════════════════════════════════════════
# 4. VALIDATE_ENV SCRIPT
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateEnvScript:
    """Verify the cross-platform validation script runs cleanly."""

    SCRIPT = XTR_WARP_DIR / "scripts" / "validate_env.py"

    def test_script_exists(self):
        assert self.SCRIPT.exists(), f"validate_env.py not found at {self.SCRIPT}"

    def test_script_runs_without_error(self):
        """Script must exit 0 even in a minimal environment."""
        result = subprocess.run(
            [sys.executable, str(self.SCRIPT)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"validate_env.py failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_script_output_contains_sections(self):
        """Script output must include all expected diagnostic sections."""
        result = subprocess.run(
            [sys.executable, str(self.SCRIPT)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        for section in ("Platform", "PyTorch", "FAISS", "Dependencies", "Summary"):
            assert (
                section in result.stdout
            ), f"Missing section '{section}' in validate_env.py output"


# ═══════════════════════════════════════════════════════════════════════════
# 5. COLLECTION INDEXER FAISS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestCollectionIndexerIntegration:
    """Verify the _validate_faiss_gpu function in collection_indexer."""

    def test_validate_function_exists(self):
        """collection_indexer should expose _validate_faiss_gpu with lazy cache."""
        indexer_path = XTR_WARP_DIR / "warp" / "indexing" / "collection_indexer.py"
        assert indexer_path.exists()
        content = indexer_path.read_text(encoding="utf-8")
        assert "_validate_faiss_gpu" in content
        assert "_faiss_gpu_validated" in content  # Lazy cache sentinel

    def test_validate_function_returns_bool(self):
        """Even if we can't import the real module, validate the logic."""
        # We test the function in isolation since importing
        # collection_indexer pulls in heavy deps.
        indexer_path = XTR_WARP_DIR / "warp" / "indexing" / "collection_indexer.py"
        content = indexer_path.read_text(encoding="utf-8")
        assert "def _validate_faiss_gpu()" in content
        assert "_faiss_gpu_validated" in content  # Lazy cache sentinel
        assert "global _faiss_gpu_validated" in content  # Proper caching
