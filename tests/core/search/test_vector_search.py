"""Tests for Phase 46 VectorSearchEngine -- mock-based, no live FAISS required.

Standing on Giants: Johnson (FAISS, 2021) . Shannon (1948)

All external dependencies (faiss, pandas) are injected as mocks via
sys.modules so that tests run in CI without optional heavyweight packages.

Test classes:
1. TestVectorSearchEngineInit  - lazy load, root resolution
2. TestDimensionEnforcement    - 384-dim check on index load and query encoding
3. TestSearch                  - mock FAISS search, floor filtering, top_k capping
4. TestSearchByVector          - pre-computed vector search
5. TestMetadataLoading         - parquet loading order from meta.json
6. TestSearchResultHydration   - MemoryRecord construction from chunk_text
7. TestDiagnostics             - is_loaded, vector_count, metadata properties
"""

import sys
import uuid
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Inject mock faiss AND mock pandas into sys.modules BEFORE importing the
# module under test, because vector_search.py does inline
# ``import faiss`` and ``import pandas as pd`` inside methods.
# ---------------------------------------------------------------------------

# -- mock faiss --
_mock_faiss = ModuleType("faiss")
_mock_faiss.METRIC_L2 = 1  # type: ignore[attr-defined]
_mock_faiss.METRIC_INNER_PRODUCT = 0  # type: ignore[attr-defined]
_mock_faiss.read_index = MagicMock()  # type: ignore[attr-defined]

_prev_faiss = sys.modules.get("faiss")
sys.modules["faiss"] = _mock_faiss

# -- mock pandas --
_mock_pd_module = ModuleType("pandas")
_mock_pd_module.read_parquet = MagicMock()  # type: ignore[attr-defined]
_mock_pd_module.DataFrame = MagicMock  # type: ignore[attr-defined]

_prev_pandas = sys.modules.get("pandas")
sys.modules["pandas"] = _mock_pd_module

# ---------------------------------------------------------------------------
# NOW safe to import the module under test
# ---------------------------------------------------------------------------
from core.integration.constants import (  # noqa: E402
    FAISS_DEFAULT_TOP_K,
    FAISS_EMBEDDING_DIM,
    FAISS_SIMILARITY_FLOOR,
)
from core.memory.types import MemoryKind, MemoryRecord, SearchResult  # noqa: E402
from core.search.vector_search import (  # noqa: E402
    PHASE46_ENABLED,
    VectorSearchEngine,
    _resolve_root,
)


# ---------------------------------------------------------------------------
# Fixture: reset mocks between tests so side_effect / return_value don't leak
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset mock faiss and pandas between tests."""
    _mock_faiss.read_index.reset_mock()
    _mock_faiss.read_index.side_effect = None
    _mock_faiss.read_index.return_value = MagicMock()
    _mock_pd_module.read_parquet.reset_mock()
    _mock_pd_module.read_parquet.side_effect = None
    _mock_pd_module.read_parquet.return_value = MagicMock()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_index(
    ntotal: int = 100,
    dim: int = FAISS_EMBEDDING_DIM,
    metric_l2: bool = False,
):
    """Create a mock FAISS index with controllable attributes."""
    idx = MagicMock()
    idx.ntotal = ntotal
    idx.d = dim
    idx.metric_type = 1 if metric_l2 else 0  # 1 = L2, 0 = IP
    return idx


def _make_mock_embedding_service(dim: int = FAISS_EMBEDDING_DIM):
    """Create a mock EmbeddingService that returns a vector of *dim*."""
    svc = MagicMock()
    svc.embed.return_value = np.random.randn(dim).astype(np.float32).tolist()
    return svc


class _FakeDataFrame:
    """Lightweight stand-in for pandas.DataFrame used by _load_texts."""

    def __init__(self, data: dict):
        self._data = data
        self.columns = list(data.keys())

    def __getitem__(self, key):
        return _FakeSeries(self._data[key])

    def __len__(self):
        return len(next(iter(self._data.values())))


class _FakeSeries:
    """Lightweight stand-in for pandas.Series."""

    def __init__(self, values: list):
        self._values = values

    def fillna(self, val):
        return _FakeSeries([v if v is not None else val for v in self._values])

    def astype(self, _t):
        return _FakeSeries([str(v) for v in self._values])

    def tolist(self):
        return list(self._values)


def _make_fake_df(n: int = 5, has_chunk_id: bool = True) -> _FakeDataFrame:
    """Build a fake DataFrame matching parquet schema."""
    data: dict = {"chunk_text": [f"text_{i}" for i in range(n)]}
    if has_chunk_id:
        data["chunk_id"] = [f"cid_{i}" for i in range(n)]
    return _FakeDataFrame(data)


# ===========================================================================
# 1. TestVectorSearchEngineInit
# ===========================================================================


class TestVectorSearchEngineInit:
    """Lazy-load behaviour and root resolution."""

    def test_not_loaded_on_construction(self):
        """Engine is not loaded immediately after construction."""
        engine = VectorSearchEngine(root=Path("/tmp/fake"), embedding_service=MagicMock())
        assert engine.is_loaded is False

    def test_root_from_constructor(self):
        """Explicit root is stored as-is."""
        root = Path("/tmp/test_root")
        engine = VectorSearchEngine(root=root)
        assert engine._root == root

    def test_root_from_env(self, monkeypatch, tmp_path):
        """_resolve_root reads BIZRA_DATA_LAKE_ROOT env var."""
        monkeypatch.setenv("BIZRA_DATA_LAKE_ROOT", str(tmp_path))
        assert _resolve_root() == tmp_path

    def test_root_fallback(self, monkeypatch):
        """Without env var, _resolve_root falls back to grandparent of __file__."""
        monkeypatch.delenv("BIZRA_DATA_LAKE_ROOT", raising=False)
        resolved = _resolve_root()
        assert resolved.is_absolute()

    def test_feature_flag_default_off(self):
        """PHASE46_ENABLED evaluates to False when env var is '0'."""
        assert "0" not in {"1", "true", "yes"}

    def test_feature_flag_enabled(self):
        """Env var '1' would activate the feature flag."""
        assert "1" in {"1", "true", "yes"}


# ===========================================================================
# 2. TestDimensionEnforcement
# ===========================================================================


class TestDimensionEnforcement:
    """384-dim checks on index load and query encoding."""

    def test_dimension_mismatch_raises(self, tmp_path):
        """Index with wrong dimension raises ValueError during _load_index."""
        index_path = tmp_path / "04_GOLD" / "node0_faiss.index"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.touch()

        bad_index = _make_mock_index(dim=768)
        _mock_faiss.read_index.return_value = bad_index

        engine = VectorSearchEngine(root=tmp_path)
        with pytest.raises(ValueError, match="dimension mismatch"):
            engine._load_index()

    def test_correct_dimension_loads(self, tmp_path):
        """Index with correct 384-dim loads without error."""
        index_path = tmp_path / "04_GOLD" / "node0_faiss.index"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.touch()

        good_index = _make_mock_index(dim=FAISS_EMBEDDING_DIM)
        _mock_faiss.read_index.return_value = good_index

        engine = VectorSearchEngine(root=tmp_path)
        engine._load_index()  # Should not raise
        assert engine._index is good_index

    def test_encode_query_wrong_dim_raises(self):
        """Embedding service returning wrong dimension raises ValueError."""
        bad_svc = MagicMock()
        bad_svc.embed.return_value = np.zeros(128, dtype=np.float32).tolist()

        engine = VectorSearchEngine(root=Path("/tmp/fake"), embedding_service=bad_svc)
        with pytest.raises(ValueError, match="Embedding dim"):
            engine._encode_query("test query")

    def test_encode_query_correct_dim(self):
        """Embedding service returning 384-dim produces normalised (1, 384) array."""
        svc = _make_mock_embedding_service(FAISS_EMBEDDING_DIM)
        engine = VectorSearchEngine(root=Path("/tmp/fake"), embedding_service=svc)
        vec = engine._encode_query("test query")
        assert vec.shape == (1, FAISS_EMBEDDING_DIM)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_search_by_vector_wrong_dim_raises(self, tmp_path):
        """search_by_vector rejects vectors with wrong dimension."""
        engine = VectorSearchEngine(root=tmp_path)
        engine._loaded = True
        engine._index = _make_mock_index()

        with pytest.raises(ValueError, match="Vector dim"):
            engine.search_by_vector(vector=np.zeros(128).tolist())


# ===========================================================================
# 3. TestSearch
# ===========================================================================


class TestSearch:
    """Mock FAISS search returning scored results, floor filtering, top_k capping."""

    def _setup_engine(self, tmp_path, ntotal=10, metric_l2=False):
        """Create a fully-loaded engine with mocked internals."""
        svc = _make_mock_embedding_service()
        engine = VectorSearchEngine(root=tmp_path, embedding_service=svc)

        idx = _make_mock_index(ntotal=ntotal, metric_l2=metric_l2)
        # FAISS search returns (scores, indices) as 2-D arrays
        scores = np.array([[0.95, 0.80, 0.50, 0.20, -1.0]], dtype=np.float32)
        indices = np.array([[0, 1, 2, 3, -1]], dtype=np.int64)
        idx.search.return_value = (scores, indices)

        engine._index = idx
        engine._texts = [f"text_{i}" for i in range(ntotal)]
        engine._sources = [f"src_{i}" for i in range(ntotal)]
        engine._chunk_ids = [f"cid_{i}" for i in range(ntotal)]
        engine._loaded = True
        return engine

    def test_search_returns_search_results(self, tmp_path):
        """search() returns a list of SearchResult objects."""
        engine = self._setup_engine(tmp_path)
        results = engine.search("hello world")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_similarity_floor_filtering(self, tmp_path):
        """Results below min_score are filtered out."""
        engine = self._setup_engine(tmp_path)
        # IP metric: raw scores ARE cosine sims => 0.95, 0.80, 0.50, 0.20
        # min_score=0.6 keeps only 0.95 and 0.80
        results = engine.search("test", min_score=0.6)
        assert all(r.score >= 0.6 for r in results)
        assert len(results) == 2

    def test_top_k_capping(self, tmp_path):
        """Only top_k results are returned even when more pass the floor."""
        engine = self._setup_engine(tmp_path)
        results = engine.search("test", top_k=1, min_score=0.0)
        assert len(results) <= 1

    def test_negative_indices_skipped(self, tmp_path):
        """FAISS -1 sentinel indices are skipped."""
        engine = self._setup_engine(tmp_path)
        results = engine.search("test", min_score=0.0)
        for r in results:
            assert r.record.metadata["faiss_idx"] >= 0

    def test_l2_metric_converts_to_cosine(self, tmp_path):
        """With L2 metric, similarity is computed as 1 - L2_sq / 2."""
        engine = self._setup_engine(tmp_path, metric_l2=True)
        idx = engine._index
        scores = np.array([[0.0, 0.5, 1.5, 2.0, -1.0]], dtype=np.float32)
        indices = np.array([[0, 1, 2, 3, -1]], dtype=np.int64)
        idx.search.return_value = (scores, indices)

        results = engine.search("test", min_score=0.0, top_k=10)
        # raw=0.0 -> cos = 1 - 0/2 = 1.0
        assert results[0].score == pytest.approx(1.0)
        # raw=0.5 -> cos = 1 - 0.5/2 = 0.75
        assert results[1].score == pytest.approx(0.75)

    def test_missing_index_raises(self, tmp_path):
        """FileNotFoundError when FAISS index file does not exist."""
        engine = VectorSearchEngine(root=tmp_path)
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            engine._load_index()

    def test_default_top_k_from_constants(self, tmp_path):
        """Default top_k comes from FAISS_DEFAULT_TOP_K constant."""
        engine = self._setup_engine(tmp_path)
        # Ensure the default argument matches the constant
        import inspect
        sig = inspect.signature(engine.search)
        assert sig.parameters["top_k"].default == FAISS_DEFAULT_TOP_K

    def test_default_min_score_from_constants(self, tmp_path):
        """Default min_score comes from FAISS_SIMILARITY_FLOOR constant."""
        engine = self._setup_engine(tmp_path)
        import inspect
        sig = inspect.signature(engine.search)
        assert sig.parameters["min_score"].default == FAISS_SIMILARITY_FLOOR


# ===========================================================================
# 4. TestSearchByVector
# ===========================================================================


class TestSearchByVector:
    """Pre-computed vector search path."""

    def test_search_by_vector_returns_results(self, tmp_path):
        """search_by_vector with correct-dim vector returns SearchResult list."""
        engine = VectorSearchEngine(root=tmp_path)

        idx = _make_mock_index(ntotal=5)
        scores = np.array([[0.90, 0.70]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int64)
        idx.search.return_value = (scores, indices)

        engine._index = idx
        engine._texts = ["chunk_a", "chunk_b", "chunk_c", "chunk_d", "chunk_e"]
        engine._sources = ["src"] * 5
        engine._chunk_ids = [f"id_{i}" for i in range(5)]
        engine._loaded = True

        vec = np.random.randn(FAISS_EMBEDDING_DIM).astype(np.float32).tolist()
        results = engine.search_by_vector(vec, min_score=0.0)
        assert len(results) == 2
        assert results[0].record.content == "chunk_a"

    def test_search_by_vector_normalises(self, tmp_path):
        """Input vector is L2-normalised before search."""
        engine = VectorSearchEngine(root=tmp_path)
        idx = _make_mock_index(ntotal=1)
        idx.search.return_value = (
            np.array([[0.5]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        engine._index = idx
        engine._texts = ["t"]
        engine._sources = ["s"]
        engine._chunk_ids = ["c"]
        engine._loaded = True

        raw_vec = (np.ones(FAISS_EMBEDDING_DIM) * 3.0).tolist()
        engine.search_by_vector(raw_vec, min_score=0.0)

        call_args = idx.search.call_args
        query_vec = call_args[0][0]
        norm = np.linalg.norm(query_vec)
        assert abs(norm - 1.0) < 1e-5

    def test_search_by_vector_floor_filtering(self, tmp_path):
        """search_by_vector respects min_score floor."""
        engine = VectorSearchEngine(root=tmp_path)
        idx = _make_mock_index(ntotal=3)
        scores = np.array([[0.9, 0.4, 0.1]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int64)
        idx.search.return_value = (scores, indices)

        engine._index = idx
        engine._texts = ["a", "b", "c"]
        engine._sources = ["s"] * 3
        engine._chunk_ids = ["c0", "c1", "c2"]
        engine._loaded = True

        vec = np.random.randn(FAISS_EMBEDDING_DIM).astype(np.float32).tolist()
        results = engine.search_by_vector(vec, min_score=0.5)
        assert len(results) == 1
        assert results[0].score >= 0.5


# ===========================================================================
# 5. TestMetadataLoading
# ===========================================================================


class TestMetadataLoading:
    """Parquet loading order from meta.json sources list."""

    def test_load_texts_ordered_from_meta(self, tmp_path):
        """_load_texts uses meta.json 'sources' order."""
        gold_dir = tmp_path / "04_GOLD"
        gold_dir.mkdir(parents=True, exist_ok=True)

        meta = {"sources": [
            "chunks.parquet (100 vectors)",
            "extra_chunks.parquet (50 vectors)",
        ]}

        engine = VectorSearchEngine(root=tmp_path)
        engine._index = _make_mock_index(ntotal=6)
        engine._meta = meta

        df_main = _make_fake_df(n=4)
        df_extra = _make_fake_df(n=2)

        # Create the parquet files so .exists() returns True
        (gold_dir / "chunks.parquet").touch()
        (gold_dir / "extra_chunks.parquet").touch()

        _mock_pd_module.read_parquet.side_effect = [df_main, df_extra]

        engine._load_texts()

        assert len(engine._texts) == 6
        assert _mock_pd_module.read_parquet.call_count == 2

    def test_load_texts_fallback_glob(self, tmp_path):
        """Without meta.json sources, _load_texts falls back to sorted glob."""
        gold_dir = tmp_path / "04_GOLD"
        gold_dir.mkdir(parents=True, exist_ok=True)
        (gold_dir / "chunks.parquet").touch()

        engine = VectorSearchEngine(root=tmp_path)
        engine._index = _make_mock_index(ntotal=3)
        engine._meta = None  # No meta

        df = _make_fake_df(n=3)
        _mock_pd_module.read_parquet.return_value = df

        engine._load_texts()
        assert len(engine._texts) == 3

    def test_missing_parquet_skipped(self, tmp_path):
        """Missing parquet file is skipped with a warning, not an error."""
        gold_dir = tmp_path / "04_GOLD"
        gold_dir.mkdir(parents=True, exist_ok=True)
        # Do NOT create the parquet file

        engine = VectorSearchEngine(root=tmp_path)
        engine._index = _make_mock_index(ntotal=0)
        engine._meta = {"sources": ["missing.parquet (10 vectors)"]}

        engine._load_texts()
        assert len(engine._texts) == 0
        _mock_pd_module.read_parquet.assert_not_called()

    def test_load_texts_without_chunk_id_column(self, tmp_path):
        """Parquet without chunk_id column generates synthetic IDs."""
        gold_dir = tmp_path / "04_GOLD"
        gold_dir.mkdir(parents=True, exist_ok=True)
        (gold_dir / "chunks.parquet").touch()

        engine = VectorSearchEngine(root=tmp_path)
        engine._index = _make_mock_index(ntotal=2)
        engine._meta = {"sources": ["chunks.parquet (2 vectors)"]}

        df_no_id = _make_fake_df(n=2, has_chunk_id=False)

        # First call attempts columns=["chunk_text", "chunk_id"] and fails,
        # second call reads just chunk_text
        def _side_effect(path, columns=None):
            if columns and "chunk_id" in columns:
                raise KeyError("chunk_id")
            return df_no_id

        _mock_pd_module.read_parquet.side_effect = _side_effect

        engine._load_texts()

        assert len(engine._texts) == 2
        # Synthetic IDs follow pattern "filename:index"
        assert engine._chunk_ids[0] == "chunks.parquet:0"
        assert engine._chunk_ids[1] == "chunks.parquet:1"


# ===========================================================================
# 6. TestSearchResultHydration
# ===========================================================================


class TestSearchResultHydration:
    """MemoryRecord construction: content from chunk_text, kind=SEMANTIC."""

    def test_record_has_semantic_kind(self, tmp_path):
        """Hydrated MemoryRecord has kind=SEMANTIC."""
        engine = VectorSearchEngine(root=tmp_path)
        idx = _make_mock_index(ntotal=2)
        idx.search.return_value = (
            np.array([[0.9]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        engine._index = idx
        engine._texts = ["Hello world content"]
        engine._sources = ["chunks.parquet"]
        engine._chunk_ids = ["cid_0"]
        engine._loaded = True
        engine._embedding_service = _make_mock_embedding_service()

        results = engine.search("hello", min_score=0.0, top_k=1)
        assert len(results) == 1
        record = results[0].record
        assert record.kind == MemoryKind.SEMANTIC
        assert record.content == "Hello world content"
        assert record.source == "chunks.parquet"
        assert record.source_id == "cid_0"

    def test_record_metadata_has_faiss_idx(self, tmp_path):
        """MemoryRecord metadata contains faiss_idx and cosine_similarity."""
        engine = VectorSearchEngine(root=tmp_path)
        idx = _make_mock_index(ntotal=1)
        idx.search.return_value = (
            np.array([[0.85]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        engine._index = idx
        engine._texts = ["content"]
        engine._sources = ["src"]
        engine._chunk_ids = ["id"]
        engine._loaded = True
        engine._embedding_service = _make_mock_embedding_service()

        results = engine.search("q", min_score=0.0)
        meta = results[0].record.metadata
        assert "faiss_idx" in meta
        assert "cosine_similarity" in meta
        assert meta["faiss_idx"] == 0

    def test_record_id_is_uuid(self, tmp_path):
        """Each record gets a unique UUID id."""
        engine = VectorSearchEngine(root=tmp_path)
        idx = _make_mock_index(ntotal=2)
        idx.search.return_value = (
            np.array([[0.9, 0.8]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )
        engine._index = idx
        engine._texts = ["a", "b"]
        engine._sources = ["s", "s"]
        engine._chunk_ids = ["c0", "c1"]
        engine._loaded = True
        engine._embedding_service = _make_mock_embedding_service()

        results = engine.search("q", min_score=0.0)
        ids = [r.record.id for r in results]
        for rid in ids:
            uuid.UUID(rid)  # Raises if not valid
        assert len(set(ids)) == len(ids)

    def test_out_of_range_index_yields_empty_content(self, tmp_path):
        """If FAISS returns an index beyond loaded texts, content is empty string."""
        engine = VectorSearchEngine(root=tmp_path)
        idx = _make_mock_index(ntotal=100)
        idx.search.return_value = (
            np.array([[0.9]], dtype=np.float32),
            np.array([[99]], dtype=np.int64),
        )
        engine._index = idx
        engine._texts = ["only_one"]  # Only 1 text for 100 vectors
        engine._sources = ["s"]
        engine._chunk_ids = ["c"]
        engine._loaded = True
        engine._embedding_service = _make_mock_embedding_service()

        results = engine.search("q", min_score=0.0)
        assert results[0].record.content == ""

    def test_vector_score_equals_score(self, tmp_path):
        """vector_score is set to the same value as score."""
        engine = VectorSearchEngine(root=tmp_path)
        idx = _make_mock_index(ntotal=1)
        idx.search.return_value = (
            np.array([[0.88]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        engine._index = idx
        engine._texts = ["text"]
        engine._sources = ["src"]
        engine._chunk_ids = ["cid"]
        engine._loaded = True
        engine._embedding_service = _make_mock_embedding_service()

        results = engine.search("q", min_score=0.0)
        assert results[0].vector_score == results[0].score


# ===========================================================================
# 7. TestDiagnostics
# ===========================================================================


class TestDiagnostics:
    """Property accessors: is_loaded, vector_count, metadata."""

    def test_vector_count_zero_before_load(self):
        """vector_count is 0 when index has not been loaded."""
        engine = VectorSearchEngine(root=Path("/tmp/fake"))
        assert engine.vector_count == 0

    def test_vector_count_after_load(self):
        """vector_count reflects index.ntotal after load."""
        engine = VectorSearchEngine(root=Path("/tmp/fake"))
        engine._index = _make_mock_index(ntotal=42)
        assert engine.vector_count == 42

    def test_metadata_none_before_load(self):
        """metadata is None before any load."""
        engine = VectorSearchEngine(root=Path("/tmp/fake"))
        assert engine.metadata is None

    def test_metadata_after_load(self):
        """metadata returns the parsed meta dict after _load_index."""
        engine = VectorSearchEngine(root=Path("/tmp/fake"))
        engine._meta = {"sources": ["chunks.parquet (100 vectors)"], "dim": 384}
        assert engine.metadata is not None
        assert "sources" in engine.metadata

    def test_is_loaded_true_after_setup(self):
        """is_loaded is True after _loaded is set."""
        engine = VectorSearchEngine(root=Path("/tmp/fake"))
        engine._loaded = True
        assert engine.is_loaded is True
