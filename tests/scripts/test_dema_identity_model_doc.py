from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "product" / "DEMA_IDENTITY_MODEL_V0_1.md"
GTM = ROOT / "docs" / "product" / "DEMA_GTM_MASTERPLAN_V0_1.md"
KERNEL = ROOT / "docs" / "product" / "DEMA_AMBIENT_KERNEL_V0_1.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dema_identity_model_locks_instance_names_and_privacy_boundary() -> None:
    text = _read(DOC)

    for required in [
        "DEMA Core",
        "DEMA-0",
        "Mumu-DEMA",
        "Node0-DEMA",
        "Genesis Dema",
        "DEMA-1..DEMA-n",
    ]:
        assert required in text

    assert "Future Dema instances inherit BIZRA DNA" in text
    assert "They do **not** inherit Mumu's private" in text
    assert "Private Node Memory != Shared URP Knowledge" in text


def test_dema_identity_model_locks_memory_ingestion_classification() -> None:
    text = _read(DOC)

    for label in [
        "MUMU_PRIVATE_MEMORY",
        "BIZRA_CANON",
        "URP_SHAREABLE_KNOWLEDGE",
        "UNVERIFIED_OR_AMBIGUOUS",
    ]:
        assert label in text

    assert "No branch in this pseudocode may write raw private chat history" in text
    assert "consent receipt" in text
    assert "access policy" in text


def test_product_docs_cross_link_identity_prerequisite() -> None:
    gtm = _read(GTM)
    kernel = _read(KERNEL)

    assert "DEMA_IDENTITY_MODEL_V0_1.md" in gtm
    assert "DEMA_IDENTITY_MODEL_V0_1.md" in kernel
    assert "Future-node bootstrap path cannot inherit Node0-private records" in gtm
