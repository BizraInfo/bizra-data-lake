from __future__ import annotations

import json

from scripts.intake_external_asset_bundle import (
    _categorize,
    build_manifest,
    main,
)


def test_categorize_high_signal_governance_artifact(tmp_path) -> None:
    path = tmp_path / "proof_kernel.py"
    path.write_text("print('proof')\n", encoding="utf-8")

    category, action, rationale = _categorize(path)

    assert category == "research_governance_candidate"
    assert action == "merge_candidate"
    assert "repo relevance" in rationale


def test_build_manifest_classifies_bundle_files(tmp_path) -> None:
    (tmp_path / "BIZRA_SovereignCockpit.jsx").write_text(
        "export default null\n", encoding="utf-8"
    )
    (tmp_path / "GTM_90Day_Launch_Plan.docx").write_bytes(b"fake-docx")

    manifest = build_manifest(tmp_path)

    assert len(manifest) == 2
    by_name = {record.name: record for record in manifest}
    assert (
        by_name["BIZRA_SovereignCockpit.jsx"].recommended_action
        == "prototype_merge_candidate"
    )
    assert by_name["GTM_90Day_Launch_Plan.docx"].recommended_action == "reference_only"


def test_main_writes_manifest_and_markdown(tmp_path, monkeypatch) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "CMN_EVIDENCE_REDRAFT.md").write_text("# hi\n", encoding="utf-8")
    output_json = tmp_path / "manifest.json"
    output_md = tmp_path / "manifest.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "intake_external_asset_bundle.py",
            "--root",
            str(root),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    exit_code = main()

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["file_count"] == 1
    assert output_md.read_text(encoding="utf-8").startswith("# External Bundle Intake")
