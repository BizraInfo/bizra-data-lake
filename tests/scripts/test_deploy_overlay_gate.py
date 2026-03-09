from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts import ci_deploy_overlay_gate as gate


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _overlay_kustomization(namespace: str) -> str:
    return f"""
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: {namespace}
images:
  - name: bizra-elite
    newName: ghcr.io/bizra/bizra-data-lake/elite
    newTag: elite-placeholder
  - name: bizra-omega
    newName: ghcr.io/bizra/bizra-data-lake/omega
    newTag: omega-placeholder
  - name: bizra-mcp
    newName: ghcr.io/bizra/bizra-data-lake/mcp
    newTag: mcp-placeholder
""".strip()


def _rendered_overlay(namespace: str, hosts: tuple[str, str]) -> str:
    return f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bizra-elite
  namespace: {namespace}
spec:
  template:
    spec:
      containers:
        - name: elite
          image: ghcr.io/bizra/bizra-data-lake/elite:sha-abc123
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bizra-ingress
  namespace: {namespace}
spec:
  rules:
    - host: {hosts[0]}
    - host: {hosts[1]}
""".strip()


def _workflow_text() -> str:
    return """
deploy/k8s/overlays/staging
deploy/k8s/overlays/production
svc/bizra-elite 18080:80
svc/bizra-omega 13001:80
/v1/health
/api/v1/health
""".strip()


def test_deploy_overlay_gate_accepts_aligned_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for overlay_name, contract in gate.OVERLAY_CONTRACT.items():
        _write(
            tmp_path
            / "deploy"
            / "k8s"
            / "overlays"
            / overlay_name
            / "kustomization.yaml",
            _overlay_kustomization(contract["namespace"]),
        )
    _write(tmp_path / ".github" / "workflows" / "deploy.yml", _workflow_text())

    def fake_run(cmd: list[str], **_: object) -> CompletedProcess[str]:
        overlay_name = Path(cmd[-1]).name
        contract = gate.OVERLAY_CONTRACT[overlay_name]
        return CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_rendered_overlay(contract["namespace"], contract["hosts"]),
            stderr="",
        )

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    report = gate.validate_deploy_overlay_governance(
        root=tmp_path,
        deploy_workflow=tmp_path / ".github" / "workflows" / "deploy.yml",
    )

    assert report.ok, report.format()


def test_deploy_overlay_gate_reports_missing_overlay(tmp_path: Path) -> None:
    _write(tmp_path / ".github" / "workflows" / "deploy.yml", _workflow_text())

    report = gate.validate_deploy_overlay_governance(
        root=tmp_path,
        deploy_workflow=tmp_path / ".github" / "workflows" / "deploy.yml",
    )

    assert not report.ok
    assert any("overlay 'dev' is missing" in issue for issue in report.issues)


def test_deploy_overlay_gate_reports_workflow_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for overlay_name, contract in gate.OVERLAY_CONTRACT.items():
        _write(
            tmp_path
            / "deploy"
            / "k8s"
            / "overlays"
            / overlay_name
            / "kustomization.yaml",
            _overlay_kustomization(contract["namespace"]),
        )
    _write(
        tmp_path / ".github" / "workflows" / "deploy.yml", "deploy/k8s/overlays/staging"
    )

    def fake_run(cmd: list[str], **_: object) -> CompletedProcess[str]:
        overlay_name = Path(cmd[-1]).name
        contract = gate.OVERLAY_CONTRACT[overlay_name]
        return CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=_rendered_overlay(contract["namespace"], contract["hosts"]),
            stderr="",
        )

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    report = gate.validate_deploy_overlay_governance(
        root=tmp_path,
        deploy_workflow=tmp_path / ".github" / "workflows" / "deploy.yml",
    )

    assert not report.ok
    assert any(
        "deploy workflow is missing required contract snippet" in issue
        for issue in report.issues
    )
