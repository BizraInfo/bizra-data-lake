#!/usr/bin/env python3
"""
Fail CI when deploy overlays or the deployment workflow drift from the reviewed
Kustomize contract.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"

OVERLAY_CONTRACT = {
    "dev": {
        "namespace": "bizra-dev",
        "hosts": ("api.dev.bizra.node0", "elite.dev.bizra.node0"),
    },
    "staging": {
        "namespace": "bizra-staging",
        "hosts": ("api.staging.bizra.node0", "elite.staging.bizra.node0"),
    },
    "production": {
        "namespace": "bizra",
        "hosts": ("api.bizra.node0", "elite.bizra.node0"),
    },
}

REQUIRED_IMAGES = ("bizra-elite", "bizra-omega", "bizra-mcp")
REQUIRED_WORKFLOW_SNIPPETS = (
    "deploy/k8s/overlays/staging",
    "deploy/k8s/overlays/production",
    "svc/bizra-elite 18080:80",
    "svc/bizra-omega 13001:80",
    "/v1/health",
    "/api/v1/health",
)


@dataclass(frozen=True)
class DeployOverlayReport:
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def format(self) -> str:
        if self.ok:
            return "No deploy-overlay issues."
        return "\n".join(f"- {issue}" for issue in self.issues)


def _render_overlay(overlay_dir: Path) -> str:
    completed = subprocess.run(
        ["kubectl", "kustomize", str(overlay_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def validate_deploy_overlay_governance(
    root: Path = ROOT,
    deploy_workflow: Path = DEPLOY_WORKFLOW,
) -> DeployOverlayReport:
    issues: list[str] = []

    for overlay_name, contract in OVERLAY_CONTRACT.items():
        overlay_dir = root / "deploy" / "k8s" / "overlays" / overlay_name
        kustomization = overlay_dir / "kustomization.yaml"
        if not overlay_dir.is_dir():
            issues.append(f"overlay '{overlay_name}' is missing at {overlay_dir}")
            continue
        if not kustomization.is_file():
            issues.append(f"overlay '{overlay_name}' is missing kustomization.yaml")
            continue

        kustomization_text = kustomization.read_text(encoding="utf-8")
        if "../../base" not in kustomization_text:
            issues.append(
                f"overlay '{overlay_name}' must inherit from ../../base"
            )
        for image_name in REQUIRED_IMAGES:
            if image_name not in kustomization_text:
                issues.append(
                    f"overlay '{overlay_name}' is missing image mapping for {image_name}"
                )

        try:
            rendered = _render_overlay(overlay_dir)
        except subprocess.CalledProcessError as exc:
            issues.append(
                f"overlay '{overlay_name}' failed to render via kubectl kustomize: "
                f"{exc.stderr.strip() or exc.stdout.strip()}"
            )
            continue

        if "RELEASE_TAG" in rendered:
            issues.append(
                f"overlay '{overlay_name}' still renders RELEASE_TAG placeholders"
            )
        if f"name: {contract['namespace']}" not in rendered:
            issues.append(
                f"overlay '{overlay_name}' does not render namespace {contract['namespace']}"
            )
        if f"namespace: {contract['namespace']}" not in rendered:
            issues.append(
                f"overlay '{overlay_name}' does not stamp namespaced resources with "
                f"{contract['namespace']}"
            )
        for host in contract["hosts"]:
            if host not in rendered:
                issues.append(
                    f"overlay '{overlay_name}' is missing ingress host {host}"
                )

    workflow_text = deploy_workflow.read_text(encoding="utf-8")
    for snippet in REQUIRED_WORKFLOW_SNIPPETS:
        if snippet not in workflow_text:
            issues.append(
                f"deploy workflow is missing required contract snippet: {snippet}"
            )

    return DeployOverlayReport(issues=tuple(issues))


def main() -> int:
    report = validate_deploy_overlay_governance()
    if not report.ok:
        print("[DEPLOY-OVERLAY-GATE] FAILED")
        print(report.format())
        return 1

    print("[DEPLOY-OVERLAY-GATE] PASS")
    for overlay_name in OVERLAY_CONTRACT:
        print(f"Validated deploy/k8s/overlays/{overlay_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
