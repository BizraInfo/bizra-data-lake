from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml_docs(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def test_no_latest_tags_in_bizra_omega_compose() -> None:
    compose_path = ROOT / "bizra-omega" / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = compose.get("services", {})
    assert services, "Expected services in bizra-omega/docker-compose.yml"

    for service_name, service in services.items():
        image = service.get("image", "")
        if image:
            assert ":latest" not in image, f"{service_name} uses mutable :latest tag"


def test_rbac_secrets_restricted_to_named_resource() -> None:
    rbac_path = ROOT / "deploy" / "k8s" / "base" / "rbac.yaml"
    docs = _load_yaml_docs(rbac_path)
    role_doc = next((doc for doc in docs if doc.get("kind") == "Role"), None)

    assert role_doc is not None, "Expected Role document in RBAC manifest"
    rules = role_doc.get("rules", [])
    secrets_rule = next(
        (rule for rule in rules if "secrets" in rule.get("resources", [])),
        None,
    )
    assert secrets_rule is not None, "Expected explicit secrets rule in RBAC manifest"
    assert secrets_rule.get("resourceNames") == ["bizra-secrets"]
    assert secrets_rule.get("verbs") == ["get"]


def test_rbac_pods_rule_is_get_only() -> None:
    rbac_path = ROOT / "deploy" / "k8s" / "base" / "rbac.yaml"
    docs = _load_yaml_docs(rbac_path)
    role_doc = next((doc for doc in docs if doc.get("kind") == "Role"), None)

    assert role_doc is not None, "Expected Role document in RBAC manifest"
    rules = role_doc.get("rules", [])
    pods_rule = next(
        (rule for rule in rules if "pods" in rule.get("resources", [])),
        None,
    )
    assert pods_rule is not None, "Expected pods rule in RBAC manifest"
    assert pods_rule.get("verbs") == ["get"]


def test_index_has_csp_and_no_inline_script_blocks() -> None:
    html_path = ROOT / "filedfs" / "index.html"
    html = html_path.read_text(encoding="utf-8")

    assert "Content-Security-Policy" in html
    assert "script-src 'self'" in html
    assert "object-src 'none'" in html
    assert '/sw-register.js"></script>' in html

    inline_scripts = re.findall(
        r"<script(?![^>]*\bsrc=)[^>]*>.*?</script>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert inline_scripts == []


def test_sw_register_exists_and_registers_worker() -> None:
    sw_register = ROOT / "filedfs" / "sw-register.js"
    content = sw_register.read_text(encoding="utf-8")

    assert "serviceWorker" in content
    assert re.search(
        r"""register\((['"])\/service-worker\.js\1\)""",
        content,
    )


def test_service_worker_uses_asset_allowlist_caching() -> None:
    sw_path = ROOT / "filedfs" / "service-worker.js"
    sw = sw_path.read_text(encoding="utf-8")

    assert "CACHEABLE_ASSET_REGEX" in sw
    assert 'request.mode === "navigate"' in sw
    assert "if (!isCacheableAssetRequest(request)) return;" in sw
    assert "manifest.json" not in sw
