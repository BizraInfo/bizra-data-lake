from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_node0_manifest_redis_is_localhost_and_password_protected():
    manifest = _read("deploy/node0/node0-manifest.yaml")
    assert "--bind" in manifest and "127.0.0.1" in manifest
    assert "--requirepass" in manifest
    assert "REDIS_PASSWORD" in manifest
    assert "redis://:${REDIS_PASSWORD}@127.0.0.1:6379" in manifest


def test_elite_compose_removes_default_grafana_admin_password():
    compose = _read("deploy/elite-compose.yaml")
    assert "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:?" in compose
    assert "GRAFANA_ADMIN_PASSWORD:-admin" not in compose


def test_elite_compose_binds_observability_ports_to_localhost():
    compose = _read("deploy/elite-compose.yaml")
    assert "127.0.0.1:8095:8095" in compose
    assert "127.0.0.1:9090:9090" in compose
    assert "127.0.0.1:3000:3000" in compose


def test_systemd_services_default_to_localhost_binding():
    api_service = _read("deploy/node0/systemd-services/bizra-api.service")
    inference_service = _read("deploy/node0/systemd-services/bizra-inference.service")
    dashboard_service = _read("deploy/node0/systemd-services/bizra-dashboard.service")

    assert "Environment=\"BIZRA_API_HOST=127.0.0.1\"" in api_service
    assert "--host ${BIZRA_API_HOST}" in api_service
    assert "Environment=\"OLLAMA_HOST=127.0.0.1:11434\"" in inference_service
    assert "Environment=\"BIZRA_DASHBOARD_HOST=127.0.0.1\"" in dashboard_service
    assert "--host ${BIZRA_DASHBOARD_HOST}" in dashboard_service


def test_node0_env_example_documents_required_secrets_and_defaults():
    env_example = _read("deploy/node0/.env.example")
    for key in (
        "REDIS_PASSWORD=",
        "POSTGRES_PASSWORD=",
        "GRAFANA_ADMIN_PASSWORD=",
        "NODE_SECRET=",
        "BIZRA_MCP_GATEWAY_TOKEN=",
        "BIZRA_BRIDGE_TOKEN=",
        "BIZRA_BRIDGE_SHUTDOWN_TOKEN=",
        "BIZRA_API_HOST=127.0.0.1",
        "BIZRA_DASHBOARD_HOST=127.0.0.1",
    ):
        assert key in env_example


def test_node_secret_wiring_and_consumption_exist():
    k8s = _read("bizra-omega/k8s/deployment.yaml")
    rust_main = _read("bizra-omega/bizra-api/src/main.rs")

    assert "name: NODE_SECRET" in k8s
    assert "secretKeyRef" in k8s
    assert "key: NODE_SECRET" in k8s
    assert 'std::env::var("NODE_SECRET")' in rust_main
    assert ".mode(0o600)" in rust_main


def test_mcp_gateway_manifests_require_token_and_disable_anonymous():
    compose = _read("deploy/mcp-compose.yaml")
    k8s_deploy = _read("deploy/k8s/base/deployment-mcp.yaml")
    k8s_secrets = _read("deploy/k8s/base/secrets.yaml")

    assert "BIZRA_MCP_GATEWAY_TOKEN=${BIZRA_MCP_GATEWAY_TOKEN:?" in compose
    assert "BIZRA_MCP_ALLOW_ANONYMOUS=0" in compose
    assert "BIZRA_MCP_ALLOW_REMOTE=1" in compose

    assert "name: BIZRA_MCP_GATEWAY_TOKEN" in k8s_deploy
    assert "key: mcp-gateway-token" in k8s_deploy
    assert "name: BIZRA_MCP_ALLOW_ANONYMOUS" in k8s_deploy
    assert 'value: "false"' in k8s_deploy

    assert "mcp-gateway-token:" in k8s_secrets
