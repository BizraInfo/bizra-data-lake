from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_bizra_bridge_binds_localhost_and_validates_clients() -> None:
    content = (ROOT / "filedfs" / "bizra-bridge.mjs").read_text(encoding="utf-8")

    assert 'LOCALHOST_BIND = "127.0.0.1"' in content
    assert "validateClientUpgrade(req)" in content
    assert "BIZRA_BRIDGE_TOKEN" in content
    assert "token_required" in content
    assert "BIZRA_BRIDGE_ALLOW_ANONYMOUS" in content
    assert "origin_rejected" in content
    assert "httpServer.listen(config.port, LOCALHOST_BIND" in content


def test_bizra_bridge_shutdown_requires_dedicated_token() -> None:
    content = (ROOT / "filedfs" / "bizra-bridge.mjs").read_text(encoding="utf-8")

    assert "BIZRA_BRIDGE_SHUTDOWN_TOKEN" in content
    assert "UNAUTHORIZED_SHUTDOWN" in content


def test_bridge_mjs_requires_json_and_rejects_non_local_origins() -> None:
    content = (ROOT / "filedfs" / "bridge.mjs").read_text(encoding="utf-8")

    assert "validateClientUpgrade(req)" in content
    assert "BIZRA_BRIDGE_TOKEN" in content
    assert "token_required" in content
    assert "isAllowedOrigin(origin)" in content
    assert "origin_rejected" in content
    assert "JSON message required" in content
    assert "const cmd = data.toString().trim();" not in content


def test_bridge_mjs_sanitizes_protocol_values() -> None:
    content = (ROOT / "filedfs" / "bridge.mjs").read_text(encoding="utf-8")

    assert "function sanitizeProtocolValue(value)" in content
    assert '.replace(/[\\t\\n\\r]/g, "")' in content
    assert "sanitizeProtocolValue(msg.line)" in content
    assert "BIZRA_BRIDGE_ALLOW_RAW" in content
    assert "RAW_DISABLED" in content
    assert "UNAUTHORIZED_SHUTDOWN" in content


def test_use_bizra_node_sanitizes_receive_and_teach_inputs() -> None:
    content = (ROOT / "filedfs" / "useBizraNode.js").read_text(encoding="utf-8")

    assert "function sanitizeProtocolValue(value)" in content
    assert "sanitizeProtocolValue(content)" in content
    assert "sanitizeProtocolValue(kind)" in content
    assert 'type: "command"' in content
