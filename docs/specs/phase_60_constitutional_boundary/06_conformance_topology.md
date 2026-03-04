# Step 6: Conformance Test Topology Alignment

## Standing on Giants: Dijkstra (contracts between modules) | Fowler (consumer-driven contracts)

## Problem Statement

SAPE audit finding F6 (verified TRUE): Conformance tests in the production
artifact pack target ports that don't match the docker-compose service map.

The compose file maps services to specific ports. The conformance tests
assume different ports. Result: CI conformance tests are permanently broken
against the actual deployment topology.

**Compounding issue (F12, verified UNDERSTATED):** The OpenAPI specification
declares 8 endpoints, but 0 are actually implemented in the gateway router.
The conformance tests that DO exist test against the OpenAPI spec's endpoints,
which don't exist in the running service.

**Solution:**
1. Extract port mappings from docker-compose.yml programmatically
2. Generate a topology manifest that conformance tests consume
3. Add a CI step that validates test expectations match compose topology
4. Mark unimplemented OpenAPI endpoints as `x-status: planned`

## Target Files

| File | Action |
|------|--------|
| `scripts/ci/topology_validator.py` | New: reads compose YAML, validates test config |
| `.tmp_prod_artifacts_v2/tests/conftest.py` | Update: load ports from topology manifest |
| `.tmp_prod_artifacts_v2/deploy/topology.json` | New: generated topology manifest |
| `.tmp_prod_artifacts_v2/deploy/docker-compose.yml` | Reference: source of truth for ports |

## Pseudocode

### scripts/ci/topology_validator.py

```pseudocode
"""Validate that conformance test port expectations match compose topology.

Usage:
    python scripts/ci/topology_validator.py \
        --compose deploy/docker-compose.yml \
        --tests tests/ \
        --output deploy/topology.json

    python scripts/ci/topology_validator.py \
        --compose deploy/docker-compose.yml \
        --verify deploy/topology.json  # CI mode
"""

IMPORT yaml, json, re, sys, argparse
FROM pathlib IMPORT Path


FUNCTION extract_topology(compose_path: Path) -> dict:
    """Extract service→port mapping from docker-compose.yml.

    Returns:
        {
            "services": {
                "node_gateway": {"port": 8000, "host": "0.0.0.0"},
                "urp_registry": {"port": 8001, "host": "0.0.0.0"},
                ...
            },
            "source": str(compose_path),
            "version": "1.0.0"
        }
    """
    WITH open(compose_path) AS f:
        compose = yaml.safe_load(f)

    topology = {"services": {}, "source": str(compose_path), "version": "1.0.0"}

    FOR service_name, service_def IN compose.get("services", {}).items():
        ports = service_def.get("ports", [])
        IF ports:
            # Parse "host:container" or "host:container/proto" format
            FOR port_spec IN ports:
                port_str = str(port_spec)
                match = re.match(r"(\d+):(\d+)", port_str)
                IF match:
                    host_port = int(match.group(1))
                    container_port = int(match.group(2))
                    topology["services"][service_name] = {
                        "host_port": host_port,
                        "container_port": container_port,
                    }
                    BREAK  # Use first port mapping

    RETURN topology


FUNCTION scan_test_ports(test_dir: Path) -> dict:
    """Scan test files for hardcoded port numbers.

    Returns:
        {
            "file.py": [8000, 8001, 8011, ...],
            ...
        }
    """
    port_pattern = re.compile(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{4,5})")
    results = {}

    FOR test_file IN test_dir.rglob("*.py"):
        content = test_file.read_text()
        ports = [int(m) FOR m IN port_pattern.findall(content)]
        IF ports:
            results[str(test_file)] = sorted(set(ports))

    RETURN results


FUNCTION validate_alignment(
    topology: dict,
    test_ports: dict,
) -> list[str]:
    """Check that test ports exist in topology.

    Returns list of mismatches (empty = aligned).
    """
    valid_ports = set()
    FOR svc IN topology["services"].values():
        valid_ports.add(svc["host_port"])
        valid_ports.add(svc["container_port"])

    mismatches = []
    FOR file, ports IN test_ports.items():
        FOR port IN ports:
            IF port NOT IN valid_ports:
                mismatches.append(
                    f"{file}: port {port} not in compose topology"
                )

    RETURN mismatches


FUNCTION main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", required=True)
    parser.add_argument("--tests", default="tests/")
    parser.add_argument("--output", default="deploy/topology.json")
    parser.add_argument("--verify", help="Verify existing topology file")
    args = parser.parse_args()

    topology = extract_topology(Path(args.compose))

    IF args.verify:
        existing = json.loads(Path(args.verify).read_text())
        IF existing["services"] != topology["services"]:
            print("DRIFT: topology.json does not match compose file")
            sys.exit(1)
        print("OK: topology.json matches compose")
        sys.exit(0)

    # Scan and validate
    test_ports = scan_test_ports(Path(args.tests))
    mismatches = validate_alignment(topology, test_ports)

    IF mismatches:
        print(f"MISMATCHES ({len(mismatches)}):")
        FOR m IN mismatches:
            print(f"  - {m}")
        # Don't fail — report only. Let tests fix their own ports.

    # Write topology manifest
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n")
    print(f"Topology written: {output}")
```

### tests/conftest.py — Topology-Aware Fixtures

```pseudocode
# Add to artifact pack's tests/conftest.py:

IMPORT json
FROM pathlib IMPORT Path

FUNCTION _load_topology() -> dict:
    """Load port topology from generated manifest."""
    manifest = Path(__file__).parent.parent / "deploy" / "topology.json"
    IF manifest.exists():
        RETURN json.loads(manifest.read_text())
    # Fallback: default ports
    RETURN {"services": {
        "node_gateway": {"host_port": 8000, "container_port": 8000},
    }}

_TOPOLOGY = _load_topology()

@pytest.fixture
FUNCTION service_url(request):
    """Fixture: returns URL for a named service from topology."""
    service_name = request.param
    svc = _TOPOLOGY["services"].get(service_name, {})
    port = svc.get("host_port", 8000)
    RETURN f"http://localhost:{port}"
```

## TDD Anchors

```pseudocode
TEST extract_topology_parses_compose:
    compose_content = """
    services:
      gateway:
        ports:
          - "8000:8000"
      registry:
        ports:
          - "8001:8001"
    """
    write_file(tmp_path / "docker-compose.yml", compose_content)
    topology = extract_topology(tmp_path / "docker-compose.yml")
    ASSERT "gateway" IN topology["services"]
    ASSERT topology["services"]["gateway"]["host_port"] == 8000
    ASSERT topology["services"]["registry"]["host_port"] == 8001

TEST scan_test_ports_finds_hardcoded:
    test_content = 'url = "http://localhost:8011/health"'
    write_file(tmp_path / "test_example.py", test_content)
    ports = scan_test_ports(tmp_path)
    ASSERT 8011 IN ports[str(tmp_path / "test_example.py")]

TEST validate_alignment_detects_mismatch:
    topology = {"services": {"gw": {"host_port": 8000, "container_port": 8000}}}
    test_ports = {"test.py": [8011]}  # 8011 not in topology
    mismatches = validate_alignment(topology, test_ports)
    ASSERT len(mismatches) == 1
    ASSERT "8011" IN mismatches[0]

TEST validate_alignment_passes_when_aligned:
    topology = {"services": {"gw": {"host_port": 8000, "container_port": 8000}}}
    test_ports = {"test.py": [8000]}
    mismatches = validate_alignment(topology, test_ports)
    ASSERT len(mismatches) == 0

TEST topology_json_is_deterministic:
    compose = write_compose(tmp_path)
    t1 = extract_topology(compose)
    t2 = extract_topology(compose)
    ASSERT t1 == t2

TEST verify_mode_detects_drift:
    """--verify catches stale topology.json."""
    compose = write_compose(tmp_path, port=8000)
    write_file(tmp_path / "topology.json", '{"services": {"gw": {"host_port": 9999}}}')
    result = subprocess.run(
        ["python", "scripts/ci/topology_validator.py",
         "--compose", str(compose), "--verify", str(tmp_path / "topology.json")],
        capture_output=True
    )
    ASSERT result.returncode == 1
```

## Acceptance Criteria

1. `topology_validator.py` extracts ports from any docker-compose.yml
2. `topology.json` generated and matches compose service map
3. `--verify` mode detects drift between topology.json and compose file
4. Test conftest loads ports from topology manifest, not hardcoded values
5. Mismatched ports logged (non-blocking in Phase 60, blocking in Phase 61)
6. Full test suite GREEN
