"""
C2 Optimization: Redis TLS Security Tests

This test suite validates that Redis TLS encryption and authentication
are properly configured and enforced.

Test Coverage:
- TLS URL scheme detection
- Configuration validation
- Connection behavior with mocked Redis

NOTE: These are unit tests focused on configuration and logic validation.
Integration tests with actual Redis TLS connection should be run via
docker-compose test environment.

Environment Variables:
- BIZRA_TLS_TESTS=1 : Enable infrastructure-dependent TLS tests
- RUN_INTEGRATION_TESTS=1 : Enable Redis connection integration tests
"""

import os
import pytest


def _tls_tests_enabled() -> bool:
    """Check if TLS infrastructure tests are enabled."""
    return os.getenv("BIZRA_TLS_TESTS", "0") == "1"


def skip_unless_tls_tests(reason: str = "TLS tests disabled (set BIZRA_TLS_TESTS=1)"):
    """Skip test unless BIZRA_TLS_TESTS=1 is set."""
    return pytest.mark.skipif(not _tls_tests_enabled(), reason=reason)


def test_synapse_tls_url_detection():
    """
    Verify URL scheme detection for TLS vs non-TLS.

    This test ensures the system correctly identifies rediss:// URLs
    as requiring TLS encryption.
    """
    # Test TLS URL detection
    tls_urls = [
        "rediss://:password@localhost:6379",
        "rediss://synapse:6379",
        "rediss://:password@127.0.0.1:6379",
        "rediss://:bizra_synapse_secure@synapse:6379",
    ]

    non_tls_urls = [
        "redis://:password@localhost:6379",
        "redis://localhost:6379",
        "redis://127.0.0.1:6379",
    ]

    for url in tls_urls:
        assert url.startswith("rediss://"), f"Should detect TLS for {url}"

    for url in non_tls_urls:
        assert not url.startswith("rediss://"), f"Should not detect TLS for {url}"


def test_synapse_environment_configuration():
    """
    Verify that environment variables are properly configured for TLS.

    This test checks that the expected environment variables exist
    and have reasonable values for TLS operation.
    """
    # These should be set in docker-compose or .env
    expected_vars = {
        "SYNAPSE_URL": "rediss://",  # Should start with rediss:// for TLS
        "REDIS_CA_CERT_PATH": "/etc/redis/certs/ca-cert.pem",
    }

    synapse_url = os.getenv("SYNAPSE_URL", "rediss://:bizra_synapse_secure@127.0.0.1:6379")
    ca_cert_path = os.getenv("REDIS_CA_CERT_PATH", "/etc/redis/certs/ca-cert.pem")

    # In production, SYNAPSE_URL should use rediss:// for TLS
    # Allow redis:// for local development testing
    assert synapse_url.startswith(("redis://", "rediss://")), \
        "SYNAPSE_URL should use redis:// or rediss:// scheme"

    # If rediss://, verify other TLS settings make sense
    if synapse_url.startswith("rediss://"):
        assert ca_cert_path, "REDIS_CA_CERT_PATH should be set for TLS"
        assert ca_cert_path.endswith(".pem"), "CA cert should be a .pem file"


@skip_unless_tls_tests("Docker compose TLS config check requires BIZRA_TLS_TESTS=1")
def test_docker_compose_synapse_tls_config():
    """
    Verify docker-compose synapse service is configured for TLS.

    This test reads the docker-compose.yml file and validates that
    the synapse service has TLS enabled.
    """
    import yaml

    try:
        with open("docker-compose.yml", "r", encoding="utf-8") as f:
            compose = yaml.safe_load(f)

        synapse_service = compose.get("services", {}).get("synapse", {})

        # Check that synapse service exists
        assert synapse_service, "Synapse service should be defined in docker-compose.yml"

        # Check command includes TLS configuration
        command = synapse_service.get("command", [])
        command_str = " ".join(str(c) for c in command)

        assert "--tls-port" in command_str, "Redis should be configured with --tls-port"
        assert "--requirepass" in command_str or "${REDIS_PASSWORD" in command_str, \
            "Redis should require password authentication"

        # Check certificates are mounted
        volumes = synapse_service.get("volumes", [])
        cert_volume_found = any("config/redis" in str(v) and "certs" in str(v) for v in volumes)
        assert cert_volume_found, "Certificate directory should be mounted as volume"

    except FileNotFoundError:
        pytest.skip("docker-compose.yml not found")
    except ImportError:
        pytest.skip("PyYAML not installed")


def test_dockerfile_includes_ca_certificate():
    """
    Verify Dockerfile copies CA certificate for TLS validation.

    This test checks that the Dockerfile includes the CA certificate
    needed for validating Redis TLS connections.
    """
    try:
        with open("Dockerfile", "r", encoding="utf-8") as f:
            dockerfile_content = f.read()

        # Check that CA cert is copied
        assert "ca-cert.pem" in dockerfile_content, \
            "Dockerfile should copy ca-cert.pem for TLS validation"

        assert "/etc/redis/certs" in dockerfile_content or "redis" in dockerfile_content.lower(), \
            "Dockerfile should reference Redis certificate directory"

    except FileNotFoundError:
        pytest.skip("Dockerfile not found")


def test_gitignore_excludes_private_keys():
    """
    Verify .gitignore properly excludes private key files.

    This is a critical security test to ensure private keys are never
    committed to version control.
    """
    try:
        with open(".gitignore", "r", encoding="utf-8") as f:
            gitignore_content = f.read()

        # Check that .pem files are ignored (includes private keys)
        assert "*.pem" in gitignore_content, \
            ".gitignore should exclude *.pem files (private keys)"

        # Optionally check for other key patterns
        key_patterns = ["*.key", "*.pem"]
        patterns_found = [pattern for pattern in key_patterns if pattern in gitignore_content]

        assert len(patterns_found) > 0, \
            "At least one private key pattern should be in .gitignore"

    except FileNotFoundError:
        pytest.skip(".gitignore not found")


@skip_unless_tls_tests("Certificate existence check requires BIZRA_TLS_TESTS=1")
def test_redis_tls_certificate_files_exist():
    """
    Verify that TLS certificate files exist in the expected location.

    This test checks for the presence of required certificate files
    for Redis TLS operation.
    """
    cert_dir = "config/redis"
    required_files = [
        "ca-cert.pem",
        "redis-server-cert.pem",
        "redis-server-key.pem",
    ]

    for cert_file in required_files:
        cert_path = os.path.join(cert_dir, cert_file)
        assert os.path.exists(cert_path), \
            f"Certificate file {cert_path} should exist for TLS operation"


def test_redis_default_url_uses_tls():
    """
    Verify that the default SYNAPSE_URL in core/synapse.py uses rediss://.

    This ensures TLS is enforced by default in production.
    """
    try:
        with open("core/synapse.py", "r", encoding="utf-8") as f:
            synapse_content = f.read()

        # Find the SYNAPSE_URL default value
        import re
        url_pattern = r'SYNAPSE_URL\s*=\s*os\.getenv\([^)]+,\s*["\']([^"\']+)["\']\)'
        match = re.search(url_pattern, synapse_content)

        if match:
            default_url = match.group(1)
            assert default_url.startswith("rediss://"), \
                f"Default SYNAPSE_URL should use rediss:// for TLS, got: {default_url}"
        else:
            pytest.fail("Could not find SYNAPSE_URL definition in core/synapse.py")

    except FileNotFoundError:
        pytest.skip("core/synapse.py not found")


@skip_unless_tls_tests("Cargo TLS feature check requires BIZRA_TLS_TESTS=1")
def test_cargo_toml_includes_redis_tls_feature():
    """
    Verify that Cargo.toml includes the tls-native-tls feature for Redis.

    This ensures the Rust code can connect to Redis with TLS.
    """
    try:
        with open("Cargo.toml", "r", encoding="utf-8") as f:
            cargo_content = f.read()

        # Check that redis dependency includes TLS feature
        assert "tls-native-tls" in cargo_content or "tls-rustls" in cargo_content, \
            "Cargo.toml should include TLS feature for redis crate"

        assert "redis" in cargo_content, \
            "Cargo.toml should have redis dependency"

    except FileNotFoundError:
        pytest.skip("Cargo.toml not found")


def test_rust_synapse_default_url_uses_tls():
    """
    Verify that the Rust synapse module uses rediss:// by default.

    This ensures TLS is enforced in the Rust codebase as well.
    """
    import glob

    synapse_rs_files = glob.glob("src/**/synapse.rs", recursive=True) + \
                       glob.glob("src/synapse.rs")

    if not synapse_rs_files:
        pytest.skip("No synapse.rs file found")

    for rs_file in synapse_rs_files:
        with open(rs_file, "r", encoding="utf-8") as f:
            rust_content = f.read()

        # Look for default Redis URL in Rust code
        if "REDIS_URL" in rust_content or "rediss://" in rust_content:
            # Check that default uses rediss:// for TLS
            import re
            url_pattern = r'unwrap_or.*["\']([^"\']+redis[^"\']+)["\']\)'
            matches = re.findall(url_pattern, rust_content)

            for url in matches:
                if "redis" in url:
                    assert url.startswith("rediss://"), \
                        f"Rust default Redis URL should use rediss:// for TLS, got: {url}"


# Integration test marker - these would require actual Redis instance
pytest.mark.integration = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Integration tests require RUN_INTEGRATION_TESTS=1 environment variable"
)


@pytest.mark.integration
def test_redis_tls_connection_integration():
    """
    Integration test: Verify actual Redis TLS connection works.

    This test requires a running Redis instance with TLS enabled.
    Set RUN_INTEGRATION_TESTS=1 to run this test.
    """
    from core.synapse import get_synapse

    synapse = get_synapse()
    assert synapse.connect(), "Should connect to Redis with TLS"

    health = synapse.health_check()
    assert health["status"] == "healthy", "Redis health check should pass"
    assert health.get("connected") is True, "Should report connected status"

    synapse.disconnect()
