"""
Tests for SP-010 (SLSA Provenance + SBOM Signing).

Proves that the SBOM generator produces valid CycloneDX SBOMs,
SLSA provenance attestations, and Ed25519-signed DSSE envelopes.

Standing on Giants:
- SLSA (Google/OpenSSF): Supply chain security
- CycloneDX (OWASP): SBOM standard
- in-toto (NYU, 2019): Attestation framework
- Bernstein (2011): Ed25519 signatures
- BIZRA Spearpoint PRD SP-010
"""

import hashlib
import json
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_sbom import (
    generate_cyclonedx,
    generate_slsa_provenance,
)


# =============================================================================
# CYCLONEDX SBOM
# =============================================================================

class TestCycloneDXGeneration:
    """Tests for CycloneDX SBOM generation."""

    def test_sbom_basic_structure(self):
        """SBOM has required CycloneDX 1.5 fields."""
        sbom = generate_cyclonedx([], [])
        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["specVersion"] == "1.5"
        assert "serialNumber" in sbom
        assert "metadata" in sbom
        assert "components" in sbom

    def test_sbom_metadata_structure(self):
        """SBOM metadata includes timestamp, component, tools."""
        sbom = generate_cyclonedx([], [])
        meta = sbom["metadata"]
        assert "timestamp" in meta
        assert meta["component"]["type"] == "application"
        assert meta["component"]["name"] == "bizra-data-lake"
        assert len(meta["tools"]) >= 1

    def test_sbom_python_packages(self):
        """SBOM includes Python packages with purl."""
        py_pkgs = [
            {"name": "requests", "version": "2.31.0"},
            {"name": "numpy", "version": "1.26.0"},
        ]
        sbom = generate_cyclonedx(py_pkgs, [])
        assert len(sbom["components"]) == 2
        assert sbom["components"][0]["purl"] == "pkg:pypi/requests@2.31.0"
        assert sbom["components"][1]["name"] == "numpy"

    def test_sbom_rust_packages(self):
        """SBOM includes Rust packages with purl."""
        rust_pkgs = [
            {"name": "serde", "version": "1.0.200"},
            {"name": "tokio", "version": "1.38.0"},
        ]
        sbom = generate_cyclonedx([], rust_pkgs)
        assert len(sbom["components"]) == 2
        assert sbom["components"][0]["purl"] == "pkg:cargo/serde@1.0.200"

    def test_sbom_combined_packages(self):
        """SBOM combines Python + Rust packages."""
        py_pkgs = [{"name": "flask", "version": "3.0.0"}]
        rust_pkgs = [{"name": "hyper", "version": "1.3.0"}]
        sbom = generate_cyclonedx(py_pkgs, rust_pkgs)
        assert len(sbom["components"]) == 2

    def test_sbom_custom_project_name(self):
        """SBOM uses custom project name."""
        sbom = generate_cyclonedx([], [], project_name="bizra-omega", project_version="2.0.0")
        assert sbom["metadata"]["component"]["name"] == "bizra-omega"
        assert sbom["metadata"]["component"]["version"] == "2.0.0"

    def test_sbom_serial_number_deterministic(self):
        """Same components produce same serial number."""
        pkgs = [{"name": "a", "version": "1.0"}]
        s1 = generate_cyclonedx(pkgs, [])
        s2 = generate_cyclonedx(pkgs, [])
        assert s1["serialNumber"] == s2["serialNumber"]

    def test_sbom_is_json_serializable(self):
        """SBOM is fully JSON serializable."""
        sbom = generate_cyclonedx(
            [{"name": "x", "version": "1.0"}],
            [{"name": "y", "version": "2.0"}],
        )
        serialized = json.dumps(sbom)
        deserialized = json.loads(serialized)
        assert deserialized["bomFormat"] == "CycloneDX"


# =============================================================================
# SLSA PROVENANCE
# =============================================================================

class TestSLSAProvenance:
    """Tests for SLSA provenance attestation generation."""

    def _make_sbom_and_hash(self):
        """Helper to create SBOM and its hash."""
        sbom = generate_cyclonedx(
            [{"name": "test", "version": "1.0"}], []
        )
        sbom_json = json.dumps(sbom, indent=2).encode("utf-8")
        sbom_hash = hashlib.sha256(sbom_json).hexdigest()
        return sbom, sbom_hash

    def test_provenance_in_toto_statement(self):
        """Provenance is a valid in-toto Statement v1."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        prov = generate_slsa_provenance(sbom, sbom_hash)
        assert prov["_type"] == "https://in-toto.io/Statement/v1"

    def test_provenance_predicate_type(self):
        """Provenance has SLSA v1 predicate type."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        prov = generate_slsa_provenance(sbom, sbom_hash)
        assert prov["predicateType"] == "https://slsa.dev/provenance/v1"

    def test_provenance_subject(self):
        """Subject contains SBOM name and digest."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        prov = generate_slsa_provenance(sbom, sbom_hash)
        subjects = prov["subject"]
        assert len(subjects) == 1
        assert subjects[0]["name"] == "bizra-sbom.cdx.json"
        assert subjects[0]["digest"]["sha256"] == sbom_hash

    def test_provenance_build_definition(self):
        """Build definition includes source, parameters, dependencies."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        prov = generate_slsa_provenance(
            sbom, sbom_hash,
            git_commit="abc123",
            git_repo="https://github.com/bizra/node0",
        )
        build_def = prov["predicate"]["buildDefinition"]
        assert build_def["buildType"] == "https://bizra.ai/build/v1"
        assert build_def["externalParameters"]["source"]["digest"]["gitCommit"] == "abc123"
        assert build_def["externalParameters"]["source"]["uri"] == "https://github.com/bizra/node0"

    def test_provenance_internal_parameters(self):
        """Internal parameters include Python version and builder ID."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        prov = generate_slsa_provenance(sbom, sbom_hash, builder_id="test-builder")
        params = prov["predicate"]["buildDefinition"]["internalParameters"]
        assert params["builder_id"] == "test-builder"
        assert "python_version" in params

    def test_provenance_resolved_dependencies(self):
        """Resolved dependencies reflect component count."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        prov = generate_slsa_provenance(sbom, sbom_hash)
        deps = prov["predicate"]["buildDefinition"]["resolvedDependencies"]
        assert deps[0]["count"] == 1  # One test component

    def test_provenance_run_details(self):
        """Run details include builder and metadata."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        prov = generate_slsa_provenance(sbom, sbom_hash)
        run = prov["predicate"]["runDetails"]
        assert run["builder"]["id"] == "bizra-node0-builder"
        assert "invocationId" in run["metadata"]
        assert "startedOn" in run["metadata"]

    def test_provenance_is_json_serializable(self):
        """Provenance is fully JSON serializable."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        prov = generate_slsa_provenance(sbom, sbom_hash)
        serialized = json.dumps(prov)
        deserialized = json.loads(serialized)
        assert deserialized["_type"] == "https://in-toto.io/Statement/v1"

    def test_provenance_invocation_id_unique(self):
        """Different timestamps produce different invocation IDs."""
        sbom, sbom_hash = self._make_sbom_and_hash()
        p1 = generate_slsa_provenance(sbom, sbom_hash, git_commit="aaa")
        p2 = generate_slsa_provenance(sbom, sbom_hash, git_commit="bbb")
        id1 = p1["predicate"]["runDetails"]["metadata"]["invocationId"]
        id2 = p2["predicate"]["runDetails"]["metadata"]["invocationId"]
        assert id1 != id2


# =============================================================================
# ED25519 SIGNING (DSSE)
# =============================================================================

class TestProvenanceSigning:
    """Tests for DSSE envelope signing of provenance."""

    def test_sign_provenance_creates_dsse_envelope(self):
        """sign_provenance returns a valid DSSE envelope."""
        from tools.generate_sbom import sign_provenance
        from core.pci.crypto import generate_keypair

        sbom = generate_cyclonedx([], [])
        sbom_hash = hashlib.sha256(json.dumps(sbom).encode()).hexdigest()
        prov = generate_slsa_provenance(sbom, sbom_hash)

        priv_hex, pub_hex = generate_keypair()
        envelope = sign_provenance(prov, priv_hex, pub_hex)

        assert envelope["payloadType"] == "application/vnd.in-toto+json"
        assert "payload" in envelope
        assert len(envelope["signatures"]) == 1
        assert "keyid" in envelope["signatures"][0]
        assert "sig" in envelope["signatures"][0]

    def test_dsse_payload_is_base64(self):
        """DSSE payload is valid base64-encoded JSON."""
        import base64

        from tools.generate_sbom import sign_provenance
        from core.pci.crypto import generate_keypair

        sbom = generate_cyclonedx([], [])
        sbom_hash = hashlib.sha256(json.dumps(sbom).encode()).hexdigest()
        prov = generate_slsa_provenance(sbom, sbom_hash)

        priv_hex, pub_hex = generate_keypair()
        envelope = sign_provenance(prov, priv_hex, pub_hex)

        decoded = base64.b64decode(envelope["payload"]).decode("utf-8")
        parsed = json.loads(decoded)
        assert parsed["_type"] == "https://in-toto.io/Statement/v1"

    def test_dsse_signature_verifiable(self):
        """DSSE signature can be verified with the public key."""
        from tools.generate_sbom import sign_provenance
        from core.pci.crypto import generate_keypair, verify_signature

        sbom = generate_cyclonedx([], [])
        sbom_hash = hashlib.sha256(json.dumps(sbom).encode()).hexdigest()
        prov = generate_slsa_provenance(sbom, sbom_hash)

        priv_hex, pub_hex = generate_keypair()
        envelope = sign_provenance(prov, priv_hex, pub_hex)

        # Reconstruct the payload digest that was signed
        payload = json.dumps(prov, sort_keys=True, separators=(",", ":"))
        payload_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        sig = envelope["signatures"][0]["sig"]

        assert verify_signature(payload_digest, sig, pub_hex)

    def test_dsse_keyid_derived_from_pubkey(self):
        """Key ID is derived from public key hash."""
        from tools.generate_sbom import sign_provenance
        from core.pci.crypto import generate_keypair

        sbom = generate_cyclonedx([], [])
        sbom_hash = hashlib.sha256(json.dumps(sbom).encode()).hexdigest()
        prov = generate_slsa_provenance(sbom, sbom_hash)

        priv_hex, pub_hex = generate_keypair()
        envelope = sign_provenance(prov, priv_hex, pub_hex)

        expected_keyid = hashlib.sha256(pub_hex.encode()).hexdigest()[:16]
        assert envelope["signatures"][0]["keyid"] == expected_keyid

    def test_dsse_envelope_is_json_serializable(self):
        """DSSE envelope is fully JSON serializable."""
        from tools.generate_sbom import sign_provenance
        from core.pci.crypto import generate_keypair

        sbom = generate_cyclonedx([], [])
        sbom_hash = hashlib.sha256(json.dumps(sbom).encode()).hexdigest()
        prov = generate_slsa_provenance(sbom, sbom_hash)

        priv_hex, pub_hex = generate_keypair()
        envelope = sign_provenance(prov, priv_hex, pub_hex)

        serialized = json.dumps(envelope)
        deserialized = json.loads(serialized)
        assert deserialized["payloadType"] == "application/vnd.in-toto+json"
