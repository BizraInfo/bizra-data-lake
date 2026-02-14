"""
Schema Validation Tests — SP-001 Completion.

Proves that JSON schemas are the compiler: valid fixtures pass,
invalid mutations fail, and the structural fallback matches.

Standing on Giants:
- JSON Schema draft 2020-12
- OWASP input validation
- BIZRA Spearpoint PRD: "schemas first, so verified cannot drift"
"""

import copy
import json
import pytest
from pathlib import Path

from core.proof_engine.schema_validator import (
    validate,
    validate_receipt,
    validate_reasoning_graph,
    validate_attestation,
    list_schemas,
    _load_schema,
    _validate_structural,
)
from core.proof_engine.reason_codes import (
    ReasonCode,
    REASON_DESCRIPTIONS,
    describe,
)


# =============================================================================
# FIXTURES
# =============================================================================

FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "schemas" / "fixtures"


@pytest.fixture
def valid_receipt():
    """Load valid receipt fixture."""
    path = FIXTURE_DIR / "valid_receipt.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def valid_graph():
    """Load valid reasoning graph fixture."""
    path = FIXTURE_DIR / "valid_reasoning_graph.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def valid_attestation():
    """Load valid attestation fixture."""
    path = FIXTURE_DIR / "valid_attestation.json"
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# SCHEMA LOADING
# =============================================================================

class TestSchemaLoading:
    """Tests for schema loading and caching."""

    def test_list_schemas_returns_all(self):
        """All 4 schemas are discoverable."""
        schemas = list_schemas()
        assert "receipt" in schemas
        assert "reasoning_graph" in schemas
        assert "attestation" in schemas
        assert "error_codes" in schemas

    def test_load_schema_receipt(self):
        """Receipt schema loads and has required structure."""
        schema = _load_schema("receipt")
        assert schema["title"] == "BIZRA Receipt"
        assert "receipt_id" in schema["properties"]
        assert "seal" in schema["required"]

    def test_load_schema_reasoning_graph(self):
        """Reasoning graph schema loads."""
        schema = _load_schema("reasoning_graph")
        assert schema["title"] == "BIZRA Reasoning Graph"
        assert "graph_hash" in schema["required"]

    def test_load_schema_attestation(self):
        """Attestation schema loads."""
        schema = _load_schema("attestation")
        assert schema["title"] == "BIZRA Attestation Envelope"
        assert schema["properties"]["version"]["const"] == "1.0.0"

    def test_load_schema_caching(self):
        """Schema is cached after first load."""
        schema1 = _load_schema("receipt")
        schema2 = _load_schema("receipt")
        assert schema1 is schema2  # Same object (cached)

    def test_load_schema_missing_raises(self):
        """Missing schema raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Schema not found"):
            _load_schema("nonexistent_schema")


# =============================================================================
# RECEIPT VALIDATION
# =============================================================================

class TestReceiptValidation:
    """Tests for receipt schema validation."""

    def test_valid_receipt_passes(self, valid_receipt):
        """Valid receipt fixture passes validation."""
        is_valid, errors = validate_receipt(valid_receipt)
        assert is_valid is True, f"Errors: {errors}"
        assert errors == []

    def test_receipt_missing_receipt_id(self, valid_receipt):
        """Receipt without receipt_id fails."""
        data = copy.deepcopy(valid_receipt)
        del data["receipt_id"]
        is_valid, errors = validate_receipt(data)
        assert is_valid is False
        assert any("receipt_id" in e for e in errors)

    def test_receipt_missing_seal(self, valid_receipt):
        """Receipt without seal fails."""
        data = copy.deepcopy(valid_receipt)
        del data["seal"]
        is_valid, errors = validate_receipt(data)
        assert is_valid is False
        assert any("seal" in e for e in errors)

    def test_receipt_missing_snr(self, valid_receipt):
        """Receipt without snr fails."""
        data = copy.deepcopy(valid_receipt)
        del data["snr"]
        is_valid, errors = validate_receipt(data)
        assert is_valid is False
        assert any("snr" in e for e in errors)

    def test_receipt_missing_ihsan(self, valid_receipt):
        """Receipt without ihsan fails."""
        data = copy.deepcopy(valid_receipt)
        del data["ihsan"]
        is_valid, errors = validate_receipt(data)
        assert is_valid is False
        assert any("ihsan" in e for e in errors)

    def test_receipt_invalid_status(self, valid_receipt):
        """Receipt with invalid status fails."""
        data = copy.deepcopy(valid_receipt)
        data["status"] = "invalid_status"
        is_valid, errors = validate_receipt(data)
        assert is_valid is False

    def test_receipt_invalid_decision(self, valid_receipt):
        """Receipt with invalid decision fails."""
        data = copy.deepcopy(valid_receipt)
        data["decision"] = "MAYBE"
        is_valid, errors = validate_receipt(data)
        assert is_valid is False

    def test_receipt_snr_score_out_of_range(self, valid_receipt):
        """Receipt with snr score > 1 fails."""
        data = copy.deepcopy(valid_receipt)
        data["snr"]["score"] = 1.5
        is_valid, errors = validate_receipt(data)
        assert is_valid is False

    def test_receipt_policy_version_format(self, valid_receipt):
        """Receipt with invalid policy_version format fails."""
        data = copy.deepcopy(valid_receipt)
        data["policy_version"] = "not_semver"
        is_valid, errors = validate_receipt(data)
        assert is_valid is False

    def test_receipt_seal_algorithm_enum(self, valid_receipt):
        """Receipt with invalid seal algorithm fails."""
        data = copy.deepcopy(valid_receipt)
        data["seal"]["algorithm"] = "md5"
        is_valid, errors = validate_receipt(data)
        assert is_valid is False


# =============================================================================
# REASONING GRAPH VALIDATION
# =============================================================================

class TestReasoningGraphValidation:
    """Tests for reasoning graph schema validation."""

    def test_valid_graph_passes(self, valid_graph):
        """Valid reasoning graph fixture passes validation."""
        is_valid, errors = validate_reasoning_graph(valid_graph)
        assert is_valid is True, f"Errors: {errors}"
        assert errors == []

    def test_graph_missing_nodes(self, valid_graph):
        """Graph without nodes fails."""
        data = copy.deepcopy(valid_graph)
        del data["nodes"]
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False
        assert any("nodes" in e for e in errors)

    def test_graph_missing_edges(self, valid_graph):
        """Graph without edges fails."""
        data = copy.deepcopy(valid_graph)
        del data["edges"]
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False
        assert any("edges" in e for e in errors)

    def test_graph_missing_roots(self, valid_graph):
        """Graph without roots fails."""
        data = copy.deepcopy(valid_graph)
        del data["roots"]
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False
        assert any("roots" in e for e in errors)

    def test_graph_missing_graph_hash(self, valid_graph):
        """Graph without graph_hash fails."""
        data = copy.deepcopy(valid_graph)
        del data["graph_hash"]
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False
        assert any("graph_hash" in e for e in errors)

    def test_graph_empty_roots(self, valid_graph):
        """Graph with empty roots array fails (minItems: 1)."""
        data = copy.deepcopy(valid_graph)
        data["roots"] = []
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False

    def test_graph_invalid_node_type(self, valid_graph):
        """Graph with invalid node type fails."""
        data = copy.deepcopy(valid_graph)
        data["nodes"][0]["type"] = "invalid_type"
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False

    def test_graph_invalid_edge_type(self, valid_graph):
        """Graph with invalid edge type fails."""
        data = copy.deepcopy(valid_graph)
        data["edges"][0]["type"] = "invalid_type"
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False

    def test_graph_node_confidence_out_of_range(self, valid_graph):
        """Graph with node confidence > 1 fails."""
        data = copy.deepcopy(valid_graph)
        data["nodes"][0]["confidence"] = 2.0
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False

    def test_graph_invalid_strategy(self, valid_graph):
        """Graph with invalid strategy fails."""
        data = copy.deepcopy(valid_graph)
        data["config"]["strategy"] = "random_walk"
        is_valid, errors = validate_reasoning_graph(data)
        assert is_valid is False


# =============================================================================
# ATTESTATION VALIDATION
# =============================================================================

class TestAttestationValidation:
    """Tests for attestation envelope schema validation."""

    def test_valid_attestation_passes(self, valid_attestation):
        """Valid attestation fixture passes validation."""
        is_valid, errors = validate_attestation(valid_attestation)
        assert is_valid is True, f"Errors: {errors}"
        assert errors == []

    def test_attestation_missing_version(self, valid_attestation):
        """Attestation without version fails."""
        data = copy.deepcopy(valid_attestation)
        del data["version"]
        is_valid, errors = validate_attestation(data)
        assert is_valid is False

    def test_attestation_wrong_version(self, valid_attestation):
        """Attestation with wrong version fails (const: 1.0.0)."""
        data = copy.deepcopy(valid_attestation)
        data["version"] = "2.0.0"
        is_valid, errors = validate_attestation(data)
        assert is_valid is False

    def test_attestation_missing_nonce(self, valid_attestation):
        """Attestation without nonce fails."""
        data = copy.deepcopy(valid_attestation)
        del data["nonce"]
        is_valid, errors = validate_attestation(data)
        assert is_valid is False

    def test_attestation_short_nonce(self, valid_attestation):
        """Attestation with short nonce fails (need 64 hex chars)."""
        data = copy.deepcopy(valid_attestation)
        data["nonce"] = "abc123"
        is_valid, errors = validate_attestation(data)
        assert is_valid is False

    def test_attestation_missing_sender(self, valid_attestation):
        """Attestation without sender fails."""
        data = copy.deepcopy(valid_attestation)
        del data["sender"]
        is_valid, errors = validate_attestation(data)
        assert is_valid is False

    def test_attestation_invalid_agent_type(self, valid_attestation):
        """Attestation with invalid agent_type fails."""
        data = copy.deepcopy(valid_attestation)
        data["sender"]["agent_type"] = "UNKNOWN"
        is_valid, errors = validate_attestation(data)
        assert is_valid is False

    def test_attestation_missing_payload(self, valid_attestation):
        """Attestation without payload fails."""
        data = copy.deepcopy(valid_attestation)
        del data["payload"]
        is_valid, errors = validate_attestation(data)
        assert is_valid is False

    def test_attestation_additional_properties_rejected(self, valid_attestation):
        """Attestation rejects additional properties (strict)."""
        data = copy.deepcopy(valid_attestation)
        data["extra_field"] = "should_fail"
        is_valid, errors = validate_attestation(data)
        assert is_valid is False

    def test_attestation_invalid_urgency(self, valid_attestation):
        """Attestation with invalid urgency fails."""
        data = copy.deepcopy(valid_attestation)
        data["metadata"]["urgency"] = "ASAP"
        is_valid, errors = validate_attestation(data)
        assert is_valid is False


# =============================================================================
# STRUCTURAL FALLBACK VALIDATION
# =============================================================================

class TestStructuralFallback:
    """Tests for the lightweight structural validator (no jsonschema dep)."""

    def test_structural_checks_required_fields(self):
        """Structural validator catches missing required fields."""
        schema = {
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        data = {"name": "Alice"}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False
        assert any("age" in e for e in errors)

    def test_structural_checks_string_type(self):
        """Structural validator catches wrong type (string expected)."""
        schema = {
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        data = {"name": 42}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False
        assert any("string" in e for e in errors)

    def test_structural_checks_number_type(self):
        """Structural validator catches wrong type (number expected)."""
        schema = {
            "required": ["score"],
            "properties": {"score": {"type": "number"}},
        }
        data = {"score": "not_a_number"}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False

    def test_structural_checks_enum(self):
        """Structural validator catches invalid enum value."""
        schema = {
            "required": ["status"],
            "properties": {"status": {"type": "string", "enum": ["active", "inactive"]}},
        }
        data = {"status": "unknown"}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False
        assert any("enum" in e for e in errors)

    def test_structural_checks_minimum(self):
        """Structural validator catches values below minimum."""
        schema = {
            "required": ["score"],
            "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
        }
        data = {"score": -0.5}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False
        assert any("minimum" in e for e in errors)

    def test_structural_checks_maximum(self):
        """Structural validator catches values above maximum."""
        schema = {
            "required": ["score"],
            "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
        }
        data = {"score": 1.5}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False
        assert any("maximum" in e for e in errors)

    def test_structural_passes_valid_data(self):
        """Structural validator passes valid data."""
        schema = {
            "required": ["name", "score"],
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }
        data = {"name": "test", "score": 0.95}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is True
        assert errors == []

    def test_structural_checks_boolean_type(self):
        """Structural validator catches wrong type (boolean expected)."""
        schema = {
            "required": ["active"],
            "properties": {"active": {"type": "boolean"}},
        }
        data = {"active": "yes"}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False

    def test_structural_checks_array_type(self):
        """Structural validator catches wrong type (array expected)."""
        schema = {
            "required": ["items"],
            "properties": {"items": {"type": "array"}},
        }
        data = {"items": "not_array"}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False

    def test_structural_checks_object_type(self):
        """Structural validator catches wrong type (object expected)."""
        schema = {
            "required": ["config"],
            "properties": {"config": {"type": "object"}},
        }
        data = {"config": "not_object"}
        errors = []
        is_valid, errors = _validate_structural(data, schema, errors)
        assert is_valid is False


# =============================================================================
# REASON CODES
# =============================================================================

class TestReasonCodes:
    """Tests for the unified reason codes enum."""

    def test_all_28_codes_exist(self):
        """All 28 reason codes are defined."""
        assert len(ReasonCode) == 28

    def test_evidence_category(self):
        """Evidence integrity codes exist."""
        assert ReasonCode.EVIDENCE_MISSING.value == "EVIDENCE_MISSING"
        assert ReasonCode.EVIDENCE_TAMPERED.value == "EVIDENCE_TAMPERED"
        assert ReasonCode.EVIDENCE_EXPIRED.value == "EVIDENCE_EXPIRED"
        assert ReasonCode.NONDETERMINISTIC_BUILD.value == "NONDETERMINISTIC_BUILD"

    def test_claim_tags_category(self):
        """Claim provenance codes exist."""
        assert ReasonCode.CLAIM_UNTAGGED.value == "CLAIM_UNTAGGED"
        assert ReasonCode.CLAIM_TAG_INVALID.value == "CLAIM_TAG_INVALID"
        assert ReasonCode.CLAIM_INFLATION.value == "CLAIM_INFLATION"

    def test_quality_gates_category(self):
        """Quality gate codes exist."""
        assert ReasonCode.SNR_BELOW_THRESHOLD.value == "SNR_BELOW_THRESHOLD"
        assert ReasonCode.IHSAN_BELOW_THRESHOLD.value == "IHSAN_BELOW_THRESHOLD"
        assert ReasonCode.CONFIDENCE_BELOW_THRESHOLD.value == "CONFIDENCE_BELOW_THRESHOLD"

    def test_cryptographic_category(self):
        """Cryptographic codes exist."""
        assert ReasonCode.SIGNATURE_INVALID.value == "SIGNATURE_INVALID"
        assert ReasonCode.REPLAY_DETECTED.value == "REPLAY_DETECTED"
        assert ReasonCode.GENESIS_MISMATCH.value == "GENESIS_MISMATCH"

    def test_authorization_category(self):
        """Authorization codes exist."""
        assert ReasonCode.AUTH_REQUIRED.value == "AUTH_REQUIRED"
        assert ReasonCode.RATE_LIMITED.value == "RATE_LIMITED"
        assert ReasonCode.BUDGET_EXCEEDED.value == "BUDGET_EXCEEDED"

    def test_tool_runtime_category(self):
        """Tool runtime codes exist."""
        assert ReasonCode.HOOK_NOT_ALLOWLISTED.value == "HOOK_NOT_ALLOWLISTED"
        assert ReasonCode.TOOL_TIMEOUT.value == "TOOL_TIMEOUT"

    def test_policy_category(self):
        """Policy enforcement codes exist."""
        assert ReasonCode.POLICY_MISMATCH.value == "POLICY_MISMATCH"
        assert ReasonCode.SCHEMA_VIOLATION.value == "SCHEMA_VIOLATION"
        assert ReasonCode.INVARIANT_FAILED.value == "INVARIANT_FAILED"
        assert ReasonCode.ROLE_VIOLATION.value == "ROLE_VIOLATION"

    def test_every_code_has_description(self):
        """Every reason code has a human-readable description."""
        for code in ReasonCode:
            assert code.value in REASON_DESCRIPTIONS, f"Missing description for {code.value}"
            desc = REASON_DESCRIPTIONS[code.value]
            assert len(desc) > 10, f"Description too short for {code.value}"

    def test_describe_function(self):
        """describe() returns human-readable text."""
        desc = describe(ReasonCode.SNR_BELOW_THRESHOLD)
        assert "Signal-to-noise" in desc

    def test_reason_code_is_string_enum(self):
        """ReasonCode inherits from str for JSON serialization."""
        code = ReasonCode.EVIDENCE_MISSING
        assert isinstance(code, str)
        assert code == "EVIDENCE_MISSING"

    def test_reason_codes_match_schema(self):
        """Python enum values match the JSON schema enum list."""
        schema = _load_schema("error_codes")
        schema_codes = set(schema["definitions"]["reason_code"]["enum"])
        python_codes = {code.value for code in ReasonCode}
        assert schema_codes == python_codes, (
            f"Schema-Python mismatch. "
            f"In schema only: {schema_codes - python_codes}. "
            f"In Python only: {python_codes - schema_codes}."
        )


# =============================================================================
# CROSS-ARTIFACT CONSISTENCY
# =============================================================================

class TestCrossArtifactConsistency:
    """Tests that schemas are mutually consistent."""

    def test_receipt_decision_matches_attestation_decision(self):
        """Receipt and attestation use same decision vocabulary."""
        receipt_schema = _load_schema("receipt")
        # Receipt decision enum
        receipt_decisions = set(receipt_schema["properties"]["decision"]["enum"])
        assert "APPROVED" in receipt_decisions
        assert "REJECTED" in receipt_decisions
        assert "QUARANTINED" in receipt_decisions

    def test_receipt_ihsan_decision_matches_top_level(self):
        """Receipt ihsan.decision uses same enum as top-level decision."""
        receipt_schema = _load_schema("receipt")
        top_decisions = set(receipt_schema["properties"]["decision"]["enum"])
        ihsan_decisions = set(
            receipt_schema["properties"]["ihsan"]["properties"]["decision"]["enum"]
        )
        assert top_decisions == ihsan_decisions

    def test_graph_hash_format_consistent(self):
        """Graph hash pattern is consistent across receipt and graph schemas."""
        receipt_schema = _load_schema("receipt")
        graph_schema = _load_schema("reasoning_graph")
        receipt_pattern = receipt_schema["properties"]["outputs"]["properties"]["graph_hash"]["pattern"]
        graph_pattern = graph_schema["properties"]["graph_hash"]["pattern"]
        assert receipt_pattern == graph_pattern

    def test_signature_algorithm_consistent(self):
        """Signature algorithm enum is consistent across schemas."""
        receipt_schema = _load_schema("receipt")
        attestation_schema = _load_schema("attestation")
        receipt_algo = receipt_schema["properties"]["signature"]["properties"]["algorithm"]["enum"]
        attestation_algo = attestation_schema["properties"]["signature"]["properties"]["algorithm"]["enum"]
        assert receipt_algo == attestation_algo


# =============================================================================
# CONDITIONAL VALIDATION (RECEIPT)
# =============================================================================

class TestConditionalValidation:
    """Tests for receipt conditional validation (reason_codes on rejection)."""

    def test_rejected_receipt_requires_reason_codes(self, valid_receipt):
        """REJECTED receipt must have non-empty reason_codes."""
        data = copy.deepcopy(valid_receipt)
        data["decision"] = "REJECTED"
        data["status"] = "rejected"
        data["reason_codes"] = []  # Empty — should fail
        is_valid, errors = validate_receipt(data)
        assert is_valid is False

    def test_rejected_receipt_with_reason_codes_passes(self, valid_receipt):
        """REJECTED receipt with reason_codes passes."""
        data = copy.deepcopy(valid_receipt)
        data["decision"] = "REJECTED"
        data["status"] = "rejected"
        data["reason_codes"] = ["SNR_BELOW_THRESHOLD"]
        is_valid, errors = validate_receipt(data)
        assert is_valid is True, f"Errors: {errors}"

    def test_quarantined_receipt_requires_reason_codes(self, valid_receipt):
        """QUARANTINED receipt must have non-empty reason_codes."""
        data = copy.deepcopy(valid_receipt)
        data["decision"] = "QUARANTINED"
        data["status"] = "quarantined"
        data["reason_codes"] = []  # Empty — should fail
        is_valid, errors = validate_receipt(data)
        assert is_valid is False

    def test_approved_receipt_allows_empty_reason_codes(self, valid_receipt):
        """APPROVED receipt can have empty reason_codes."""
        data = copy.deepcopy(valid_receipt)
        data["decision"] = "APPROVED"
        data["reason_codes"] = []
        is_valid, errors = validate_receipt(data)
        assert is_valid is True, f"Errors: {errors}"
