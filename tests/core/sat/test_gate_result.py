"""
Tests for SAT-5 GateResult data model.

Standing on Giants:
- Shannon (1948): Information measurement
- Dijkstra (1972): Verification over testing
"""

from core.sat.gate_result import CheckResult, CheckStatus, GateResult


class TestCheckStatus:
    def test_all_statuses_exist(self):
        assert len(CheckStatus) == 5

    def test_values(self):
        assert CheckStatus.PASS.value == "PASS"
        assert CheckStatus.FAIL.value == "FAIL"
        assert CheckStatus.PARTIAL.value == "PARTIAL"
        assert CheckStatus.NOT_IMPLEMENTED.value == "NOT_IMPLEMENTED"
        assert CheckStatus.SKIPPED.value == "SKIPPED"


class TestCheckResult:
    def test_pass_check(self):
        c = CheckResult("test_check", CheckStatus.PASS, "all good")
        assert c.passed is True
        assert c.name == "test_check"
        assert c.evidence == "all good"
        assert c.is_manual is False

    def test_fail_check(self):
        c = CheckResult("bad_check", CheckStatus.FAIL, "broken")
        assert c.passed is False

    def test_manual_check(self):
        c = CheckResult("mother_test", CheckStatus.SKIPPED, "skipped", is_manual=True)
        assert c.is_manual is True
        assert c.passed is False

    def test_to_dict(self):
        c = CheckResult("x", CheckStatus.PASS, "evidence")
        d = c.to_dict()
        assert d["name"] == "x"
        assert d["status"] == "PASS"
        assert d["passed"] is True
        assert d["evidence"] == "evidence"
        assert d["is_manual"] is False

    def test_partial_not_passed(self):
        c = CheckResult("x", CheckStatus.PARTIAL)
        assert c.passed is False

    def test_not_impl_not_passed(self):
        c = CheckResult("x", CheckStatus.NOT_IMPLEMENTED)
        assert c.passed is False


class TestGateResult:
    def test_empty_gate_passes(self):
        g = GateResult(agent="Test", layer="TEST")
        assert g.passed is True
        assert g.verdict == CheckStatus.PASS

    def test_all_pass(self):
        g = GateResult(
            agent="Sentinel",
            layer="STRUCTURAL_INTEGRITY",
            checks=[
                CheckResult("a", CheckStatus.PASS),
                CheckResult("b", CheckStatus.PASS),
            ],
        )
        assert g.passed is True
        assert len(g.failed) == 0

    def test_one_fail_blocks(self):
        g = GateResult(
            agent="Sentinel",
            layer="STRUCTURAL_INTEGRITY",
            checks=[
                CheckResult("a", CheckStatus.PASS),
                CheckResult("b", CheckStatus.FAIL, "broken"),
                CheckResult("c", CheckStatus.PASS),
            ],
        )
        assert g.passed is False
        assert g.verdict == CheckStatus.FAIL
        assert len(g.failed) == 1
        assert g.failed[0].name == "b"

    def test_skipped_does_not_block(self):
        g = GateResult(
            agent="Oracle-S",
            layer="CONSTITUTIONAL_COMPLIANCE",
            checks=[
                CheckResult("a", CheckStatus.PASS),
                CheckResult("manual", CheckStatus.SKIPPED, is_manual=True),
            ],
        )
        assert g.passed is True

    def test_partial_does_not_block(self):
        g = GateResult(
            agent="Conductor",
            layer="OPERATIONAL_READINESS",
            checks=[
                CheckResult("a", CheckStatus.PASS),
                CheckResult("b", CheckStatus.PARTIAL),
            ],
        )
        assert g.passed is True

    def test_not_impl_does_not_block(self):
        g = GateResult(
            agent="Conductor",
            layer="OPERATIONAL_READINESS",
            checks=[
                CheckResult("a", CheckStatus.PASS),
                CheckResult("b", CheckStatus.NOT_IMPLEMENTED),
            ],
        )
        assert g.passed is True

    def test_stats(self):
        g = GateResult(
            agent="Test",
            layer="TEST",
            checks=[
                CheckResult("a", CheckStatus.PASS),
                CheckResult("b", CheckStatus.PASS),
                CheckResult("c", CheckStatus.FAIL),
                CheckResult("d", CheckStatus.PARTIAL),
                CheckResult("e", CheckStatus.NOT_IMPLEMENTED),
                CheckResult("f", CheckStatus.SKIPPED),
            ],
        )
        s = g.stats
        assert s["pass"] == 2
        assert s["fail"] == 1
        assert s["partial"] == 1
        assert s["not_impl"] == 1
        assert s["skipped"] == 1

    def test_to_dict(self):
        g = GateResult(
            agent="Sentinel",
            layer="STRUCTURAL_INTEGRITY",
            checks=[CheckResult("a", CheckStatus.PASS, "ok")],
        )
        d = g.to_dict()
        assert d["agent"] == "Sentinel"
        assert d["layer"] == "STRUCTURAL_INTEGRITY"
        assert d["passed"] is True
        assert d["verdict"] == "PASS"
        assert len(d["checks"]) == 1
        assert d["checks"][0]["name"] == "a"

    def test_to_dict_with_stats(self):
        g = GateResult(
            agent="Test",
            layer="TEST",
            checks=[
                CheckResult("a", CheckStatus.PASS),
                CheckResult("b", CheckStatus.FAIL),
            ],
        )
        d = g.to_dict()
        assert d["stats"]["pass"] == 1
        assert d["stats"]["fail"] == 1
