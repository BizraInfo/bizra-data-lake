"""Tests for core.proactive.infra_health — InfraHealthProbe.

Covers:
- Probe construction with and without guardian
- check() returns report or fallback when guardian unavailable
- check_and_fix() handles absent guardian gracefully
- ihsan_score() returns 0.0 when guardian unavailable
- summary() returns string with status info
- available property reflects guardian presence

Blueprint Reference: P3 Coverage Ratchet — proactive module (0% → tested)
"""

from unittest.mock import MagicMock, patch


from core.proactive.infra_health import InfraHealthProbe


class TestInfraHealthProbeNoGuardian:
    """Tests when infra_guardian is NOT available."""

    @patch("core.proactive.infra_health._import_guardian", return_value=None)
    def test_available_false(self, _mock):
        probe = InfraHealthProbe()
        assert probe.available is False

    @patch("core.proactive.infra_health._import_guardian", return_value=None)
    def test_check_returns_unknown(self, _mock):
        probe = InfraHealthProbe()
        report = probe.check()
        assert report["overall"] == "UNKNOWN"
        assert report["ihsan"] == 0.0
        assert "error" in report

    @patch("core.proactive.infra_health._import_guardian", return_value=None)
    def test_check_and_fix_returns_unknown(self, _mock):
        probe = InfraHealthProbe()
        report = probe.check_and_fix()
        assert report["overall"] == "UNKNOWN"
        assert report["ihsan"] == 0.0

    @patch("core.proactive.infra_health._import_guardian", return_value=None)
    def test_ihsan_score_zero(self, _mock):
        probe = InfraHealthProbe()
        assert probe.ihsan_score() == 0.0

    @patch("core.proactive.infra_health._import_guardian", return_value=None)
    def test_summary_shows_unavailable(self, _mock):
        probe = InfraHealthProbe()
        s = probe.summary()
        assert "unknown" in s.lower() or "unavailable" in s.lower()


class TestInfraHealthProbeWithGuardian:
    """Tests with a mocked guardian module."""

    def _make_mock_guardian(self):
        guardian = MagicMock()
        guardian.Severity.OK = "OK"
        guardian.Severity.FIXED = "FIXED"
        guardian.Severity.WARNING = "WARNING"

        mock_state = MagicMock()
        guardian.GuardianState.return_value = mock_state

        mock_result = MagicMock()
        mock_result.severity = "OK"

        guardian.run_all_probes.return_value = [mock_result]
        guardian.results_to_report.return_value = {
            "overall": "HEALTHY",
            "ihsan": 0.97,
            "summary": {"ok": 5, "warnings": 0, "critical": 0, "auto_fixed": 0},
        }
        return guardian

    @patch("core.proactive.infra_health._import_guardian")
    def test_available_true(self, mock_import):
        mock_import.return_value = self._make_mock_guardian()
        probe = InfraHealthProbe()
        assert probe.available is True

    @patch("core.proactive.infra_health._import_guardian")
    def test_check_returns_report(self, mock_import):
        mock_import.return_value = self._make_mock_guardian()
        probe = InfraHealthProbe()
        report = probe.check()
        assert report["overall"] == "HEALTHY"
        assert report["ihsan"] == 0.97

    @patch("core.proactive.infra_health._import_guardian")
    def test_ihsan_score_from_report(self, mock_import):
        mock_import.return_value = self._make_mock_guardian()
        probe = InfraHealthProbe()
        score = probe.ihsan_score()
        assert score == 0.97

    @patch("core.proactive.infra_health._import_guardian")
    def test_summary_shows_healthy(self, mock_import):
        mock_import.return_value = self._make_mock_guardian()
        probe = InfraHealthProbe()
        s = probe.summary()
        assert "HEALTHY" in s
        assert "0.970" in s

    @patch("core.proactive.infra_health._import_guardian")
    def test_check_and_fix_calls_correct(self, mock_import):
        guardian = self._make_mock_guardian()
        mock_import.return_value = guardian
        probe = InfraHealthProbe()
        probe.check_and_fix()
        guardian.run_all_probes.assert_called_once()
        call_kwargs = guardian.run_all_probes.call_args
        assert (
            call_kwargs[1].get("correct", call_kwargs[0][0] if call_kwargs[0] else None)
            is not None
        )
