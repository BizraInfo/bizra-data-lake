# BIZRA Metrics Dashboard Tests
# Unit tests for operational metrics and SNR monitoring

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetricsCollector:
    """Test suite for metrics collection"""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create metrics collector with temp directory"""
        import metrics_dashboard
        original_root = metrics_dashboard.METRICS_ROOT
        metrics_dashboard.METRICS_ROOT = tmp_path / "metrics"
        metrics_dashboard.METRICS_ROOT.mkdir(parents=True, exist_ok=True)

        from metrics_dashboard import MetricsCollector
        collector = MetricsCollector()

        yield collector

        # Restore original path
        metrics_dashboard.METRICS_ROOT = original_root

    def test_collector_initialization(self, collector):
        """Test collector initializes correctly"""
        assert collector is not None
        assert collector.metrics_dir.exists()

    def test_record_snr(self, collector):
        """Test SNR recording"""
        from metrics_dashboard import SNRMetrics

        metrics = SNRMetrics(
            overall_snr=0.95,
            signal_strength=0.92,
            information_density=0.88,
            symbolic_grounding=0.90,
            coverage_balance=0.85,
            ihsan_achieved=False,
            timestamp=datetime.now().isoformat()
        )

        collector.record_snr(metrics)

        # Check it was recorded
        assert len(collector._snr_buffer) == 1
        assert collector._snr_buffer[0]["value"] == 0.95

    def test_record_latency(self, collector):
        """Test latency recording"""
        collector.record_latency(150.5, "test_operation")

        assert len(collector._latency_buffer) == 1
        assert collector._latency_buffer[0]["value"] == 150.5

    def test_record_error(self, collector):
        """Test error recording"""
        collector.record_error("TestError", "Test message", recoverable=True)

        assert len(collector._error_buffer) == 1
        assert collector._error_buffer[0]["context"]["error_type"] == "TestError"

    def test_snr_statistics(self, collector):
        """Test SNR statistics calculation"""
        from metrics_dashboard import SNRMetrics

        # Record multiple SNR values
        for snr_value in [0.92, 0.95, 0.97, 0.99, 0.88]:
            metrics = SNRMetrics(
                overall_snr=snr_value,
                signal_strength=0.9,
                information_density=0.9,
                symbolic_grounding=0.9,
                coverage_balance=0.9,
                ihsan_achieved=snr_value >= 0.99,
                timestamp=datetime.now().isoformat()
            )
            collector.record_snr(metrics)

        stats = collector.get_snr_statistics()

        assert stats["count"] == 5
        assert abs(stats["average"] - 0.942) < 0.01
        assert stats["min"] == 0.88
        assert stats["max"] == 0.99

    def test_latency_statistics(self, collector):
        """Test latency statistics calculation"""
        latencies = [100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]

        for lat in latencies:
            collector.record_latency(lat, "test")

        stats = collector.get_latency_statistics()

        assert stats["count"] == 10
        assert stats["min"] == 100
        assert stats["max"] == 1000
        # P95 should be around 500
        assert 400 <= stats["p95_ms"] <= 1000

    def test_error_rate_calculation(self, collector):
        """Test error rate calculation"""
        from metrics_dashboard import SNRMetrics

        # Record 10 successful operations
        for _ in range(10):
            metrics = SNRMetrics(
                overall_snr=0.95,
                signal_strength=0.9,
                information_density=0.9,
                symbolic_grounding=0.9,
                coverage_balance=0.9,
                ihsan_achieved=False,
                timestamp=datetime.now().isoformat()
            )
            collector.record_snr(metrics)

        # Record 2 errors
        collector.record_error("Error1", "msg", True)
        collector.record_error("Error2", "msg", True)

        error_rate = collector.get_error_rate()

        # 2 errors out of 12 total = ~0.167
        assert abs(error_rate - 0.167) < 0.02


class TestMetricsDashboard:
    """Test suite for dashboard interface"""

    @pytest.fixture
    def dashboard(self, tmp_path):
        """Create dashboard with temp directory"""
        import metrics_dashboard
        original_root = metrics_dashboard.METRICS_ROOT
        metrics_dashboard.METRICS_ROOT = tmp_path / "metrics"
        metrics_dashboard.METRICS_ROOT.mkdir(parents=True, exist_ok=True)

        from metrics_dashboard import MetricsDashboard
        dashboard = MetricsDashboard()

        yield dashboard

        metrics_dashboard.METRICS_ROOT = original_root

    def test_dashboard_initialization(self, dashboard):
        """Test dashboard initializes correctly"""
        assert dashboard is not None
        assert dashboard.collector is not None

    def test_get_system_health_empty(self, dashboard):
        """Test system health with no data"""
        health = dashboard.get_system_health()

        assert health.status in ["healthy", "degraded", "critical"]
        assert health.last_check is not None

    def test_get_system_health_healthy(self, dashboard):
        """Test system health with good metrics"""
        from metrics_dashboard import SNRMetrics

        # Record good SNR values
        for _ in range(10):
            metrics = SNRMetrics(
                overall_snr=0.99,
                signal_strength=0.98,
                information_density=0.97,
                symbolic_grounding=0.96,
                coverage_balance=0.95,
                ihsan_achieved=True,
                timestamp=datetime.now().isoformat()
            )
            dashboard.collector.record_snr(metrics)

        health = dashboard.get_system_health()

        assert health.status == "healthy"
        assert health.snr_average >= 0.95

    def test_get_system_health_degraded(self, dashboard):
        """Test system health with poor metrics"""
        from metrics_dashboard import SNRMetrics

        # Record poor SNR values
        for _ in range(5):
            metrics = SNRMetrics(
                overall_snr=0.80,
                signal_strength=0.75,
                information_density=0.78,
                symbolic_grounding=0.77,
                coverage_balance=0.79,
                ihsan_achieved=False,
                timestamp=datetime.now().isoformat()
            )
            dashboard.collector.record_snr(metrics)

        health = dashboard.get_system_health()

        assert health.status in ["degraded", "critical"]
        assert health.snr_average < 0.95

    def test_export_report(self, dashboard, tmp_path):
        """Test report export"""
        from metrics_dashboard import SNRMetrics

        # Record some data
        for i in range(5):
            metrics = SNRMetrics(
                overall_snr=0.9 + i * 0.02,
                signal_strength=0.9,
                information_density=0.9,
                symbolic_grounding=0.9,
                coverage_balance=0.9,
                ihsan_achieved=False,
                timestamp=datetime.now().isoformat()
            )
            dashboard.collector.record_snr(metrics)

        output_path = tmp_path / "report.json"
        report = dashboard.export_report(output_path)

        assert output_path.exists()
        assert "generated_at" in report
        assert "system_health" in report
        assert "snr_statistics" in report


class TestAlerts:
    """Test suite for alert system"""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create metrics collector with temp directory"""
        import metrics_dashboard
        original_root = metrics_dashboard.METRICS_ROOT
        metrics_dashboard.METRICS_ROOT = tmp_path / "metrics"
        metrics_dashboard.METRICS_ROOT.mkdir(parents=True, exist_ok=True)

        from metrics_dashboard import MetricsCollector
        collector = MetricsCollector()

        yield collector

        metrics_dashboard.METRICS_ROOT = original_root

    def test_snr_below_threshold_triggers_alert(self, collector):
        """Test alert triggered when SNR below threshold"""
        from metrics_dashboard import SNRMetrics

        metrics = SNRMetrics(
            overall_snr=0.85,  # Below acceptable threshold (0.95)
            signal_strength=0.80,
            information_density=0.82,
            symbolic_grounding=0.81,
            coverage_balance=0.80,
            ihsan_achieved=False,
            timestamp=datetime.now().isoformat()
        )

        collector.record_snr(metrics)

        alerts = collector.get_active_alerts()
        assert len(alerts) > 0
        assert any(a.level == "critical" for a in alerts)

    def test_high_latency_triggers_alert(self, collector):
        """Test alert triggered on high latency"""
        collector.record_latency(6000, "slow_operation")  # 6 seconds

        alerts = collector.get_active_alerts()
        assert len(alerts) > 0
        assert any("latency" in a.message.lower() for a in alerts)

    def test_non_recoverable_error_triggers_critical_alert(self, collector):
        """Test critical alert for non-recoverable error"""
        collector.record_error("CriticalError", "System failure", recoverable=False)

        alerts = collector.get_active_alerts()
        assert len(alerts) > 0
        assert any(a.level == "critical" for a in alerts)


class TestIhsanCompliance:
    """Test suite for Ihsān compliance tracking"""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create metrics collector"""
        import metrics_dashboard
        original_root = metrics_dashboard.METRICS_ROOT
        metrics_dashboard.METRICS_ROOT = tmp_path / "metrics"
        metrics_dashboard.METRICS_ROOT.mkdir(parents=True, exist_ok=True)

        from metrics_dashboard import MetricsCollector
        collector = MetricsCollector()

        yield collector

        metrics_dashboard.METRICS_ROOT = original_root

    def test_ihsan_compliance_rate(self, collector):
        """Test Ihsān compliance rate calculation"""
        from metrics_dashboard import SNRMetrics

        # Record 7 Ihsān-compliant and 3 non-compliant
        for i in range(10):
            snr = 0.99 if i < 7 else 0.90
            metrics = SNRMetrics(
                overall_snr=snr,
                signal_strength=0.9,
                information_density=0.9,
                symbolic_grounding=0.9,
                coverage_balance=0.9,
                ihsan_achieved=snr >= 0.99,
                timestamp=datetime.now().isoformat()
            )
            collector.record_snr(metrics)

        stats = collector.get_snr_statistics()

        # 7 out of 10 = 0.70 compliance rate
        assert abs(stats["ihsan_compliance_rate"] - 0.70) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
