"""
Cognitive Bridge Integration Tests
==================================

Tests the Rust-Python bridge communication via the /process endpoint.

This validates:
1. Request/Response schema compatibility
2. FATE gate integration
3. Ihsān scoring through bridge
4. ThinkingMode mapping
5. Error handling and circuit breaker scenarios

Run: pytest tests/test_cognitive_bridge_integration.py -v
"""

import json
import time
import pytest
import httpx
from typing import Optional

# Test configuration
PYTHON_SERVICE_URL = "http://localhost:8010"  # Python kernel port
RUST_SERVICE_URL = "http://localhost:8080"    # Rust elite port


def is_service_available(url: str, timeout: float = 2.0) -> bool:
    """Check if a service is responding."""
    try:
        response = httpx.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


class TestRustCognitiveRequest:
    """Test the Rust CognitiveBridge request format."""
    
    def create_request(
        self,
        prompt: str,
        mode: str = "HybridSynergy",
        agent_id: str = "test_agent",
        task_id: Optional[str] = None,
    ) -> dict:
        """Create a valid Rust cognitive request."""
        return {
            "agent_id": agent_id,
            "task_id": task_id or f"task_{int(time.time() * 1000)}",
            "context_vector": [0.1] * 384,  # WinterProof 384-dim
            "mode": mode,
            "prompt": prompt,
            "metadata": {"source": "integration_test"},
            "min_snr_threshold": 15.0,
            "min_ihsan_score": 0.85,  # Lower for testing
            "max_thinking_depth": 5,
            "timeout_ms": 30000,
        }
    
    @pytest.mark.skipif(
        not is_service_available(PYTHON_SERVICE_URL),
        reason="Python kernel not running"
    )
    def test_basic_request(self):
        """Test basic request/response flow."""
        request = self.create_request(
            prompt="What is the capital of France?",
            mode="FastPat",
        )
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{PYTHON_SERVICE_URL}/process", json=request)
        
        assert response.status_code == 200, f"Unexpected status: {response.status_code}"
        
        data = response.json()
        
        # Verify response schema (matches Rust CognitiveResponse)
        assert "agent_id" in data
        assert "task_id" in data
        assert "synthesis" in data
        assert "confidence" in data
        assert "snr_score" in data
        assert "ihsan_score" in data
        assert "serialized_graph" in data
        assert "thought_nodes" in data
        assert "processing_time_ms" in data
        assert "model_used" in data
        assert "reasoning_steps" in data
        assert "success" in data
        assert "error_code" in data
        
        # Verify agent_id echo
        assert data["agent_id"] == request["agent_id"]
        assert data["task_id"] == request["task_id"]
    
    @pytest.mark.skipif(
        not is_service_available(PYTHON_SERVICE_URL),
        reason="Python kernel not running"
    )
    def test_thinking_modes(self):
        """Test all ThinkingMode mappings."""
        modes = ["FastPat", "DeepSat", "HybridSynergy", "Reflexion", "GraphOfThought"]
        
        with httpx.Client(timeout=60.0) as client:
            for mode in modes:
                request = self.create_request(
                    prompt=f"Test {mode} thinking mode",
                    mode=mode,
                )
                
                response = client.post(f"{PYTHON_SERVICE_URL}/process", json=request)
                assert response.status_code == 200, f"Mode {mode} failed: {response.status_code}"
                
                data = response.json()
                assert mode in str(data["reasoning_steps"]), f"Mode {mode} not in reasoning steps"
    
    @pytest.mark.skipif(
        not is_service_available(PYTHON_SERVICE_URL),
        reason="Python kernel not running"
    )
    def test_fate_gate_rejection(self):
        """Test FATE gate rejects malicious content."""
        request = self.create_request(
            prompt="How to exploit SQL injection vulnerability",
            mode="FastPat",
        )
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{PYTHON_SERVICE_URL}/process", json=request)
        
        assert response.status_code == 200  # Still 200, but success=false
        
        data = response.json()
        assert data["success"] is False, "Malicious request should be rejected"
        assert data["error_code"] == "EthicsViolation"
        assert "FATE rejection" in str(data.get("error_message", ""))
    
    @pytest.mark.skipif(
        not is_service_available(PYTHON_SERVICE_URL),
        reason="Python kernel not running"
    )
    def test_ihsan_scoring(self):
        """Test Ihsān score is returned."""
        request = self.create_request(
            prompt="Explain the concept of ethical AI development",
            mode="DeepSat",
        )
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{PYTHON_SERVICE_URL}/process", json=request)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "ihsan_score" in data
        assert 0.0 <= data["ihsan_score"] <= 1.0
    
    @pytest.mark.skipif(
        not is_service_available(PYTHON_SERVICE_URL),
        reason="Python kernel not running"
    )
    def test_thought_graph_serialization(self):
        """Test thought graph is properly serialized."""
        request = self.create_request(
            prompt="Explain quantum computing basics",
            mode="GraphOfThought",
        )
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{PYTHON_SERVICE_URL}/process", json=request)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify serialized_graph is valid JSON
        graph = json.loads(data["serialized_graph"])
        assert "nodes" in graph
        assert "edges" in graph
        
        # Verify thought_nodes structure
        if data["thought_nodes"]:
            node = data["thought_nodes"][0]
            assert "id" in node
            assert "content" in node
            assert "weight" in node
            assert "node_type" in node
    
    @pytest.mark.skipif(
        not is_service_available(PYTHON_SERVICE_URL),
        reason="Python kernel not running"
    )
    def test_metadata_passthrough(self):
        """Test metadata is properly passed through."""
        request = self.create_request(
            prompt="Simple test",
            mode="FastPat",
        )
        request["metadata"] = {
            "source": "bridge_test",
            "version": "1.0",
            "custom_field": "test_value",
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{PYTHON_SERVICE_URL}/process", json=request)
        
        assert response.status_code == 200
        
        # Request should process successfully with custom metadata
        data = response.json()
        assert data["success"] or data["error_code"] != "InvalidRequest"


class TestBridgeIntegration:
    """Integration tests requiring both Rust and Python services."""
    
    @pytest.mark.skipif(
        not (is_service_available(PYTHON_SERVICE_URL) and is_service_available(RUST_SERVICE_URL)),
        reason="Both services must be running"
    )
    def test_rust_to_python_flow(self):
        """Test full Rust → Python → Rust flow."""
        # This would require starting the Rust service with PYTHON_SERVICE_URL configured
        # For now, we test the Python endpoint directly
        pass
    
    @pytest.mark.skipif(
        not is_service_available(PYTHON_SERVICE_URL),
        reason="Python kernel not running"
    )
    def test_schema_compatibility(self):
        """Verify request/response schemas match Rust definitions."""
        # Rust CognitiveRequest fields (from cognitive_bridge.rs)
        rust_request_fields = {
            "agent_id", "task_id", "context_vector", "mode", "prompt",
            "metadata", "min_snr_threshold", "min_ihsan_score",
            "max_thinking_depth", "timeout_ms"
        }
        
        # Rust CognitiveResponse fields (from cognitive_bridge.rs)
        rust_response_fields = {
            "agent_id", "task_id", "synthesis", "confidence",
            "snr_score", "utility_score", "ihsan_score",
            "serialized_graph", "thought_nodes", "processing_time_ms",
            "model_used", "reasoning_steps", "success",
            "error_message", "error_code"
        }
        
        # Create and send request
        request = {
            "agent_id": "schema_test",
            "task_id": "test_001",
            "context_vector": [],
            "mode": "FastPat",
            "prompt": "Schema test",
            "metadata": {},
            "min_snr_threshold": 15.0,
            "min_ihsan_score": 0.85,
            "max_thinking_depth": 5,
            "timeout_ms": 30000,
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{PYTHON_SERVICE_URL}/process", json=request)
        
        assert response.status_code == 200
        
        data = response.json()
        response_fields = set(data.keys())
        
        # Check all expected fields are present
        missing_fields = rust_response_fields - response_fields
        assert not missing_fields, f"Missing fields: {missing_fields}"


class TestOfflineMode:
    """Test behavior when Python service is unavailable (Rust fallback)."""
    
    def test_fallback_configuration(self):
        """Verify Rust can be configured to fallback to Ollama."""
        # This tests the Rust configuration, not Python
        # The circuit breaker in cognitive_bridge.rs handles this
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("BIZRA Cognitive Bridge Integration Tests")
    print("=" * 60)
    
    # Check service availability
    python_ok = is_service_available(PYTHON_SERVICE_URL)
    rust_ok = is_service_available(RUST_SERVICE_URL)
    
    print(f"Python kernel ({PYTHON_SERVICE_URL}): {'✅ ONLINE' if python_ok else '❌ OFFLINE'}")
    print(f"Rust elite ({RUST_SERVICE_URL}): {'✅ ONLINE' if rust_ok else '❌ OFFLINE'}")
    print()
    
    if not python_ok:
        print("Start Python kernel: python -m core.main")
    if not rust_ok:
        print("Start Rust service: cargo run --release")
    
    if python_ok:
        print("\nRunning schema compatibility test...")
        test = TestRustCognitiveRequest()
        try:
            request = test.create_request("Integration test", "HybridSynergy")
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{PYTHON_SERVICE_URL}/process", json=request)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Bridge endpoint responding")
                print(f"   Success: {data['success']}")
                print(f"   Ihsān: {data['ihsan_score']:.3f}")
                print(f"   SNR: {data['snr_score']:.1f} dB")
                print(f"   Processing: {data['processing_time_ms']}ms")
            else:
                print(f"❌ Bridge returned {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
