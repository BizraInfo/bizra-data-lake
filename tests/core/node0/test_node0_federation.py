import pytest
import asyncio
from unittest.mock import patch, MagicMock
from pathlib import Path

from core.node0.heartbeat import Node0Heartbeat
from core.federation.interaction_boundary import FederationAmbassador

@pytest.fixture
def temp_node0_dir(tmp_path):
    yield tmp_path / "sovereign_state"

@patch("core.federation.interaction_boundary.FederationAmbassador.start")
@patch("core.federation.interaction_boundary.FederationAmbassador.broadcast_heartbeat_receipt")
def test_node0_federation_wiring(mock_broadcast, mock_start, temp_node0_dir):
    """
    Test Phase 48 Distributed Receipt Verification Wiring.
    
    Verifies that:
    1. FederationAmbassador is initialized during boot.
    2. Ambassador.start() is called to bind to a port.
    3. Ambassador.broadcast_heartbeat_receipt() is called at each breath.
    """
    heartbeat = Node0Heartbeat(
        data_dir=temp_node0_dir,
        node_id="test_node_48",
        signer_public_key_hex="a" * 64,
        genesis_backed=False
    )
    
    # 1. & 2. Boot triggers Ambassador start
    receipt_boot = heartbeat.boot()
    assert receipt_boot.sovereignty_proven
    assert heartbeat._federation_ambassador is not None
    assert isinstance(heartbeat._federation_ambassador, FederationAmbassador)
    mock_start.assert_called_once_with(bind_address="0.0.0.0:0")
    
    # 3. Breathe triggers Ambassador broadcast
    receipt_breath = heartbeat.breathe()
    mock_broadcast.assert_called_once()
    
    # Verify it broadcast the correct receipt data
    broadcast_args = mock_broadcast.call_args[0][0]
    assert broadcast_args["tick_number"] == 1
    assert "evidence_hash" in broadcast_args
    assert "chain_hash" in broadcast_args
    assert broadcast_args["tick_number"] == receipt_breath.tick_number

def test_federation_ambassador_lifecycle():
    """Test the Ambassador manages the background thread correctly."""
    ambassador = FederationAmbassador(
        node_id="test_ambassador",
        public_key="a" * 64,
        private_key="b" * 64
    )
    
    # Start it on a system-assigned port
    ambassador.start(bind_address="127.0.0.1:0")
    assert ambassador._thread is not None
    assert ambassador._thread.is_alive()
    
    # Broadcast a mock receipt
    mock_receipt = {"tick_number": 99, "ihsan_composite": 0.99}
    ambassador.broadcast_heartbeat_receipt(mock_receipt)
    
    # Stop it
    ambassador.stop()
    assert not ambassador._thread.is_alive()
