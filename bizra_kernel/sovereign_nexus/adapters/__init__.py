"""
Adapters for the BIZRA Sovereign Nexus

This package contains adapters that connect the Nexus to various external systems:
- RustBridgeAdapter: Connects to Rust FFI components
- SynapseAdapter: Connects to the trinity synapse system
- DataLakeAdapter: Connects to the BIZRA-DATA-LAKE
"""

from .rust_bridge_adapter import RustBridgeAdapter
from .synapse_adapter import SynapseAdapter
from .data_lake_adapter import DataLakeAdapter

__all__ = [
    'RustBridgeAdapter',
    'SynapseAdapter',
    'DataLakeAdapter'
]