"""
Dual Agentic Adapter for BIZRA Sovereign Nexus

Provides an interface to connect with the Dual Agentic Server (Rust),
providing PAT/SAT orchestration and metric analysis.
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional

class DualAgenticAdapter:
    """
    Adapter to connect with the Dual Agentic Server (Port 9091).
    """
    
    def __init__(self, base_url: str = "http://localhost:9091", timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False

    async def connect(self) -> bool:
        """Establishes connection (session) to the Dual Agentic Server."""
        try:
            # We don't need a persistent connection for REST, but we initialize the session
            if self.session is None or self.session.closed:
                 # Localhost often doesn't need SSL verification, but explicit disable is safe for dev
                self.session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(ssl=False)
                )
            
            # Health check
            async with self.session.get(f"{self.base_url}/health", timeout=5) as response:
                if response.status == 200:
                    self.connected = True
                    print(f"Connected to Dual Agentic Server at {self.base_url}")
                    return True
                else:
                    print(f"Dual Agentic Server health check failed: {response.status}")
                    self.connected = False
                    return False
        except Exception as e:
            print(f"Failed to connect to Dual Agentic Server: {e}")
            self.connected = False
            return False

    async def execute_dual(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a request against the /dual/execute endpoint.
        """
        if not self.session:
            await self.connect()
            
        try:
            async with self.session.post(
                f"{self.base_url}/dual/execute", 
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    print(f"Dual execute failed: {response.status} - {text}")
                    return None
        except Exception as e:
            print(f"Error executing dual request: {e}")
            return None

    async def execute_enhanced(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a request against the /enhanced/execute endpoint.
        """
        if not self.session:
            await self.connect()
            
        try:
            async with self.session.post(
                f"{self.base_url}/enhanced/execute", 
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    print(f"Enhanced execute failed: {response.status} - {text}")
                    return None
        except Exception as e:
            print(f"Error executing enhanced request: {e}")
            return None

    async def get_covenant_metrics(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metrics from /api/covenant/metrics.
        """
        if not self.session:
            await self.connect()

        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with self.session.get(
                f"{self.base_url}/api/covenant/metrics",
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Metrics retrieval failed: {response.status}")
                    return None
        except Exception as e:
            print(f"Error retrieving metrics: {e}")
            return None

    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.connected = False
