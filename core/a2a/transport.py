"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA A2A — TRANSPORT LAYER                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Multi-protocol transport for A2A communication:                            ║
║   - HTTP REST for request/response                                           ║
║   - WebSocket for streaming                                                  ║
║   - UDP (via Federation) for gossip                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any, Awaitable, List

sys.path.insert(0, "c:\\BIZRA-DATA-LAKE")

from .schema import A2AMessage, MessageType, AgentCard


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSPORT BASE
# ═══════════════════════════════════════════════════════════════════════════════

class A2ATransport:
    """
    Base transport layer for A2A communication.
    
    Manages:
    - Outbound message sending
    - Inbound message receiving
    - Connection lifecycle
    """
    
    def __init__(
        self,
        agent_id: str,
        on_message: Optional[Callable[[A2AMessage], Awaitable[Optional[A2AMessage]]]] = None
    ):
        """
        Initialize transport.
        
        Args:
            agent_id: This agent's ID
            on_message: Handler for received messages
        """
        self.agent_id = agent_id
        self.on_message = on_message
        
        # Connection tracking
        self.connections: Dict[str, Any] = {}
        
        # Running state
        self._running = False
    
    async def start(self):
        """Start the transport layer."""
        self._running = True
        print(f"🚀 A2ATransport started for {self.agent_id}")
    
    async def stop(self):
        """Stop the transport layer."""
        self._running = False
        print(f"🛑 A2ATransport stopped for {self.agent_id}")
    
    async def send(self, msg: A2AMessage, target: str) -> bool:
        """
        Send a message to a target agent.
        
        Args:
            msg: Message to send
            target: Target agent ID or address
        
        Returns:
            True if sent successfully
        """
        raise NotImplementedError
    
    async def broadcast(self, msg: A2AMessage) -> int:
        """
        Broadcast a message to all known agents.
        
        Returns:
            Number of agents reached
        """
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL TRANSPORT (In-Memory)
# ═══════════════════════════════════════════════════════════════════════════════

class LocalTransport(A2ATransport):
    """
    In-memory transport for local multi-agent testing.
    
    All agents share a common message bus.
    """
    
    # Class-level message bus
    _bus: Dict[str, 'LocalTransport'] = {}
    
    def __init__(
        self,
        agent_id: str,
        on_message: Optional[Callable[[A2AMessage], Awaitable[Optional[A2AMessage]]]] = None
    ):
        super().__init__(agent_id, on_message)
        LocalTransport._bus[agent_id] = self
    
    async def start(self):
        await super().start()
    
    async def stop(self):
        await super().stop()
        if self.agent_id in LocalTransport._bus:
            del LocalTransport._bus[self.agent_id]
    
    async def send(self, msg: A2AMessage, target: str) -> bool:
        """Send directly to target agent's handler."""
        if target not in LocalTransport._bus:
            print(f"⚠️ Target agent not found: {target}")
            return False
        
        target_transport = LocalTransport._bus[target]
        if target_transport.on_message:
            response = await target_transport.on_message(msg)
            if response and self.on_message:
                # Handle response
                await self.on_message(response)
        
        return True
    
    async def broadcast(self, msg: A2AMessage) -> int:
        """Broadcast to all registered agents."""
        count = 0
        for agent_id, transport in LocalTransport._bus.items():
            if agent_id != self.agent_id:
                if transport.on_message:
                    await transport.on_message(msg)
                    count += 1
        return count
    
    @classmethod
    def clear_bus(cls):
        """Clear the message bus (for testing)."""
        cls._bus.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# UDP TRANSPORT (Federation Integration)
# ═══════════════════════════════════════════════════════════════════════════════

class UDPTransport(A2ATransport):
    """
    UDP transport using the Federation gossip layer.
    
    Integrates with core.federation.gossip for P2P communication.
    """
    
    def __init__(
        self,
        agent_id: str,
        bind_address: str,
        on_message: Optional[Callable[[A2AMessage], Awaitable[Optional[A2AMessage]]]] = None
    ):
        super().__init__(agent_id, on_message)
        self.bind_address = bind_address
        self._transport = None
        self._protocol = None
    
    async def start(self):
        """Start UDP listener."""
        await super().start()
        
        host, port = self.bind_address.split(":")
        port = int(port)
        
        loop = asyncio.get_event_loop()
        
        class A2AProtocol(asyncio.DatagramProtocol):
            def __init__(self, transport_ref):
                self.transport_ref = transport_ref
                self.udp_transport = None
            
            def connection_made(self, transport):
                self.udp_transport = transport
            
            def datagram_received(self, data, addr):
                asyncio.create_task(self._handle(data, addr))
            
            async def _handle(self, data, addr):
                try:
                    msg = A2AMessage.from_bytes(data)
                    if self.transport_ref.on_message:
                        response = await self.transport_ref.on_message(msg)
                        if response and self.udp_transport:
                            self.udp_transport.sendto(response.to_bytes(), addr)
                except Exception as e:
                    print(f"⚠️ UDP message error: {e}")
        
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: A2AProtocol(self),
            local_addr=(host, port)
        )
        
        print(f"🌐 UDP transport bound to {host}:{port}")
    
    async def stop(self):
        """Stop UDP listener."""
        await super().stop()
        if self._transport:
            self._transport.close()
    
    async def send(self, msg: A2AMessage, target: str) -> bool:
        """Send UDP datagram to target address."""
        if not self._transport:
            return False
        
        try:
            host, port = target.split(":")
            data = msg.to_bytes()
            self._transport.sendto(data, (host, int(port)))
            return True
        except Exception as e:
            print(f"⚠️ UDP send error: {e}")
            return False
    
    async def broadcast(self, msg: A2AMessage) -> int:
        """Broadcast requires peer list - not implemented for raw UDP."""
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID TRANSPORT
# ═══════════════════════════════════════════════════════════════════════════════

class HybridTransport(A2ATransport):
    """
    Hybrid transport combining multiple protocols.
    
    Uses:
    - Local for same-process agents
    - UDP for network agents
    """
    
    def __init__(
        self,
        agent_id: str,
        bind_address: str = "",
        on_message: Optional[Callable[[A2AMessage], Awaitable[Optional[A2AMessage]]]] = None
    ):
        super().__init__(agent_id, on_message)
        self.bind_address = bind_address
        
        # Sub-transports
        self._local = LocalTransport(agent_id, on_message)
        self._udp: Optional[UDPTransport] = None
        
        # Agent address registry
        self.addresses: Dict[str, str] = {}  # agent_id -> address
    
    async def start(self):
        await super().start()
        await self._local.start()
        
        if self.bind_address:
            self._udp = UDPTransport(self.agent_id, self.bind_address, self.on_message)
            await self._udp.start()
    
    async def stop(self):
        await super().stop()
        await self._local.stop()
        if self._udp:
            await self._udp.stop()
    
    def register_address(self, agent_id: str, address: str):
        """Register a network address for an agent."""
        self.addresses[agent_id] = address
    
    async def send(self, msg: A2AMessage, target: str) -> bool:
        """Send via appropriate transport."""
        # Check if local
        if target in LocalTransport._bus:
            return await self._local.send(msg, target)
        
        # Check if we have network address
        if target in self.addresses and self._udp:
            return await self._udp.send(msg, self.addresses[target])
        
        # Target might be an address directly
        if ":" in target and self._udp:
            return await self._udp.send(msg, target)
        
        print(f"⚠️ No route to agent: {target}")
        return False
    
    async def broadcast(self, msg: A2AMessage) -> int:
        """Broadcast via all transports."""
        count = await self._local.broadcast(msg)
        
        # Also send to known network addresses
        if self._udp:
            for agent_id, address in self.addresses.items():
                if await self._udp.send(msg, address):
                    count += 1
        
        return count
