"""
Synapse Adapter for BIZRA Sovereign Nexus

Provides an interface to connect with the Trinity Synapse system,
handling state persistence, coordination, and communication.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import redis.asyncio as redis


@dataclass
class SynapseMessage:
    """Represents a message in the synapse system."""
    channel: str
    sender: str
    content: Union[str, Dict[str, Any]]
    timestamp: float
    correlation_id: Optional[str] = None


class SynapseAdapter:
    """
    Adapter to connect with the Trinity Synapse system.
    
    Handles communication with the core synapse bus for state persistence,
    agent coordination, and cross-component messaging.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize the Synapse adapter.
        
        Args:
            host: Redis host for the synapse system
            port: Redis port for the synapse system
            db: Redis database number
        """
        self.host = host
        self.port = port
        self.db = db
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub = None
        self.connected = False
        
    async def connect(self) -> bool:
        """
        Establish connection to the Synapse system.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self.redis_client = redis.Redis(host=self.host, port=self.port, db=self.db)
            
            # Test the connection
            await self.redis_client.ping()
            self.connected = True
            
            print(f"Connected to Synapse at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to Synapse: {e}")
            self.connected = False
            return False
    
    async def publish_message(self, channel: str, message: SynapseMessage) -> bool:
        """
        Publish a message to a synapse channel.
        
        Args:
            channel: Channel to publish to
            message: Message to publish
            
        Returns:
            True if published successfully, False otherwise
        """
        if not self.connected or not self.redis_client:
            print("Not connected to Synapse")
            return False
        
        try:
            # Serialize the message
            message_dict = {
                "channel": message.channel,
                "sender": message.sender,
                "content": message.content,
                "timestamp": message.timestamp,
                "correlation_id": message.correlation_id
            }
            serialized_message = json.dumps(message_dict)
            
            # Publish to the channel
            await self.redis_client.publish(channel, serialized_message)
            return True
            
        except Exception as e:
            print(f"Error publishing message: {e}")
            return False
    
    async def subscribe_to_channel(self, channel: str):
        """
        Subscribe to a synapse channel for receiving messages.
        
        Args:
            channel: Channel to subscribe to
        """
        if not self.connected or not self.redis_client:
            print("Not connected to Synapse")
            return None
        
        try:
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe(channel)
            return self.pubsub
            
        except Exception as e:
            print(f"Error subscribing to channel: {e}")
            return None
    
    async def get_message(self, pubsub) -> Optional[SynapseMessage]:
        """
        Get a message from the subscribed channel.
        
        Args:
            pubsub: PubSub object from subscribe_to_channel
            
        Returns:
            Received message or None if none available
        """
        if not pubsub:
            return None
            
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True)
            if message and message['type'] == 'message':
                # Deserialize the message
                data = json.loads(message['data'].decode('utf-8'))
                return SynapseMessage(
                    channel=data['channel'],
                    sender=data['sender'],
                    content=data['content'],
                    timestamp=data['timestamp'],
                    correlation_id=data.get('correlation_id')
                )
            return None
            
        except Exception as e:
            print(f"Error getting message: {e}")
            return None
    
    async def store_state(self, key: str, value: Union[str, Dict[str, Any]]) -> bool:
        """
        Store state in the synapse system.
        
        Args:
            key: Key to store the value under
            value: Value to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.connected or not self.redis_client:
            print("Not connected to Synapse")
            return False
        
        try:
            # Serialize value if it's a dict
            serialized_value = json.dumps(value) if isinstance(value, dict) else value
            await self.redis_client.set(key, serialized_value)
            return True
            
        except Exception as e:
            print(f"Error storing state: {e}")
            return False
    
    async def retrieve_state(self, key: str) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Retrieve state from the synapse system.
        
        Args:
            key: Key to retrieve the value for
            
        Returns:
            Retrieved value or None if not found
        """
        if not self.connected or not self.redis_client:
            print("Not connected to Synapse")
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                decoded_value = value.decode('utf-8')
                # Try to deserialize as JSON, fallback to string
                try:
                    return json.loads(decoded_value)
                except json.JSONDecodeError:
                    return decoded_value
            return None
            
        except Exception as e:
            print(f"Error retrieving state: {e}")
            return None
    
    async def update_state(self, key: str, updates: Dict[str, Any]) -> bool:
        """
        Update state in the synapse system by merging with existing state.
        
        Args:
            key: Key of the state to update
            updates: Updates to merge into the existing state
            
        Returns:
            True if updated successfully, False otherwise
        """
        current_state = await self.retrieve_state(key)
        
        if current_state is None:
            # No existing state, treat as new
            return await self.store_state(key, updates)
        
        if not isinstance(current_state, dict):
            print(f"Current state for key '{key}' is not a dict, cannot update")
            return False
        
        # Merge updates with current state
        merged_state = {**current_state, **updates}
        return await self.store_state(key, merged_state)
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """
        List keys in the synapse system matching a pattern.
        
        Args:
            pattern: Pattern to match keys against (Redis glob-style)
            
        Returns:
            List of matching keys
        """
        if not self.connected or not self.redis_client:
            print("Not connected to Synapse")
            return []
        
        try:
            keys = await self.redis_client.keys(pattern)
            return [key.decode('utf-8') for key in keys]
            
        except Exception as e:
            print(f"Error listing keys: {e}")
            return []
    
    async def disconnect(self):
        """Close the connection to the Synapse system."""
        self.connected = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()