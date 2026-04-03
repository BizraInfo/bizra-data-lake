"""
Data Lake Adapter for BIZRA Sovereign Nexus

Provides an interface to connect with the BIZRA-DATA-LAKE system,
enabling pattern mining and data access for the autonomous dreaming system.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class DataLakeQuery:
    """Represents a query to the data lake."""
    query_text: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 100
    offset: int = 0


@dataclass
class DataLakeResult:
    """Represents a result from the data lake."""
    data: List[Dict[str, Any]]
    total_count: int
    query_id: str
    timestamp: float


class DataLakeAdapter:
    """
    Adapter to connect with the BIZRA-DATA-LAKE system.
    
    Enables access to the data lake for pattern mining, research,
    and knowledge discovery to support autonomous dreaming and
    hypothesis generation.
    """
    
    def __init__(self, base_url: str = "https://localhost:8443", api_key: Optional[str] = None):
        """
        Initialize the Data Lake adapter.
        
        Args:
            base_url: Base URL for the Data Lake API
            api_key: API key for authentication (if required)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        
    async def connect(self) -> bool:
        """
        Establish connection to the Data Lake system.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Create an HTTP session
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self.session = aiohttp.ClientSession(headers=headers)
            
            # Test the connection
            test_url = f"{self.base_url}/health"
            # Disable SSL verification for local dev environment
            async with self.session.get(test_url, ssl=False) as resp:
                if resp.status == 200:
                    self.connected = True
                    print(f"Connected to Data Lake at {self.base_url}")
                    return True
                else:
                    print(f"Health check failed: {resp.status}")
                    return False
                    
        except Exception as e:
            print(f"Failed to connect to Data Lake: {e}")
            self.connected = False
            return False
    
    async def query(self, query_obj: DataLakeQuery) -> Optional[DataLakeResult]:
        """
        Query the data lake with the specified query using MCP protocol.
        
        Args:
            query_obj: Query object containing text and filters
            
        Returns:
            Query result or None if failed
        """
        if not self.connected or not self.session:
            print("Not connected to Data Lake")
            return None
        
        try:
            # MCP uses the root endpoint for JSON-RPC
            url = f"{self.base_url}/"
            
            # Construct MCP JSON-RPC 2.0 request
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "knowledge_retrieve",
                    "arguments": {
                        "query": query_obj.query_text,
                        "limit": query_obj.limit
                    }
                },
                "id": str(query_obj.query_text.__hash__())
            }
            
            async with self.session.post(url, json=payload, ssl=False) as resp:
                if resp.status == 200:
                    json_response = await resp.json()
                    
                    # Check for RPC error
                    if 'error' in json_response:
                        print(f"MCP Error: {json_response['error']}")
                        return None
                        
                    # Extract result from tool execution
                    # MCP response format: { ..., "result": { "content": [...] } }
                    # But our bridge returns the result dict directly in the result field for this tool
                    mcp_result = json_response.get('result', {})
                    
                    # The bridge returns { "results": [...], "total_nodes": ... } inside the result
                    data = mcp_result.get('results', [])
                    total = mcp_result.get('results_count', len(data))
                    
                    return DataLakeResult(
                        data=data,
                        total_count=total,
                        query_id=json_response.get('id', ''),
                        timestamp=0  # Timestamp not always returned in top level
                    )
                else:
                    print(f"Query failed: {resp.status}")
                    return None
            
        except Exception as e:
            print(f"Error querying data lake: {e}")
            return None
    
    async def search_patterns(self, seed: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for patterns related to a seed concept in the data lake.
        
        Args:
            seed: The seed concept to search for
            limit: Maximum number of results to return
            
        Returns:
            List of pattern dictionaries or empty list if failed
        """
        if not self.connected or not self.session:
            print("Not connected to Data Lake")
            return []
        
        try:
            # Construct a search query for patterns related to the seed
            search_query = f"patterns related to '{seed}' OR connections between '{seed}' and *"
            
            query_obj = DataLakeQuery(
                query_text=search_query,
                filters={
                    "category": ["patterns", "relationships", "connections"],
                    "min_quality_score": 0.7
                },
                limit=limit
            )
            
            result = await self.query(query_obj)
            if result:
                return result.data
            else:
                return []
                
        except Exception as e:
            print(f"Error searching patterns in data lake: {e}")
            return []
    
    async def get_related_concepts(self, concept: str, limit: int = 20) -> List[str]:
        """
        Get concepts related to a given concept from the data lake.
        
        Args:
            concept: The concept to find related concepts for
            limit: Maximum number of related concepts to return
            
        Returns:
            List of related concept strings or empty list if failed
        """
        if not self.connected or not self.session:
            print("Not connected to Data Lake")
            return []
        
        try:
            search_query = f"concepts related to '{concept}'"
            
            query_obj = DataLakeQuery(
                query_text=search_query,
                filters={"type": "concept"},
                limit=limit
            )
            
            result = await self.query(query_obj)
            if result:
                # Extract concept names from the results
                concepts = []
                for item in result.data:
                    if 'name' in item:
                        concepts.append(item['name'])
                    elif 'concept' in item:
                        concepts.append(item['concept'])
                    elif 'title' in item:
                        concepts.append(item['title'])
                return concepts
            else:
                return []
                
        except Exception as e:
            print(f"Error getting related concepts from data lake: {e}")
            return []
    
    async def get_knowledge_graph_fragment(self, seed: str) -> Optional[Dict[str, Any]]:
        """
        Get a fragment of the knowledge graph centered on a seed concept.
        
        Args:
            seed: The seed concept to center the graph on
            
        Returns:
            Knowledge graph fragment or None if failed
        """
        if not self.connected or not self.session:
            print("Not connected to Data Lake")
            return None
        
        try:
            search_query = f"knowledge graph around '{seed}'"
            
            query_obj = DataLakeQuery(
                query_text=search_query,
                filters={"type": "knowledge_graph"},
                limit=1
            )
            
            result = await self.query(query_obj)
            if result and result.data:
                return result.data[0]
            else:
                return None
                
        except Exception as e:
            print(f"Error getting knowledge graph from data lake: {e}")
            return None
    
    async def submit_insight(self, insight: Dict[str, Any]) -> bool:
        """
        Submit a newly discovered insight to the data lake.
        
        Args:
            insight: The insight to submit
            
        Returns:
            True if submitted successfully, False otherwise
        """
        if not self.connected or not self.session:
            print("Not connected to Data Lake")
            return False
        
        try:
            url = f"{self.base_url}/insights"
            
            async with self.session.post(url, json=insight) as resp:
                return resp.status in [200, 201]
                
        except Exception as e:
            print(f"Error submitting insight to data lake: {e}")
            return False
    
    async def disconnect(self):
        """Close the connection to the Data Lake system."""
        self.connected = False
        if self.session:
            await self.session.close()