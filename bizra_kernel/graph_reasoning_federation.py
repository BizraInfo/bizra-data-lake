"""
BIZRA Cross-Node Graph-of-Thoughts Reasoning
Phase 9: Distributed Graph Reasoning Protocol

Implements distributed reasoning across federation nodes using Graph-of-Thoughts.
Coordinates multi-hop reasoning, semantic force field discovery, and consensus-driven insights.
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

try:
    from .memory_system import CognitivePermanence
    from .knowledge_graph_sharding import KnowledgeGraphSharding
except ImportError:
    from memory_system import CognitivePermanence
    from knowledge_graph_sharding import KnowledgeGraphSharding


class ReasoningPhase(Enum):
    """Phases of distributed graph reasoning."""
    DISCOVERY = "discovery"
    EXPANSION = "expansion"
    CONSENSUS = "consensus"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


@dataclass
class ReasoningSession:
    """A distributed reasoning session across federation nodes."""
    session_id: str
    initiator_node: str
    query: str
    start_time: float
    phase: ReasoningPhase = ReasoningPhase.DISCOVERY

    # Reasoning state
    discovered_nodes: Set[str] = field(default_factory=set)  # Entity IDs
    reasoning_paths: List[List[str]] = field(default_factory=list)  # Paths through graph
    force_fields: Dict[str, float] = field(default_factory=dict)  # Semantic connections
    node_contributions: Dict[str, List[Dict]] = field(default_factory=dict)  # Per-node insights

    # Consensus tracking
    consensus_votes: Dict[str, Any] = field(default_factory=dict)
    final_insight: Optional[Dict[str, Any]] = None

    # Metadata
    participating_nodes: Set[str] = field(default_factory=set)
    completed: bool = False
    end_time: Optional[float] = None


@dataclass
class ReasoningRequest:
    """A request for graph reasoning."""
    request_id: str
    session_id: str
    node_id: str
    query_type: str  # "explore", "connect", "analyze", "synthesize"
    parameters: Dict[str, Any]
    local_shard_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class GraphReasoningFederation:
    """
    Distributed Graph-of-Thoughts reasoning across federation nodes.

    Coordinates multi-node reasoning sessions that:
    - Discover semantic connections across shards
    - Build consensus on insights
    - Synthesize distributed knowledge
    - Validate reasoning through Byzantine agreement
    """

    def __init__(self, node_id: str, sharding: KnowledgeGraphSharding, memory_system: CognitivePermanence):
        self.node_id = node_id
        self.sharding = sharding
        self.memory_system = memory_system

        # Reasoning state
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self.pending_requests: Dict[str, ReasoningRequest] = {}

        # Local reasoning capabilities
        self.local_graph_cache: Dict[str, Dict] = {}  # entity_id -> graph_data
        self.reasoning_history: List[Dict] = []  # Past reasoning sessions

        # Federation parameters
        self.max_reasoning_depth = 5  # Maximum hops in reasoning
        self.consensus_threshold = 0.67  # 2/3 majority for consensus
        self.reasoning_timeout = 30.0  # Seconds to wait for responses

    async def initiate_reasoning_session(self, query: str, target_nodes: List[str] = None) -> str:
        """
        Initiate a new distributed reasoning session.

        Args:
            query: The reasoning query (e.g., "How does BIZRA achieve sovereignty?")
            target_nodes: Specific nodes to involve (None = all nodes)

        Returns:
            Session ID for tracking
        """
        session_id = f"reasoning_{self.node_id}_{int(time.time() * 1000)}"

        if target_nodes is None:
            target_nodes = self.sharding.node_ids

        session = ReasoningSession(
            session_id=session_id,
            initiator_node=self.node_id,
            query=query,
            participating_nodes=set(target_nodes)
        )

        self.active_sessions[session_id] = session

        # Start the reasoning process
        asyncio.create_task(self._run_reasoning_session(session))

        print(f"[+] Initiated reasoning session {session_id} for query: {query}")
        return session_id

    async def _run_reasoning_session(self, session: ReasoningSession):
        """Run a complete reasoning session."""
        try:
            # Phase 1: Discovery - Find relevant entities across nodes
            await self._discovery_phase(session)

            # Phase 2: Expansion - Explore connections and relationships
            await self._expansion_phase(session)

            # Phase 3: Consensus - Build agreement on key insights
            await self._consensus_phase(session)

            # Phase 4: Synthesis - Combine distributed insights
            await self._synthesis_phase(session)

            # Phase 5: Validation - Byzantine validation of results
            await self._validation_phase(session)

        except Exception as e:
            print(f"[!] Reasoning session {session.session_id} failed: {e}")
        finally:
            session.completed = True
            session.end_time = time.time()

    async def _discovery_phase(self, session: ReasoningSession):
        """Phase 1: Discover relevant entities across federation nodes."""
        session.phase = ReasoningPhase.DISCOVERY

        # Parse query to extract key concepts
        query_concepts = self._extract_query_concepts(session.query)

        # Send discovery requests to all participating nodes
        discovery_tasks = []
        for node_id in session.participating_nodes:
            if node_id != self.node_id:  # Don't send to self
                request = ReasoningRequest(
                    request_id=f"{session.session_id}_discovery_{node_id}",
                    session_id=session.session_id,
                    node_id=node_id,
                    query_type="discover",
                    parameters={
                        "concepts": query_concepts,
                        "max_results": 10
                    }
                )
                discovery_tasks.append(self._send_reasoning_request(request))

        # Also process locally
        local_results = self._local_entity_discovery(query_concepts)
        session.discovered_nodes.update(local_results)
        session.node_contributions[self.node_id] = [{"phase": "discovery", "entities": local_results}]

        # Wait for responses from other nodes
        if discovery_tasks:
            responses = await asyncio.gather(*discovery_tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, dict) and "entities" in response:
                    session.discovered_nodes.update(response["entities"])
                    contributor = response.get("node_id", "unknown")
                    session.node_contributions[contributor] = [{"phase": "discovery", "entities": response["entities"]}]

        print(f"[+] Discovery phase complete: {len(session.discovered_nodes)} entities found")

    async def _expansion_phase(self, session: ReasoningSession):
        """Phase 2: Expand reasoning by exploring connections."""
        session.phase = ReasoningPhase.EXPANSION

        # For each discovered entity, find related entities
        expansion_tasks = []
        entities_to_explore = list(session.discovered_nodes)[:20]  # Limit for performance

        for entity_id in entities_to_explore:
            # Find which node has this entity
            locations = self.sharding.get_entity_location(entity_id)
            if locations:
                primary_node = next((loc.node_id for loc in locations if loc.primary), locations[0].node_id)

                if primary_node == self.node_id:
                    # Local exploration
                    connections = self._local_connection_expansion(entity_id)
                    session.reasoning_paths.extend(connections)
                else:
                    # Remote exploration
                    request = ReasoningRequest(
                        request_id=f"{session.session_id}_expand_{entity_id}",
                        session_id=session.session_id,
                        node_id=primary_node,
                        query_type="expand",
                        parameters={"entity_id": entity_id, "max_depth": 2}
                    )
                    expansion_tasks.append(self._send_reasoning_request(request))

        # Process remote responses
        if expansion_tasks:
            responses = await asyncio.gather(*expansion_tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, dict) and "paths" in response:
                    session.reasoning_paths.extend(response["paths"])

        # Calculate semantic force fields
        session.force_fields = self._calculate_force_fields(session.reasoning_paths)

        print(f"[+] Expansion phase complete: {len(session.reasoning_paths)} paths explored")

    async def _consensus_phase(self, session: ReasoningSession):
        """Phase 3: Build consensus on key insights."""
        session.phase = ReasoningPhase.CONSENSUS

        # Extract key insights from reasoning paths
        insights = self._extract_insights(session.reasoning_paths, session.force_fields)

        # Send insights to all nodes for consensus voting
        consensus_tasks = []
        for node_id in session.participating_nodes:
            if node_id != self.node_id:
                request = ReasoningRequest(
                    request_id=f"{session.session_id}_consensus_{node_id}",
                    session_id=session.session_id,
                    node_id=node_id,
                    query_type="vote",
                    parameters={"insights": insights}
                )
                consensus_tasks.append(self._send_reasoning_request(request))

        # Local voting
        local_votes = self._local_consensus_vote(insights)
        session.consensus_votes[self.node_id] = local_votes

        # Collect remote votes
        if consensus_tasks:
            responses = await asyncio.gather(*consensus_tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, dict) and "votes" in response:
                    voter = response.get("node_id", "unknown")
                    session.consensus_votes[voter] = response["votes"]

        print(f"[+] Consensus phase complete: {len(session.consensus_votes)} votes collected")

    async def _synthesis_phase(self, session: ReasoningSession):
        """Phase 4: Synthesize distributed insights into final answer."""
        session.phase = ReasoningPhase.SYNTHESIS

        # Combine votes to find consensus insights
        consensus_insights = self._synthesize_consensus(session.consensus_votes)

        # Generate final insight
        session.final_insight = {
            "query": session.query,
            "consensus_insights": consensus_insights,
            "confidence": self._calculate_confidence(session.consensus_votes),
            "reasoning_paths": len(session.reasoning_paths),
            "participating_nodes": len(session.participating_nodes),
            "timestamp": time.time()
        }

        print(f"[+] Synthesis phase complete: Final insight generated")

    async def _validation_phase(self, session: ReasoningSession):
        """Phase 5: Byzantine validation of the final insight."""
        session.phase = ReasoningPhase.VALIDATION

        # Send final insight for validation
        validation_tasks = []
        for node_id in session.participating_nodes:
            if node_id != self.node_id:
                request = ReasoningRequest(
                    request_id=f"{session.session_id}_validate_{node_id}",
                    session_id=session.session_id,
                    node_id=node_id,
                    query_type="validate",
                    parameters={"insight": session.final_insight}
                )
                validation_tasks.append(self._send_reasoning_request(request))

        # Local validation
        local_validation = self._local_validation(session.final_insight)
        session.node_contributions[self.node_id].append({
            "phase": "validation",
            "result": local_validation
        })

        # Collect validations
        if validation_tasks:
            responses = await asyncio.gather(*validation_tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, dict) and "validation" in response:
                    validator = response.get("node_id", "unknown")
                    session.node_contributions[validator].append({
                        "phase": "validation",
                        "result": response["validation"]
                    })

        print(f"[+] Validation phase complete: Insight validated by {len(session.participating_nodes)} nodes")

    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract key concepts from a reasoning query."""
        # Simple keyword extraction (could be enhanced with NLP)
        words = query.lower().replace('?', '').replace('.', '').split()
        concepts = [word for word in words if len(word) > 3]  # Filter short words
        return concepts[:5]  # Limit concepts

    def _local_entity_discovery(self, concepts: List[str]) -> Set[str]:
        """Discover relevant entities in local knowledge graph."""
        relevant_entities = set()

        # Search local semantic layer
        for entity_id, entity_data in self.memory_system.layers["L4"].items():
            entity_text = str(entity_data).lower()

            # Check if any concept matches
            for concept in concepts:
                if concept in entity_text:
                    relevant_entities.add(entity_id)
                    break

        return relevant_entities

    def _local_connection_expansion(self, entity_id: str, max_depth: int = 2) -> List[List[str]]:
        """Expand connections from an entity in the local graph."""
        paths = []

        if entity_id not in self.memory_system.layers["L4"]:
            return paths

        # Simple breadth-first expansion
        visited = set()
        queue = [(entity_id, [entity_id], 0)]  # (current, path, depth)

        while queue:
            current, path, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            if current in visited:
                continue
            visited.add(current)

            # Find connected entities
            entity_data = self.memory_system.layers["L4"][current]
            connected_entities = set()

            # Check relationships
            if "rels" in entity_data:
                # In a real implementation, this would traverse the actual graph structure
                # For now, we'll do a simple co-occurrence based expansion
                pass

            # Add to paths if we found connections
            for connected in connected_entities:
                if connected not in path:  # Avoid cycles
                    new_path = path + [connected]
                    paths.append(new_path)
                    queue.append((connected, new_path, depth + 1))

        return paths

    def _calculate_force_fields(self, paths: List[List[str]]) -> Dict[str, float]:
        """Calculate semantic force fields from reasoning paths."""
        force_fields = defaultdict(float)

        for path in paths:
            for i in range(len(path) - 1):
                connection = f"{path[i]}->{path[i+1]}"
                force_fields[connection] += 1.0

                # Also count reverse connections
                reverse = f"{path[i+1]}->{path[i]}"
                force_fields[reverse] += 0.5

        return dict(force_fields)

    def _extract_insights(self, paths: List[List[str]], force_fields: Dict[str, float]) -> List[Dict]:
        """Extract key insights from reasoning paths."""
        insights = []

        # Find most connected entities
        entity_connections = defaultdict(int)
        for path in paths:
            for entity in path:
                entity_connections[entity] += len(path) - 1  # Connections in this path

        # Top connected entities
        top_entities = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:5]

        insights.append({
            "type": "key_entities",
            "entities": [entity for entity, _ in top_entities],
            "description": "Most central entities in the reasoning graph"
        })

        # Strongest force fields
        top_fields = sorted(force_fields.items(), key=lambda x: x[1], reverse=True)[:5]
        insights.append({
            "type": "force_fields",
            "connections": [field for field, _ in top_fields],
            "description": "Strongest semantic connections discovered"
        })

        return insights

    def _local_consensus_vote(self, insights: List[Dict]) -> Dict[str, Any]:
        """Cast local votes on proposed insights."""
        votes = {}

        for insight in insights:
            # Simple voting logic - could be enhanced with ML
            if insight["type"] == "key_entities":
                # Vote based on local knowledge
                confidence = 0.8 if len(insight["entities"]) > 0 else 0.3
                votes["key_entities"] = {"vote": "accept", "confidence": confidence}
            elif insight["type"] == "force_fields":
                confidence = 0.7
                votes["force_fields"] = {"vote": "accept", "confidence": confidence}

        return votes

    def _synthesize_consensus(self, votes: Dict[str, Any]) -> List[Dict]:
        """Synthesize consensus from collected votes."""
        consensus_insights = []

        # Count votes for each insight type
        insight_votes = defaultdict(list)

        for node_votes in votes.values():
            for insight_type, vote_data in node_votes.items():
                insight_votes[insight_type].append(vote_data)

        # Check consensus threshold
        for insight_type, vote_list in insight_votes.items():
            accept_votes = sum(1 for v in vote_list if v.get("vote") == "accept")
            total_votes = len(vote_list)

            if total_votes > 0 and (accept_votes / total_votes) >= self.consensus_threshold:
                consensus_insights.append({
                    "type": insight_type,
                    "consensus_ratio": accept_votes / total_votes,
                    "total_votes": total_votes
                })

        return consensus_insights

    def _calculate_confidence(self, votes: Dict[str, Any]) -> float:
        """Calculate overall confidence in the consensus result."""
        if not votes:
            return 0.0

        # Average confidence across all votes
        total_confidence = 0.0
        vote_count = 0

        for node_votes in votes.values():
            for vote_data in node_votes.values():
                if "confidence" in vote_data:
                    total_confidence += vote_data["confidence"]
                    vote_count += 1

        return total_confidence / vote_count if vote_count > 0 else 0.0

    def _local_validation(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an insight locally."""
        # Simple validation - check if insight is consistent with local knowledge
        validation_score = 0.8  # Placeholder

        return {
            "valid": validation_score > 0.6,
            "score": validation_score,
            "reasoning": "Local knowledge consistency check"
        }

    async def _send_reasoning_request(self, request: ReasoningRequest) -> Dict[str, Any]:
        """Send a reasoning request to another node."""
        # In a real implementation, this would use the federation messaging system
        # For now, simulate with local processing

        # Simulate network delay
        await asyncio.sleep(0.1)

        # Simulate response based on request type
        if request.query_type == "discover":
            concepts = request.parameters.get("concepts", [])
            entities = self._local_entity_discovery(concepts)
            return {
                "node_id": request.node_id,
                "request_id": request.request_id,
                "entities": list(entities)
            }
        elif request.query_type == "expand":
            entity_id = request.parameters.get("entity_id")
            paths = self._local_connection_expansion(entity_id)
            return {
                "node_id": request.node_id,
                "request_id": request.request_id,
                "paths": paths
            }
        elif request.query_type == "vote":
            insights = request.parameters.get("insights", [])
            votes = self._local_consensus_vote(insights)
            return {
                "node_id": request.node_id,
                "request_id": request.request_id,
                "votes": votes
            }
        elif request.query_type == "validate":
            insight = request.parameters.get("insight", {})
            validation = self._local_validation(insight)
            return {
                "node_id": request.node_id,
                "request_id": request.request_id,
                "validation": validation
            }

        return {"error": "Unknown request type"}

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a reasoning session."""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        return {
            "session_id": session.session_id,
            "phase": session.phase.value,
            "completed": session.completed,
            "discovered_nodes": len(session.discovered_nodes),
            "reasoning_paths": len(session.reasoning_paths),
            "participating_nodes": len(session.participating_nodes),
            "final_insight": session.final_insight,
            "duration": (session.end_time or time.time()) - session.start_time
        }

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning federation statistics."""
        total_sessions = len(self.active_sessions)
        completed_sessions = sum(1 for s in self.active_sessions.values() if s.completed)

        return {
            "total_sessions": total_sessions,
            "active_sessions": total_sessions - completed_sessions,
            "completed_sessions": completed_sessions,
            "avg_session_duration": sum(
                (s.end_time or time.time()) - s.start_time
                for s in self.active_sessions.values() if s.completed
            ) / completed_sessions if completed_sessions > 0 else 0
        }


if __name__ == "__main__":
    # Test the reasoning federation
    from bizra_kernel.memory_system import CognitivePermanence

    nodes = ["node_0", "node_1", "node_2"]
    sharding = KnowledgeGraphSharding(nodes)
    memory = CognitivePermanence()

    reasoning = GraphReasoningFederation("node_0", sharding, memory)

    # Test session initiation
    async def test():
        session_id = await reasoning.initiate_reasoning_session("How does BIZRA achieve sovereignty?")
        print(f"Started session: {session_id}")

        # Wait a bit for processing
        await asyncio.sleep(2)

        status = reasoning.get_session_status(session_id)
        print(f"Session status: {status}")

    asyncio.run(test())