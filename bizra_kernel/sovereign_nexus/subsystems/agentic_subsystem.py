"""
Agentic Subsystem for BIZRA Sovereign Nexus

Handles PAT/SAT orchestration and warm pool management.
Wraps the agent factory and manages agentic behaviors.
"""

import asyncio
import random
from contextlib import suppress
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from core.agent_factory import AgentFactory  # Assuming this exists in the codebase


@dataclass
class AgenticTask:
    """Represents a task for an agent."""
    task_id: str
    agent_type: str
    goal: str
    context: Dict[str, Any]
    priority: int = 1
    deadline: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class AgenticResult:
    """Result from agentic processing."""
    task_id: str
    agent_id: str
    result: Any
    success: bool
    execution_time: float
    metadata: Dict[str, Any]


class AgenticSubsystem:
    """
    Agentic processing subsystem of the BIZRA Sovereign Nexus.
    
    Handles:
    - PAT (Personal Agentic Team) orchestration
    - SAT (System Agentic Team) coordination
    - Warm pool management for efficient agent reuse
    - Task assignment and result collection
    """
    
    def __init__(
        self,
        agent_factory: Optional[AgentFactory] = None,
        pool_size: int = 10,
        queue_size: Optional[int] = None,
    ):
        """
        Initialize the Agentic Subsystem.
        
        Args:
            agent_factory: Factory for creating agents
            pool_size: Size of the warm agent pool
        """
        self.agent_factory = agent_factory or AgentFactory()
        self.pool_size = pool_size
        self.warm_pool = []
        self.active_agents = {}
        self.task_queue = asyncio.Queue(maxsize=queue_size or max(pool_size * 10, 50))
        self.results = {}
        self.running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._pool_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the agentic subsystem and warm agent pool."""
        if self.running:
            return
        try:
            # Create a warm pool of agents for quick task execution
            for i in range(self.pool_size):
                agent = await self.agent_factory.create_agent(agent_type="general")
                await self._return_agent_to_pool(agent)
            
            self.running = True
            if self._worker_task is None or self._worker_task.done():
                self._worker_task = asyncio.create_task(self._process_tasks())
            print(f"Agentic Subsystem initialized with pool of {len(self.warm_pool)} agents")
        except Exception as e:
            print(f"Failed to initialize Agentic Subsystem: {e}")
            self.running = False
    
    async def assign_task(self, task: AgenticTask) -> str:
        """
        Assign a task to an available agent.
        
        Args:
            task: The task to assign
            
        Returns:
            Task ID for tracking
        """
        if not self.running:
            await self.initialize()
        if not self.running:
            raise RuntimeError("Agentic subsystem failed to initialize")
        
        # Add task to queue
        try:
            await asyncio.wait_for(self.task_queue.put(task), timeout=1.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Agentic task queue is full. Apply backpressure before submitting more tasks."
            ) from exc
        
        return task.task_id
    
    async def _process_tasks(self):
        """Process tasks from the queue using available agents."""
        while self.running:
            try:
                task = await self.task_queue.get()
            except asyncio.CancelledError:
                break
            
            # Get an agent from the warm pool or create a new one
            try:
                agent = await self._get_available_agent(task.agent_type)
                
                if agent is None:
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        await asyncio.sleep(0.05 * (task.retry_count + 1))
                        await self.task_queue.put(task)
                    else:
                        self.results[task.task_id] = AgenticResult(
                            task_id=task.task_id,
                            agent_id="unassigned",
                            result="No agent available after retries",
                            success=False,
                            execution_time=0.0,
                            metadata={"error": "no_agent_available"},
                        )
                    continue

                # Mark agent as busy
                agent_id = getattr(agent, 'id', f'agent_{random.randint(1000, 9999)}')
                self.active_agents[agent_id] = agent
                
                # Execute the task
                start_time = asyncio.get_event_loop().time()
                try:
                    result = await self._execute_task(agent, task)
                    execution_time = asyncio.get_event_loop().time() - start_time
                    
                    agentic_result = AgenticResult(
                        task_id=task.task_id,
                        agent_id=agent_id,
                        result=result,
                        success=True,
                        execution_time=execution_time,
                        metadata={}
                    )
                    
                    self.results[task.task_id] = agentic_result
                    
                except Exception as e:
                    execution_time = asyncio.get_event_loop().time() - start_time
                    agentic_result = AgenticResult(
                        task_id=task.task_id,
                        agent_id=agent_id,
                        result=str(e),
                        success=False,
                        execution_time=execution_time,
                        metadata={'error': str(e)}
                    )
                    self.results[task.task_id] = agentic_result
                finally:
                    self.active_agents.pop(agent_id, None)
                    await self._return_agent_to_pool(agent)
            finally:
                self.task_queue.task_done()
    
    async def _get_available_agent(self, agent_type: str):
        """Get an available agent of the specified type."""
        # First, try to find an agent of the right type in the warm pool
        suitable_agent = None
        async with self._pool_lock:
            for i, agent in enumerate(self.warm_pool):
                if hasattr(agent, 'agent_type') and agent.agent_type == agent_type:
                    suitable_agent = self.warm_pool.pop(i)
                    break
            
            # If no suitable agent found, use any available agent
            if suitable_agent is None and self.warm_pool:
                suitable_agent = self.warm_pool.pop()
        
        if suitable_agent is not None:
            return suitable_agent

        # Pool empty: attempt controlled creation
        try:
            return await self.agent_factory.create_agent(agent_type=agent_type)
        except Exception:
            try:
                return await self.agent_factory.create_agent(agent_type="general")
            except Exception:
                return None

    async def _return_agent_to_pool(self, agent) -> None:
        """Return an agent to pool while keeping pool bounded."""
        if agent is None:
            return
        async with self._pool_lock:
            if len(self.warm_pool) < self.pool_size:
                self.warm_pool.append(agent)
    
    async def _execute_task(self, agent, task: AgenticTask):
        """Execute a task with the given agent."""
        # This is a simplified execution - in a real system this would depend on the agent implementation
        try:
            # If the agent has an execute_task method, use it
            if hasattr(agent, 'execute_task'):
                return await agent.execute_task(task.goal, task.context)
            # Otherwise, try to call the agent directly with the task
            elif callable(agent):
                return await agent(task.goal, **task.context)
            else:
                # Fallback: just return the task goal as the result
                return f"Executed task: {task.goal}"
        except Exception as e:
            raise e
    
    async def get_result(self, task_id: str) -> Optional[AgenticResult]:
        """
        Get the result of a task.
        
        Args:
            task_id: ID of the task to get result for
            
        Returns:
            AgenticResult if available, None otherwise
        """
        return self.results.get(task_id)
    
    async def create_pat_team(self, team_spec: Dict[str, Any]) -> List[str]:
        """
        Create a Personal Agentic Team (PAT) based on specifications.
        
        Args:
            team_spec: Specification of the team to create
            
        Returns:
            List of agent IDs in the team
        """
        if not self.running:
            await self.initialize()
        
        pat_agents = []
        
        # Define standard PAT roles if not specified
        pat_roles = team_spec.get('roles', [
            'researcher', 'analyst', 'planner', 'executor', 
            'validator', 'reporter', 'coordinator'
        ])
        
        for role in pat_roles:
            try:
                agent = await self.agent_factory.create_agent(agent_type=role)
                agent_id = getattr(agent, 'id', f'{role}_agent_{random.randint(1000, 9999)}')
                
                # Add to warm pool initially
                await self._return_agent_to_pool(agent)
                pat_agents.append(agent_id)
                
            except Exception as e:
                print(f"Failed to create {role} agent: {e}")
                # Create a general agent as fallback
                agent = await self.agent_factory.create_agent(agent_type="general")
                agent_id = f'fallback_agent_{random.randint(1000, 9999)}'
                await self._return_agent_to_pool(agent)
                pat_agents.append(agent_id)
        
        return pat_agents
    
    async def create_sat_team(self, team_spec: Dict[str, Any]) -> List[str]:
        """
        Create a System Agentic Team (SAT) based on specifications.
        
        Args:
            team_spec: Specification of the team to create
            
        Returns:
            List of agent IDs in the team
        """
        if not self.running:
            await self.initialize()
        
        sat_agents = []
        
        # Define standard SAT roles if not specified
        sat_roles = team_spec.get('roles', [
            'monitor', 'validator', 'security', 'compliance',
            'governance', 'audit'
        ])
        
        for role in sat_roles:
            try:
                agent = await self.agent_factory.create_agent(agent_type=role)
                agent_id = getattr(agent, 'id', f'{role}_agent_{random.randint(1000, 9999)}')
                
                # Add to warm pool initially
                await self._return_agent_to_pool(agent)
                sat_agents.append(agent_id)
                
            except Exception as e:
                print(f"Failed to create {role} agent: {e}")
                # Create a general agent as fallback
                agent = await self.agent_factory.create_agent(agent_type="general")
                agent_id = f'fallback_agent_{random.randint(1000, 9999)}'
                await self._return_agent_to_pool(agent)
                sat_agents.append(agent_id)
        
        return sat_agents
    
    async def coordinate_multi_agent_task(
        self,
        task_descriptions: List[Dict[str, Any]],
        coordination_strategy: str = "parallel"
    ) -> List[AgenticResult]:
        """
        Coordinate a task across multiple agents.
        
        Args:
            task_descriptions: List of task descriptions for different agents
            coordination_strategy: Strategy for coordination ("parallel", "sequential", "hierarchical")
            
        Returns:
            List of AgenticResults from all agents
        """
        if not self.running:
            await self.initialize()
        
        # Create tasks from descriptions
        tasks = []
        for desc in task_descriptions:
            task = AgenticTask(
                task_id=desc.get('task_id', f'task_{random.randint(10000, 99999)}'),
                agent_type=desc.get('agent_type', 'general'),
                goal=desc['goal'],
                context=desc.get('context', {}),
                priority=desc.get('priority', 1)
            )
            tasks.append(task)
        
        # Assign all tasks
        task_ids = []
        for task in tasks:
            task_id = await self.assign_task(task)
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        for task_id in task_ids:
            # Poll for results (in a real system, this would be more elegant)
            result = None
            timeout = 30  # 30 second timeout
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < timeout:
                result = await self.get_result(task_id)
                if result:
                    break
                await asyncio.sleep(0.1)
            
            if result:
                results.append(result)
            else:
                # Create a timeout result
                results.append(AgenticResult(
                    task_id=task_id,
                    agent_id="unknown",
                    result="Task timed out",
                    success=False,
                    execution_time=timeout,
                    metadata={'error': 'Timeout'}
                ))
        
        return results
    
    async def shutdown(self):
        """Shutdown the agentic subsystem."""
        self.running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        # Clean up any active agents
        self.active_agents.clear()
        async with self._pool_lock:
            self.warm_pool.clear()
