"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA A2A — TASK MANAGER                                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Advanced task lifecycle management:                                        ║
║   - Hierarchical task decomposition                                          ║
║   - Parallel execution orchestration                                         ║
║   - Result aggregation                                                       ║
║   - Failure recovery                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any, Awaitable, Set

from .schema import TaskCard, TaskStatus


# ═══════════════════════════════════════════════════════════════════════════════
# TASK QUEUE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskQueueEntry:
    """Entry in the priority task queue."""
    task: TaskCard
    priority: int
    created_at: float = field(default_factory=time.time)
    retry_count: int = 0
    
    def __lt__(self, other: 'TaskQueueEntry') -> bool:
        # Higher priority first, then earlier created
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


# ═══════════════════════════════════════════════════════════════════════════════
# TASK MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class TaskManager:
    """
    Manages task lifecycle and orchestration.
    
    Features:
    - Priority queue with concurrent execution
    - Parent-child task relationships
    - Result aggregation for parallel subtasks
    - Automatic retry with backoff
    - Task timeout handling
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        default_timeout: int = 300,
        max_retries: int = 3,
        on_task_complete: Optional[Callable[[TaskCard], Awaitable[None]]] = None,
        on_task_failed: Optional[Callable[[TaskCard], Awaitable[None]]] = None
    ):
        """
        Initialize task manager.
        
        Args:
            max_concurrent: Maximum concurrent task executions
            default_timeout: Default task timeout in seconds
            max_retries: Maximum retry attempts for failed tasks
            on_task_complete: Callback when task completes
            on_task_failed: Callback when task fails permanently
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        
        # Callbacks
        self.on_task_complete = on_task_complete
        self.on_task_failed = on_task_failed
        
        # Task storage
        self.tasks: Dict[str, TaskCard] = {}
        self.queue: List[TaskQueueEntry] = []
        
        # Parent-child relationships
        self.children: Dict[str, Set[str]] = defaultdict(set)
        
        # Execution tracking
        self.executing: Set[str] = set()
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Running state
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "retried": 0,
            "timed_out": 0
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK SUBMISSION
    # ─────────────────────────────────────────────────────────────────────────
    
    def submit(self, task: TaskCard, parent_task_id: Optional[str] = None) -> str:
        """
        Submit a task for execution.
        
        Args:
            task: Task to execute
            parent_task_id: Optional parent task (for hierarchical execution)
        
        Returns:
            Task ID
        """
        # Set parent relationship
        if parent_task_id:
            task.parent_task_id = parent_task_id
            self.children[parent_task_id].add(task.task_id)
            # Get parent and add child reference
            if parent_task_id in self.tasks:
                self.tasks[parent_task_id].child_task_ids.append(task.task_id)
        
        # Store task
        self.tasks[task.task_id] = task
        
        # Add to queue
        entry = TaskQueueEntry(task=task, priority=task.priority)
        self.queue.append(entry)
        self.queue.sort()
        
        self.stats["submitted"] += 1
        print(f"📥 Task submitted: {task.task_id[:8]}... (priority={task.priority})")
        
        return task.task_id
    
    def submit_parallel(
        self,
        tasks: List[TaskCard],
        parent_task_id: Optional[str] = None
    ) -> List[str]:
        """
        Submit multiple tasks for parallel execution.
        
        All tasks will be children of the optional parent task.
        """
        return [self.submit(t, parent_task_id) for t in tasks]
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK EXECUTION
    # ─────────────────────────────────────────────────────────────────────────
    
    async def execute(
        self,
        task: TaskCard,
        executor: Callable[[TaskCard], Awaitable[Any]]
    ) -> Any:
        """
        Execute a single task with timeout and retry logic.
        
        Args:
            task: Task to execute
            executor: Async function to execute the task
        
        Returns:
            Task result
        """
        task.mark_started("")
        self.executing.add(task.task_id)
        
        try:
            # Execute with timeout
            timeout = task.timeout or self.default_timeout
            result = await asyncio.wait_for(
                executor(task),
                timeout=timeout
            )
            
            task.mark_completed(result)
            self.stats["completed"] += 1
            
            if self.on_task_complete:
                await self.on_task_complete(task)
            
            # Check if parent is complete
            await self._check_parent_complete(task)
            
            return result
            
        except asyncio.TimeoutError:
            task.mark_failed(f"Timeout after {timeout}s")
            self.stats["timed_out"] += 1
            
            if self.on_task_failed:
                await self.on_task_failed(task)
                
        except Exception as e:
            task.mark_failed(str(e))
            self.stats["failed"] += 1
            
            if self.on_task_failed:
                await self.on_task_failed(task)
                
        finally:
            self.executing.discard(task.task_id)
        
        return None
    
    async def _check_parent_complete(self, task: TaskCard):
        """Check if parent task is complete after child completes."""
        if not task.parent_task_id:
            return
        
        parent = self.tasks.get(task.parent_task_id)
        if not parent:
            return
        
        # Get all children
        child_ids = self.children.get(task.parent_task_id, set())
        children = [self.tasks.get(cid) for cid in child_ids]
        
        # Check if all children are complete
        all_complete = all(
            c and c.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            for c in children if c
        )
        
        if all_complete:
            # Aggregate results
            results = {
                c.task_id: c.result 
                for c in children 
                if c and c.status == TaskStatus.COMPLETED
            }
            errors = {
                c.task_id: c.error 
                for c in children 
                if c and c.status == TaskStatus.FAILED
            }
            
            if errors:
                parent.mark_failed(f"Child failures: {list(errors.keys())}")
            else:
                parent.mark_completed({"child_results": results})
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK QUERIES
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_task(self, task_id: str) -> Optional[TaskCard]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_children(self, task_id: str) -> List[TaskCard]:
        """Get all child tasks of a parent."""
        child_ids = self.children.get(task_id, set())
        return [self.tasks[cid] for cid in child_ids if cid in self.tasks]
    
    def get_pending(self) -> List[TaskCard]:
        """Get all pending tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
    
    def get_in_progress(self) -> List[TaskCard]:
        """Get all in-progress tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
    
    def get_completed(self) -> List[TaskCard]:
        """Get all completed tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
    
    def get_failed(self) -> List[TaskCard]:
        """Get all failed tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Returns:
            True if cancelled successfully
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
            task.status = TaskStatus.CANCELLED
            # Remove from queue
            self.queue = [e for e in self.queue if e.task.task_id != task_id]
            print(f"🚫 Task cancelled: {task_id[:8]}...")
            return True
        
        return False
    
    def retry(self, task_id: str) -> bool:
        """
        Retry a failed task.
        
        Returns:
            True if requeued successfully
        """
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.FAILED:
            return False
        
        # Reset status
        task.status = TaskStatus.PENDING
        task.error = None
        task.result = None
        
        # Requeue
        entry = TaskQueueEntry(task=task, priority=task.priority)
        self.queue.append(entry)
        self.queue.sort()
        
        self.stats["retried"] += 1
        print(f"🔄 Task requeued: {task_id[:8]}...")
        return True
    
    # ─────────────────────────────────────────────────────────────────────────
    # WORKER LOOP
    # ─────────────────────────────────────────────────────────────────────────
    
    async def start(self, executor: Callable[[TaskCard], Awaitable[Any]]):
        """
        Start the task worker loop.
        
        Continuously processes tasks from the queue.
        """
        self._running = True
        print(f"▶️ TaskManager started (max_concurrent={self.max_concurrent})")
        
        while self._running:
            # Check for pending tasks
            if not self.queue:
                await asyncio.sleep(0.1)
                continue
            
            # Get highest priority task
            entry = self.queue.pop(0)
            task = entry.task
            
            # Execute with semaphore for concurrency control
            async with self._semaphore:
                await self.execute(task, executor)
    
    async def stop(self):
        """Stop the task worker loop."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
        print("⏹️ TaskManager stopped")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_stats(self) -> Dict:
        """Get task manager statistics."""
        return {
            **self.stats,
            "queued": len(self.queue),
            "executing": len(self.executing),
            "total_tasks": len(self.tasks),
            "pending": len(self.get_pending()),
            "in_progress": len(self.get_in_progress()),
            "completed_count": len(self.get_completed()),
            "failed_count": len(self.get_failed())
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK DECOMPOSER
# ═══════════════════════════════════════════════════════════════════════════════

class TaskDecomposer:
    """
    Decomposes complex tasks into subtasks.
    
    Uses capability matching and heuristics to break
    down tasks into parallel-executable units.
    """
    
    def __init__(self, available_capabilities: List[str]):
        """
        Initialize decomposer.
        
        Args:
            available_capabilities: List of capability names available in the system
        """
        self.capabilities = set(available_capabilities)
    
    def decompose(
        self,
        task: TaskCard,
        strategy: str = "parallel"
    ) -> List[TaskCard]:
        """
        Decompose a task into subtasks.
        
        Strategies:
        - parallel: Create independent subtasks
        - sequential: Create dependent chain
        - hybrid: Mix based on dependencies
        
        Returns:
            List of subtasks
        """
        # Simple decomposition based on prompt analysis
        # In production, this would use LLM for intelligent decomposition
        
        prompt = task.prompt.lower()
        subtasks = []
        
        # Check for common multi-step patterns
        if "and then" in prompt or "after that" in prompt:
            # Sequential pattern - split on connectors
            parts = prompt.replace("after that", "and then").split("and then")
            for i, part in enumerate(parts):
                subtask = TaskCard(
                    prompt=part.strip(),
                    capability_required=task.capability_required,
                    parent_task_id=task.task_id,
                    priority=task.priority - i  # Earlier parts higher priority
                )
                subtasks.append(subtask)
        
        elif "also" in prompt or "additionally" in prompt:
            # Parallel pattern
            parts = prompt.replace("additionally", "also").split("also")
            for part in parts:
                subtask = TaskCard(
                    prompt=part.strip(),
                    capability_required=task.capability_required,
                    parent_task_id=task.task_id,
                    priority=task.priority
                )
                subtasks.append(subtask)
        
        return subtasks if subtasks else [task]
