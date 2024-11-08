import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from queue import PriorityQueue
from typing import Dict, List

from bson import ObjectId
from models import TaskState

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    task_id: str
    priority: TaskPriority
    steps: List[Dict]
    dependencies: List[str] = None
    max_retries: int = 3
    timeout: int = 300  # seconds


class TaskExecutionStrategy(ABC):
    @abstractmethod
    async def execute(self, task: Task) -> bool:
        pass


class DefaultTaskExecutionStrategy(TaskExecutionStrategy):
    async def execute(self, task: Task) -> bool:
        try:
            # Implementation of default execution strategy
            return True
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            return False


class TaskExecutorObserver(ABC):
    @abstractmethod
    def on_task_completed(self, task: Task, result: bool):
        pass

    @abstractmethod
    def on_task_failed(self, task: Task, error: Exception):
        pass


class TaskExecutor:
    def __init__(
        self,
        max_workers: int = 10,
        strategy: TaskExecutionStrategy = None,
        task_timeout: int = 300,
    ):
        self.max_workers = max_workers
        self.strategy = strategy or DefaultTaskExecutionStrategy()
        self.task_timeout = task_timeout
        self.task_queue = PriorityQueue()
        self.executing_tasks = {}
        self.observers: List[TaskExecutorObserver] = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Task tracking
        self.task_dependencies = {}
        self.completed_tasks = set()

    def add_observer(self, observer: TaskExecutorObserver):
        self.observers.append(observer)

    def submit_task(self, task: Task):
        """Submit a task to the execution queue"""
        with self.lock:
            # Add to priority queue with negative priority for max-heap behavior
            self.task_queue.put((-task.priority.value, task))
            if task.dependencies:
                self.task_dependencies[task.task_id] = set(task.dependencies)

    async def execute_task(self, task: Task) -> bool:
        """Execute a single task with the current strategy"""
        try:
            if task.dependencies and not self._are_dependencies_met(task):
                logger.info(f"Task {task.task_id} waiting for dependencies")
                return False

            result = await self.strategy.execute(task)

            with self.lock:
                if result:
                    self.completed_tasks.add(task.task_id)
                    for observer in self.observers:
                        observer.on_task_completed(task, result)
                else:
                    for observer in self.observers:
                        observer.on_task_failed(
                            task, Exception("Task execution failed")
                        )

            return result

        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            with self.lock:
                for observer in self.observers:
                    observer.on_task_failed(task, e)
            return False

    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies for a task are met"""
        if not task.dependencies:
            return True
        return all(dep in self.completed_tasks for dep in task.dependencies)

    async def process_tasks(self):
        """Process tasks from the queue using async/await pattern"""
        while not self._stop_event.is_set():
            try:
                # Get next task from priority queue
                priority, task = self.task_queue.get_nowait()

                # Skip if dependencies aren't met
                if not self._are_dependencies_met(task):
                    self.task_queue.put((priority, task))
                    continue

                # Execute task
                async with asyncio.TaskGroup() as tg:
                    task_coroutine = tg.create_task(self.execute_task(task))
                    try:
                        await asyncio.wait_for(
                            task_coroutine, timeout=self.task_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"Task {task.task_id} timed out")
                        for observer in self.observers:
                            observer.on_task_failed(task, Exception("Task timed out"))

            except Exception as e:
                if not isinstance(e, asyncio.QueueEmpty):
                    logger.error(f"Error processing tasks: {e}")
                await asyncio.sleep(1)

    def start(self):
        """Start the task executor"""
        self._stop_event.clear()
        asyncio.run(self.process_tasks())

    def stop(self):
        """Stop the task executor"""
        self._stop_event.set()
        self.executor.shutdown(wait=True)


class BOSSTaskExecutor(TaskExecutor):
    """Specialized TaskExecutor for BOSS system"""

    def __init__(self, boss_instance, **kwargs):
        super().__init__(**kwargs)
        self.boss = boss_instance

    async def execute_task(self, task: Task) -> bool:
        """Custom execution logic for BOSS tasks"""
        try:
            # Convert Task to BOSS-specific format
            boss_task = self._convert_to_boss_task(task)

            # Process using BOSS's task processing logic
            await self.boss._process_task_with_inference(boss_task)

            return True

        except Exception as e:
            logger.error(f"Error executing BOSS task {task.task_id}: {e}")
            return False

    def _convert_to_boss_task(self, task: Task) -> Dict:
        """Convert generic Task to BOSS-specific task format"""
        return {
            "_id": ObjectId(task.task_id),
            "steps": task.steps,
            "status": TaskState.CREATED.value,
            "priority": task.priority.value,
            "current_step_index": None,
            "current_agent": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
