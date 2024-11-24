from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import logging
import os
import signal
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import colorlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from openai import AsyncOpenAI

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class WrapperAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        kafka_bootstrap_servers: str = "localhost:9092",
        max_concurrent_tasks: int = 5,  # Maximum number of concurrent tasks
    ):
        self.agent_id = agent_id
        self.task_topic = f"tasks_for_{self.agent_id}"
        self.result_topic = f"results_from_{self.agent_id}"
        self.running = False
        self.max_concurrent_tasks = max_concurrent_tasks

        # Kafka setup
        self.consumer = AIOKafkaConsumer(
            self.task_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="boss_group",
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
        )

        self.producer = AIOKafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Semaphore to limit concurrent tasks
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # List to keep track of asyncio tasks
        self.tasks = []

        # Setup logger
        self.setup_task_logger()

    def setup_task_logger(self):
        """Set up a logger with color formatting for task processing."""
        self.task_logger = logging.getLogger(f"{self.agent_id}_tasks")
        if self.task_logger.hasHandlers():
            self.task_logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "blue",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        handler.setFormatter(formatter)
        self.task_logger.addHandler(handler)
        self.task_logger.setLevel(logging.INFO)

    async def start(self):
        """Start the agent with proper logging and signal handling."""
        self.running = True
        try:
            await self.consumer.start()
            self.task_logger.info("Kafka consumer started successfully.")
        except Exception as e:
            self.task_logger.error(f"Failed to start Kafka consumer: {e}")
            return

        try:
            await self.producer.start()
            self.task_logger.info("Kafka producer started successfully.")
        except Exception as e:
            self.task_logger.error(f"Failed to start Kafka producer: {e}")
            return

        self.task_logger.info(f"Starting agent '{self.agent_id}'.")

        # Create a task to consume messages
        consume_task = asyncio.create_task(self.consume())

        self.task_logger.info(
            f"Agent '{self.agent_id}' is now running with {self.max_concurrent_tasks} concurrent tasks."
        )
        await consume_task

    async def shutdown(self, signal_received):
        """Handle graceful shutdown."""
        self.task_logger.info(
            f"Signal {signal_received.name} received. Initiating graceful shutdown."
        )
        self.running = False

        # Wait for all running tasks to complete
        if self.tasks:
            self.task_logger.info("Waiting for running tasks to complete...")
            await asyncio.gather(*self.tasks, return_exceptions=True)

        await self.consumer.stop()
        await self.producer.stop()

        self.task_logger.info(f"Agent '{self.agent_id}' has been stopped.")

    async def consume(self):
        """Continuously listen for tasks and process them."""
        self.running = True
        try:
            async for message in self.consumer:
                if not self.running:
                    break

                task = message.value
                task_id = task.get("task_id", "unknown")
                self.task_logger.info(
                    f"Received task: {task_id} for agent {self.agent_id} with task {task}"
                )

                # Acquire semaphore before starting a new task
                await self.semaphore.acquire()
                task_coro = self.handle_task(task)
                task_future = asyncio.create_task(task_coro)
                task_future.add_done_callback(lambda t: self.semaphore.release())
                self.tasks.append(task_future)

                # Clean up completed tasks
                self.tasks = [t for t in self.tasks if not t.done()]

        except Exception as e:
            self.task_logger.error(f"Error in consume method: {e}")
        finally:
            await self.shutdown(signal.Signals.SIGTERM)

    async def handle_task(self, task: Dict[str, Any]):
        task_id = task.get("task_id", "unknown")
        try:
            await asyncio.wait_for(
                self._handle_task_with_timeout(task), timeout=300
            )  # Example: 5-minute timeout
        except asyncio.TimeoutError:
            self.task_logger.error(f"Timeout occurred for task {task_id}")

    async def _handle_task_with_timeout(self, task: Dict[str, Any]):
        """
        Processes the task with a timeout handling mechanism.

        Args:
            task (Dict[str, Any]): The task details in the form of a dictionary.
        """
        task_id = task.get("task_id", "unknown")
        step_id = task.get("step_id", "unknown")
        try:
            self.task_logger.info(f"Starting to process task with ID {task_id}")

            # Simulate task processing: Replace this with actual task processing logic
            result = await self.process_task(task)

            # Log successful processing
            self.task_logger.info(f"Successfully processed task with ID {task_id}")

            # Send the result back to Kafka
            await self.send_result_to_boss(result)

        except asyncio.CancelledError:
            # Handle cancellation due to timeout or other reasons
            self.task_logger.warning(f"Task with ID {task_id} was cancelled.")
            result = self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                error="Task cancelled due to timeout or agent shutdown.",
            )
            await self.send_result_to_boss(result)

        except Exception as e:
            # Catch any other exceptions and log an error
            self.task_logger.error(
                f"An error occurred while processing task {task_id}: {str(e)}"
            )
            result = self._create_task_result(
                task_id=task_id, step_id=step_id, error=f"Error occurred: {str(e)}"
            )
            await self.send_result_to_boss(result)

    async def send_result_to_boss(self, result: Dict[str, Any]):
        """
        Asynchronously send results back to the BOSS via Kafka.

        Args:
            result (Dict[str, Any]): The result dictionary to be sent.
        """
        try:
            await self.producer.send_and_wait(self.result_topic, result)
            self.task_logger.info(
                f"Sent result back to BOSS: {result.get('task_id', 'unknown')}"
            )
        except Exception as e:
            self.task_logger.error(f"Error sending result to BOSS: {e}")

    @abstractmethod
    async def process_task(self, task: Dict) -> Dict[str, Any]:
        """Asynchronously process a single task and return the result."""
        pass

    def _create_task_result(
        self,
        task_id: str,
        step_id: str,
        step_description: Optional[str] = None,
        result: Any = "None",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Helper function to create a standardized task result."""
        task_result = {
            "task_id": str(task_id),
            "step_id": str(step_id),
            "agent_id": self.agent_id,
            "result": result,
            "agent_output": result,
            "success": True,
        }

        if step_description:
            task_result["step_description"] = step_description
        if error:
            task_result["error"] = error
            task_result["result"] = "None"
            task_result["success"] = False
        if metadata:
            task_result["metadata"] = metadata
        return task_result
