import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict

import colorlog
from anthropic import Anthropic
from kafka import KafkaConsumer, KafkaProducer
from openai import OpenAI
from pymongo import MongoClient

from models import StepResult
from utils import serialize_task
from workflow_state_manager import WorkflowStateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class WrapperAgent(ABC):
    def __init__(
        self,
        agent_id,
        db_uri="mongodb://localhost:27017/",
        kafka_bootstrap_servers="localhost:9092",
    ):
        self.agent_id = agent_id
        self.topic = f"tasks_for_{self.agent_id}"
        self.running = True
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(serialize_task(v)).encode("utf-8"),
        )
        self.client = MongoClient(db_uri)
        self.db = self.client["task_db"]
        self.tasks_collection = self.db["tasks"]
        self.agents_collection = self.db["agents"]
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        self.workflow_state_manager = WorkflowStateManager(
            self.tasks_collection, self.db["task_history"], self.agents_collection
        )

    def process_step(self, task: Dict) -> Dict[str, Any]:
        """Process a single step of the workflow"""
        # pass
        try:
            current_step = self.workflow_state_manager.get_current_step(task["_id"])
            if not current_step:
                logger.warning(
                    f"No current step found for task {task['_id']}. Checking if workflow is complete."
                )

                # Check if workflow is actually complete
                workflow_state = self.workflow_state_manager.get_workflow_state(
                    task["_id"]
                )

                if workflow_state and workflow_state.get("completed", False):
                    return {
                        "success": True,
                        "result": "Workflow completed",
                        "task_id": task["_id"],
                    }

                return {
                    "success": False,
                    "error": "No current step found and workflow not complete",
                    "task_id": task["_id"],
                }

            # Process the step using the abstract method
            start_time = datetime.now(timezone.utc)
            result = self.process_task(task)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Create StepResult object
            step_result = StepResult(
                step_description=current_step["step_description"],
                success=result["success"],
                result=result.get("result", ""),
                error=result.get("error", ""),
                execution_time=execution_time,
                metadata={"agent_id": self.agent_id},
            )

            # Update workflow state with step result
            updated = self.workflow_state_manager.update_workflow_state(
                task_id=task["_id"],
                step_result=step_result,
                agent_id=self.agent_id,
            )

            if not updated:
                logger.error(f"Failed to update workflow state for task {task['_id']}")
                return {
                    "success": False,
                    "error": "Failed to update workflow state",
                    "task_id": task["_id"],
                }

            return {
                "success": True,
                "result": result.get("result", ""),
                "task_id": task["_id"],
            }

        except Exception as e:
            logger.error(f"Error processing step: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task["_id"],
            }

    def receive(self) -> None:
        logger.info(f"{self.agent_id} started listening for tasks.")
        while self.running:
            try:
                for message in self.consumer:
                    if not self.running:
                        break
                    task = message.value

                    logger.info(
                        f"Received task: {task['_id']} with status {task['status']}"
                    )

                    # Process the current step
                    result = self.process_step(task)

                    # logger.info(f"wrapper_agent Result recieve: {pformat(result)}")

            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Error in receive method: {e}")

    @abstractmethod
    def process_task(self, task: Dict) -> Dict[str, Any]:
        """Process a single task step"""
        pass

    def start(self):
        self.setup_task_logger()
        self.receive_thread = threading.Thread(target=self.receive, daemon=True)
        self.receive_thread.start()

    def stop(self):
        self.running = False
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        if self.client:
            self.client.close()
        if self.receive_thread.is_alive():
            self.receive_thread.join()
        logger.info(f"{self.agent_id} has been stopped.")

    def setup_task_logger(self):
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
