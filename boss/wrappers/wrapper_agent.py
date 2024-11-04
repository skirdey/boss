# wrappers/wrapper_agent.py

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict

import colorlog
from kafka import KafkaConsumer, KafkaProducer
from openai import OpenAI

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
        kafka_bootstrap_servers="localhost:9092",
    ):
        self.agent_id = agent_id
        self.task_topic = f"tasks_for_{self.agent_id}"
        self.result_topic = f"results_from_{self.agent_id}"
        self.running = True
        self.consumer = KafkaConsumer(
            self.task_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

                    # Process the task
                    result = self.process_task(task)

                    logger.info(f"Processed result in WrapperAgent: {result}")

                    # Send the result back to BOSS via Kafka
                    self.send_result_to_boss(result)

            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Error in receive method: {e}")

    def send_result_to_boss(self, result: Dict[str, Any]) -> None:
        """Send the task result back to the BOSS via Kafka"""
        try:
            self.producer.send(self.result_topic, result)
            self.producer.flush()
            logger.info(f"Sent result back to BOSS: {result}")
        except Exception as e:
            logger.error(f"Error sending result to BOSS: {e}")

    @abstractmethod
    def process_task(self, task: Dict) -> Dict[str, Any]:
        """Process a single task and return the result"""
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
