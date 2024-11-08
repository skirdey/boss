# wrappers/wrapper_agent.py
import json
import logging
import os
import signal
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict

import colorlog
from kafka import KafkaConsumer, KafkaProducer
from openai import OpenAI

from boss.events import shutdown_event

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
            group_id="boss_group",
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        signal.signal(signal.SIGINT, self.signal_handler)  # Handles Ctrl+C
        signal.signal(signal.SIGTERM, self.signal_handler)  # Handles termination

    def signal_handler(self, signal_received, frame):
        """Handle termination signals for graceful shutdown."""
        logger.info(f"Signal {signal_received} received. Initiating graceful shutdown.")
        self.stop()

    def receive(self) -> None:
        """Continuously listen for tasks and process them."""
        logger.info(f"{self.agent_id} started listening for tasks.")
        while self.running and not shutdown_event.is_set():
            try:
                for message in self.consumer:
                    if not self.running or shutdown_event.is_set():
                        break
                    task = message.value

                    logger.info(
                        f"Received task: {task.get('_id', 'unknown')} with status {task.get('status', 'unknown')}"
                    )

                    # Process the task
                    result = self.process_task(task)

                    logger.info(f"Processed result: {result}")

                    # Send the result back to BOSS via Kafka
                    self.send_result_to_boss(result)

            except Exception as e:
                if self.running:
                    logger.error(f"Error in receive method: {e}")
                else:
                    logger.info("Receive loop terminated due to shutdown.")
                break

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
        """Start the agent with proper logging and signal handling."""
        logger.info(f"Starting agent '{self.agent_id}'.")
        self.receive_thread = threading.Thread(target=self.receive)
        self.receive_thread.start()
        logger.info(f"Agent '{self.agent_id}' is now running.")

        try:
            # Keep the main thread alive to handle signals
            while self.running:
                self.receive_thread.join(timeout=1)
        except KeyboardInterrupt:
            # This block is optional since signal_handler handles SIGINT
            self.stop()

    def stop(self):
        """Stop the agent gracefully."""
        if not self.running:
            return  # Prevent multiple stop attempts
        logger.info(f"Stopping agent '{self.agent_id}'...")
        self.running = False
        try:
            if self.consumer:
                self.consumer.close()
                logger.info("Kafka consumer closed.")
        except Exception as e:
            logger.error(f"Error closing Kafka consumer: {e}")

        try:
            if self.producer:
                self.producer.close(timeout=1)
                logger.info("Kafka producer closed.")
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {e}")

        try:
            if self.receive_thread.is_alive():
                self.receive_thread.join(timeout=5)
                logger.info("Receive thread terminated.")
        except Exception as e:
            logger.error(f"Error stopping receive thread: {e}")

        logger.info(f"Agent '{self.agent_id}' has been stopped.")

    def setup_task_logger(self):
        """Set up a separate logger for task processing."""
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
