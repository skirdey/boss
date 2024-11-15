import json
import logging
import os
import signal
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Lock

import colorlog
from kafka import KafkaConsumer, KafkaProducer
from openai import OpenAI

from boss.events import shutdown_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class WrapperAgent(ABC):
    def __init__(
        self,
        agent_id,
        kafka_bootstrap_servers="localhost:9092",
        max_workers=5,  # Number of worker threads
        task_queue_size=100  # Maximum number of tasks to queue
    ):
        self.agent_id = agent_id
        self.task_topic = f"tasks_for_{self.agent_id}"
        self.result_topic = f"results_from_{self.agent_id}"
        self.running = True
        self.max_workers = max_workers
        
        # Thread-safe task queue
        self.task_queue = queue.Queue(maxsize=task_queue_size)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, 
                                            thread_name_prefix=f"{self.agent_id}_worker")
        
        # Lock for thread-safe logging
        self.logger_lock = Lock()
        
        # Kafka setup
        self.consumer = KafkaConsumer(
            self.task_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
            group_id="boss_group",
            # Enable multiple consumers in the same group
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # List to keep track of futures
        self.futures = []
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signal_received, frame):
        """Handle termination signals for graceful shutdown."""
        with self.logger_lock:
            logger.info(f"Signal {signal_received} received. Initiating graceful shutdown.")
        self.stop()

    def process_and_send(self, task: Dict[str, Any]) -> None:
        """Process a single task and send results in a worker thread."""
        try:
            with self.logger_lock:
                logger.info(f"Processing task {task.get('_id', 'unknown')} in thread {threading.current_thread().name}")
            
            result = self.process_task(task)
            
            # Send result back to BOSS
            self.send_result_to_boss(result)
            
            with self.logger_lock:
                logger.info(f"Completed processing task {task.get('_id', 'unknown')}")
                
        except Exception as e:
            with self.logger_lock:
                logger.error(f"Error processing task {task.get('_id', 'unknown')}: {str(e)}")

    def receive(self) -> None:
        """Continuously listen for tasks and distribute them to worker threads."""
        with self.logger_lock:
            logger.info(f"{self.agent_id} started listening for tasks with {self.max_workers} workers.")
        
        while self.running and not shutdown_event.is_set():
            try:
                for message in self.consumer:
                    if not self.running or shutdown_event.is_set():
                        break
                    
                    task = message.value
                    
                    with self.logger_lock:
                        logger.info(f"Received task: {task.get('_id', 'unknown')}")
                    
                    # Submit task to thread pool
                    future = self.thread_pool.submit(self.process_and_send, task)
                    self.futures.append(future)
                    
                    # Clean up completed futures
                    self.futures = [f for f in self.futures if not f.done()]
                    
            except Exception as e:
                if self.running:
                    with self.logger_lock:
                        logger.error(f"Error in receive method: {e}")
                break

    def send_result_to_boss(self, result: Dict[str, Any]) -> None:
        """Thread-safe method to send results back to BOSS via Kafka"""
        try:
            self.producer.send(self.result_topic, result)
            self.producer.flush()
            with self.logger_lock:
                logger.info(f"Sent result back to BOSS: {result.get('task_id', 'unknown')}")
        except Exception as e:
            with self.logger_lock:
                logger.error(f"Error sending result to BOSS: {e}")

    @abstractmethod
    def process_task(self, task: Dict) -> Dict[str, Any]:
        """Process a single task and return the result"""
        pass

    def start(self):
        """Start the agent with proper logging and signal handling."""
        with self.logger_lock:
            logger.info(f"Starting agent '{self.agent_id}'.")
        
        self.receive_thread = threading.Thread(target=self.receive, 
                                             name=f"{self.agent_id}_receiver")
        self.receive_thread.start()
        
        with self.logger_lock:
            logger.info(f"Agent '{self.agent_id}' is now running with {self.max_workers} workers.")

        try:
            while self.running:
                self.receive_thread.join(timeout=1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the agent gracefully."""
        if not self.running:
            return

        with self.logger_lock:
            logger.info(f"Stopping agent '{self.agent_id}'...")
        
        self.running = False

        # Wait for all pending tasks to complete
        for future in self.futures:
            try:
                future.result(timeout=5)
            except Exception as e:
                with self.logger_lock:
                    logger.error(f"Error waiting for task completion: {e}")

        try:
            self.thread_pool.shutdown(wait=True, cancel_futures=True)
            with self.logger_lock:
                logger.info("Thread pool shut down.")
        except Exception as e:
            with self.logger_lock:
                logger.error(f"Error shutting down thread pool: {e}")

        try:
            if self.consumer:
                self.consumer.close()
                with self.logger_lock:
                    logger.info("Kafka consumer closed.")
        except Exception as e:
            with self.logger_lock:
                logger.error(f"Error closing Kafka consumer: {e}")

        try:
            if self.producer:
                self.producer.close(timeout=5)
                with self.logger_lock:
                    logger.info("Kafka producer closed.")
        except Exception as e:
            with self.logger_lock:
                logger.error(f"Error closing Kafka producer: {e}")

        try:
            if self.receive_thread.is_alive():
                self.receive_thread.join(timeout=5)
                with self.logger_lock:
                    logger.info("Receive thread terminated.")
        except Exception as e:
            with self.logger_lock:
                logger.error(f"Error stopping receive thread: {e}")

        with self.logger_lock:
            logger.info(f"Agent '{self.agent_id}' has been stopped.")

    def setup_task_logger(self):
        """Set up a thread-safe logger for task processing."""
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