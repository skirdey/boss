from abc import abstractmethod
import os
import colorlog
from kafka import KafkaConsumer
from kafka import KafkaProducer
import threading
import json
import requests  # For OpenAI API calls
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class WrapperAgent:
    def __init__(self, agent_id, db_uri='mongodb://localhost:27017/', kafka_bootstrap_servers='localhost:9092'):
        self.agent_id = agent_id
        self.topic = f"agent-{self.agent_id}"
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.client = MongoClient(db_uri)
        self.db = self.client['task_db']
        self.tasks_collection = self.db['tasks']
        self.running = True

    def setup_task_logger(self):
        # Remove any existing handlers
        if self.task_logger.hasHandlers():
            self.task_logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s%(reset)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG':    'blue',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            }
        )
        handler.setFormatter(formatter)
        self.task_logger.addHandler(handler)
        self.task_logger.setLevel(logging.INFO)

    def receive(self):
        logger.info(f"{self.agent_id} started listening for tasks.")
        for message in self.consumer:
            task = message.value

            logger.info(f"Received task: {task}")

            # Check that the task is addressed to this agent
            if task.get("agent_id") != self.agent_id:
                continue  # Ignore tasks not addressed to this agent

            # Check that the task has evaluation criteria set
            if "evaluation_criteria" not in task or not task["evaluation_criteria"]:
                # Update task status to "Awaiting_Human"
                logger.info(f"Evaluation criteria missing for task: {task}")
                self.tasks_collection.update_one(
                    {"_id": ObjectId(task["_id"])},
                    {
                        "$set": {
                            "status": "Awaiting_Human",
                            "updated_at": datetime.now(timezone.utc),
                            "note": "Evaluation criteria missing"
                        }
                    }
                )
                continue  # Skip processing this task

            logger.info(f"Processing task: {task}")

            result = self.process_task(task)

            logger.info(f"Processing result: {result}")

            # Update task status based on processing result
            if result["success"]:
                status = "Completed_Success"
                self.tasks_collection.update_one(
                    {"_id": ObjectId(task["_id"])},
                    {
                        "$set": {
                            "status": status,
                            "result": result["result"],
                            "updated_at": datetime.now(timezone.utc),
                            "note": result.get("note", "")
                        }
                    }
                )
            else:
                # Handle retries
                retry_count = task.get("retry_count", 0)
                max_retries = task.get("max_retries", 3)
                if retry_count < max_retries:
                    # Increment retry count and re-queue the task
                    updated_retry_count = retry_count + 1
                    self.tasks_collection.update_one(
                        {"_id": ObjectId(task["_id"])},
                        {
                            "$set": {
                                "retry_count": updated_retry_count,
                                "updated_at": datetime.now(timezone.utc)
                            }
                        }
                    )
                    # Re-fetch the updated task and re-send it
                    updated_task = self.tasks_collection.find_one({"_id": task["_id"]})
                    self.producer.send(self.topic, updated_task)
                    self.producer.flush()
                else:
                    # Retries exhausted, update status
                    self.tasks_collection.update_one(
                        {"_id": ObjectId(task["_id"])},
                        {
                            "$set": {
                                "status": "Awaiting_Human",
                                "updated_at": datetime.now(timezone.utc),
                                "note": "Max retries exceeded"
                            }
                        }
                    )

            if not self.running:
                break

    def send(self, result):
        # Optionally send the result to a specific Kafka topic or handle it as needed
        self.producer.send('task-results', result)
        self.producer.flush()

    def self_evaluate(self, task):
        # Evaluate if the task is suitable for this agent
        # For example, check if the agent's capabilities match the task type
        logger.info(f"Task type: {task['type']}")
        logger.info(f"Agent capabilities: {self.get_agent_capabilities()}")

        return task["type"] in self.get_agent_capabilities()

    def get_agent_capabilities(self):
        # Retrieve agent capabilities from the database
        agent = self.db['agents'].find_one({"agent_id": self.agent_id})
        logger.info(f"Agent capabilities: {agent.get('capabilities', [])}")
        return agent.get("capabilities", []) if agent else []

    @abstractmethod
    def process_task(self, task):
        pass

    def call_openai_api(self, prompt):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}'
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()

    @abstractmethod
    def evaluate_response(self, response, criteria):
        pass

    def start(self):
        self.setup_task_logger()
        threading.Thread(target=self.receive).start()

    def stop(self):
        self.running = False
        self.consumer.close()
        self.producer.close()
        self.client.close()
