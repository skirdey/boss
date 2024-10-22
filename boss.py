from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
import requests  # For OpenAI API calls
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from kafka import KafkaProducer
import json
from datetime import datetime
import logging
from bson import ObjectId

logger = logging.getLogger(__name__)


from datetime import datetime, timezone

def serialize_task(task):
    """
    Recursively converts ObjectId and datetime instances in a MongoDB document to strings,
    ensuring datetimes are in ISO 8601 format with 'Z' at the end.
    """
    if isinstance(task, dict):
        return {k: serialize_task(v) for k, v in task.items()}
    elif isinstance(task, list):
        return [serialize_task(element) for element in task]
    elif isinstance(task, ObjectId):
        return str(task)
    elif isinstance(task, datetime):
        # Ensure the datetime is timezone-aware and in UTC
        if task.tzinfo is None:
            task = task.replace(tzinfo=timezone.utc)
        else:
            task = task.astimezone(timezone.utc)
        # Format datetime as ISO 8601 string with 'Z' at the end
        iso_string = task.isoformat()
        if iso_string.endswith('+00:00'):
            iso_string = iso_string[:-6] + 'Z'
        elif iso_string.endswith('+00'):
            iso_string = iso_string[:-3] + 'Z'
        else:
            # If there's no timezone info, append 'Z' directly
            iso_string += 'Z'
        return iso_string
    else:
        return task

class BOSS:
    def __init__(self, db_uri='mongodb://localhost:27017/', kafka_bootstrap_servers='localhost:9092', check_interval=5, openai_api_key=None):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.check_interval = check_interval
        self.running = True
        self.openai_api_key = openai_api_key if openai_api_key is not None else os.getenv('OPENAI_API_KEY')

        try:
            self.client = MongoClient(db_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()
            self.db = self.client['task_db']
            self.tasks_collection = self.db['tasks']
            self.agents_collection = self.db['agents']
        except ServerSelectionTimeoutError as err:
            print(f"Error: Could not connect to MongoDB server: {err}")
            self.client = None
            self.tasks_collection = None
            self.agents_collection = None

    def check_tasks(self):
        while self.running:
            if self.tasks_collection is None or self.agents_collection is None:
                logger.warning("Database is not available. Skipping task checking.")
                time.sleep(self.check_interval)
                continue

            try:
                new_tasks = self.tasks_collection.find({"status": "Created"})
                for task in new_tasks:
                    agent_id = self.assign_agent(task)
                    if agent_id:
                        # Update task in MongoDB with agent_id before sending to queue
                        self.tasks_collection.update_one(
                            {"_id": task["_id"]},
                            {
                                "$set": {
                                    "status": "In Progress",
                                    "agent_id": agent_id,
                                    "updated_at": datetime.utcnow()
                                }
                            }
                        )
                        # Fetch the updated task
                        updated_task = self.tasks_collection.find_one({"_id": task["_id"]})
                        self.send_task_to_agent(agent_id, updated_task)
                    else:
                        # If no agent is found, set status to Awaiting_Human
                        self.tasks_collection.update_one(
                            {"_id": task["_id"]},
                            {
                                "$set": {
                                    "status": "Awaiting_Human",
                                    "updated_at": datetime.utcnow()
                                }
                            }
                        )
            except Exception as e:
                logger.error(f"An error occurred while checking tasks: {e}")

            time.sleep(self.check_interval)

    def assign_agent(self, task):
        # Get all active agents and their capabilities
        agents = list(self.agents_collection.find({"status": "active"}))
        if not agents:
            return None

        # Prepare the prompt for OpenAI
        agents_info = "\n".join([f"Agent ID: {agent['agent_id']}, Capabilities: {', '.join(agent['capabilities'])}" for agent in agents])

        prompt = f"""
You are an intelligent task assignment system. Your job is to assign the following task to the most suitable agent.

Rules:
1. Only assign tasks to agents that have the necessary capabilities.
2. If no agent has the necessary capabilities, set the status of the task to Awaiting_Human.

Task Description:
{task['description']}

Available Agents:
{agents_info}

Based on the task description and the agent capabilities, which Agent ID is best suited to handle this task?

Please respond with only the Agent ID or None if no agent has the necessary capabilities.
"""

        agent_id = self.call_openai_api(prompt)

        logger.info(f"Assigned agent: {agent_id} for task: {task['_id']} with description: {task['description']}")

        # Validate if the agent_id exists in the list of agents
        if agent_id and any(agent['agent_id'] == agent_id for agent in agents):
            return agent_id.strip()
        else:
            return None

    def call_openai_api(self, prompt):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.openai_api_key}'
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        agent_id = response_json['choices'][0]['message']['content']
        return agent_id.strip()

    def send_task_to_agent(self, agent_id, task):
        """
        Sends the serialized task to the specified Kafka topic.
        """
        task_serializable = serialize_task(task)  # Convert ObjectId to string
        topic = f"agent-{agent_id}"
        self.producer.send(topic, task_serializable)
        self.producer.flush()
        logger.info(f"Task sent to agent {agent_id} on topic {topic}.")

    def start(self):
        threading.Thread(target=self.check_tasks, daemon=True).start()

    def stop(self):
        self.running = False
        if self.client:
            self.client.close()
        if self.producer:
            self.producer.close()
