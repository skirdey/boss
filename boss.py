from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
import requests
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from kafka import KafkaProducer
import json
from datetime import datetime, timezone
import logging
from bson import ObjectId
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class TaskState:
    CREATED = "Created"
    IN_PROGRESS = "In_Progress"
    WAITING_FOR_EVALUATION = "Waiting_For_Evaluation"
    AWAITING_HUMAN = "Awaiting_Human"
    COMPLETED = "Completed"
    FAILED = "Failed"
    PENDING_NEXT_STEP = "Pending_Next_Step"

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
    def __init__(self, db_uri='mongodb://localhost:27017/', 
                 kafka_bootstrap_servers='localhost:9092', 
                 check_interval=5, 
                 openai_api_key=None):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.check_interval = check_interval
        self.running = True
        self.openai_api_key = openai_api_key if openai_api_key else os.getenv('OPENAI_API_KEY')

        try:
            self.client = MongoClient(db_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()
            self.db = self.client['task_db']
            self.tasks_collection = self.db['tasks']
            self.agents_collection = self.db['agents']
            self.task_history_collection = self.db['task_history']
            
            # Create indexes
            self.tasks_collection.create_index([("status", 1)])
            self.task_history_collection.create_index([("task_id", 1)])
            
        except ServerSelectionTimeoutError as err:
            logger.error(f"Error: Could not connect to MongoDB server: {err}")
            raise

    def check_tasks(self):
        """Main task processing loop"""
        while self.running:
            try:
                # Check for new tasks
                new_tasks = self.tasks_collection.find({"status": TaskState.CREATED})
                for task in new_tasks:
                    self._process_new_task(task)

                # Check for tasks waiting for evaluation
                eval_tasks = self.tasks_collection.find({"status": TaskState.WAITING_FOR_EVALUATION})
                for task in eval_tasks:
                    self._evaluate_task_completion(task)

                # Check for tasks pending next step
                pending_tasks = self.tasks_collection.find({"status": TaskState.PENDING_NEXT_STEP})
                for task in pending_tasks:
                    self._determine_next_step(task)

            except Exception as e:
                logger.error(f"Error in check_tasks: {e}")

            time.sleep(self.check_interval)

    def _process_new_task(self, task):
        """Process a newly created task and store step estimates"""
        # Generate step estimates without affecting processing
        step_estimation = self._generate_step_estimates(task)
        
        # Update task with estimates and initial workflow information
        self.tasks_collection.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "step_estimation": step_estimation,
                    "estimated_total_steps": len(step_estimation["estimated_steps"]),
                    "current_step": 0,
                    "workflow_state": {
                        "completed_steps": [],
                        "remaining_steps": [],
                        "current_agent": None
                    },
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Continue with normal processing
        self._determine_next_step(task)

    def _generate_step_estimates(self, task) -> Dict:
        """
        Generate estimates for the steps needed to complete the task.
        This is stored for future reference but doesn't affect actual processing.
        """
        prompt = f"""
        Analyze the following task and estimate the sequence of steps needed:
        
        Task Description: {task['description']}
        
        Provide a detailed estimation in JSON format with the following structure:
        {{
            "estimated_steps": [
                {{
                    "step_type": "one of: ping, port_scan, kali_cli, service_detection, vulnerability_scan, web_enumeration",
                    "estimated_duration_minutes": "estimated time in minutes",
                    "confidence_score": "0.0 to 1.0",
                    "expected_outcome": "what this step should achieve",
                    "potential_findings": ["possible discoveries from this step"]
                }}
            ],
            "total_estimated_duration": "total minutes",
            "estimation_confidence": "0.0 to 1.0"
        }}
        
        Consider dependencies between steps and logical progression of information gathering.
        """
        
        try:
            response = self.call_openai_api(prompt)
            estimation = json.loads(response)
            
            # Add metadata to the estimation
            estimation["estimation_timestamp"] = datetime.now(timezone.utc)
            estimation["estimation_model_version"] = "gpt-4o-mini"
            
            return estimation
            
        except Exception as e:
            logger.error(f"Error in step estimation: {e}")
            # Return a basic estimation as fallback
            return {
                "estimated_steps": [
                    {
                        "step_type": "ping",
                        "estimated_duration_minutes": 1,
                        "confidence_score": 0.9,
                        "expected_outcome": "Basic connectivity check",
                        "potential_findings": ["connectivity_status"]
                    }
                ],
                "total_estimated_duration": 1,
                "estimation_confidence": 0.9,
                "estimation_timestamp": datetime.now(timezone.utc),
                "estimation_model_version": "gpt-4o-mini"
            }

    def _evaluate_task_completion(self, task):
        """Evaluate if the current step was successful and determine next actions"""
        last_action = self.task_history_collection.find_one(
            {"task_id": task["_id"]},
            sort=[("timestamp", -1)]
        )
        
        if not last_action:
            logger.error(f"No history found for task {task['_id']}")
            return

        prompt = f"""
        Evaluate if the following task step was completed successfully:
        
        Task Description: {task['description']}
        Agent Action: {last_action['action']}
        Result: {last_action['result']}
        
        Respond with either 'SUCCESS' or 'FAILURE' followed by a brief explanation.
        """
        
        evaluation = self.call_openai_api(prompt)
        success = evaluation.startswith('SUCCESS')
        
        if success:
            # Update task status and move to next step
            self.tasks_collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": TaskState.PENDING_NEXT_STEP,
                        "current_step": task.get("current_step", 0) + 1,
                        "updated_at": datetime.now(timezone.utc)
                    },
                    "$push": {
                        "workflow_state.completed_steps": {
                            "agent_id": last_action['agent_id'],
                            "result": last_action['result'],
                            "timestamp": datetime.now(timezone.utc)
                        }
                    }
                }
            )
        else:
            # Handle failure
            self._handle_task_failure(task, last_action, evaluation)

    def _determine_next_step(self, task):
        """Determine the next step based on available agents"""
        # Get list of active agents
        available_agents = list(self.agents_collection.find({"status": "active"}))
        available_agent_ids = [agent["agent_id"] for agent in available_agents]
        
        prompt = f"""
        Given the following task and available agents, determine the next best agent to handle it.
        Only select from the available agents listed.
        
        Task Description: {task['description']}
        Available Agents: {json.dumps(available_agent_ids)}
        Task History: {json.dumps(task.get('workflow_state', {}).get('completed_steps', []))}
        
        If no suitable agent is available, respond with 'HUMAN_REVIEW'.
        Otherwise, respond with the exact agent_id from the available list.
        """
        
        next_agent = self.call_openai_api(prompt).strip()
        
        if next_agent == 'HUMAN_REVIEW' or next_agent not in available_agent_ids:
            self.tasks_collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": TaskState.AWAITING_HUMAN,
                        "updated_at": datetime.now(timezone.utc),
                        "notes": "No suitable agent available for next step"
                    }
                }
            )
        else:
            self._assign_to_agent(task, next_agent)

    def _assign_to_agent(self, task, agent_id):
        """Assign task to specific agent and send to Kafka"""
        # First check if the agent exists and is active
        agent = self.agents_collection.find_one({
            "agent_id": agent_id,
            "status": "active"
        })
        
        if not agent:
            logger.warning(f"Agent {agent_id} not found or not active. Marking task for human review.")
            self.tasks_collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": TaskState.AWAITING_HUMAN,
                        "updated_at": datetime.now(timezone.utc),
                        "notes": f"Required agent {agent_id} not available"
                    },
                    "$push": {
                        "audit_trail": {
                            "action": "agent_unavailable",
                            "agent_id": agent_id,
                            "timestamp": datetime.now(timezone.utc)
                        }
                    }
                }
            )
            return

        # Update task status and agent assignment
        self.tasks_collection.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "status": TaskState.IN_PROGRESS,
                    "agent_id": agent_id,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        # Record in history
        self.task_history_collection.insert_one({
            "task_id": task["_id"],
            "agent_id": agent_id,
            "action": "ASSIGNED",
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Send to Kafka
        self.send_task_to_agent(agent_id, task)

    def _handle_task_failure(self, task, last_action, evaluation):
        """Handle failed task execution"""
        retry_count = task.get("retry_count", 0)
        max_retries = task.get("max_retries", 3)
        
        if retry_count < max_retries:
            # Retry with same agent
            self.tasks_collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": TaskState.CREATED,
                        "retry_count": retry_count + 1,
                        "updated_at": datetime.now(timezone.utc)
                    },
                    "$push": {
                        "failure_history": {
                            "agent_id": last_action['agent_id'],
                            "evaluation": evaluation,
                            "timestamp": datetime.now(timezone.utc)
                        }
                    }
                }
            )
        else:
            # Mark as failed and requiring human intervention
            self.tasks_collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": TaskState.FAILED,
                        "failure_reason": evaluation,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )

    def _complete_task(self, task):
        """Mark task as completed and perform final updates"""
        self.tasks_collection.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "status": TaskState.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )

    def call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API with error handling and rate limiting"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.openai_api_key}'
        }
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def send_task_to_agent(self, agent_id, task):
        """Send serialized task to Kafka topic"""
        task_serializable = serialize_task(task)
        topic = f"agent-{agent_id}"
        self.producer.send(topic, task_serializable)
        self.producer.flush()
        logger.info(f"Task sent to agent {agent_id} on topic {topic}")

    def start(self):
        """Start the BOSS service"""
        threading.Thread(target=self.check_tasks, daemon=True).start()
        logger.info("BOSS service started")

    def stop(self):
        """Stop the BOSS service"""
        self.running = False
        if self.client:
            self.client.close()
        if self.producer:
            self.producer.close()
        logger.info("BOSS service stopped")