from dotenv import load_dotenv

from agent_scheduler import AgentScheduler
from boss_prompts import BossPrompts
from models import (
    AgentSelectionAnalysis,
    StepEstimationResponse,
    StepResult,
    TaskEvaluationResponse,
    TaskState,
)

load_dotenv()

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from kafka import KafkaConsumer, KafkaProducer
from openai import OpenAI
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from utils import serialize_task

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BOSS:
    def __init__(
        self,
        db_uri="mongodb://localhost:27017/",
        kafka_bootstrap_servers="localhost:9092",
        check_interval=10,
        openai_api_key=None,
    ):
        self.prompts = BossPrompts()
        self.threads = []
        self.shutdown_event = threading.Event()

        self.task_thread = None
        self.monitor_thread = None
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.check_interval = check_interval
        self.running = True
        self.openai_api_key = (
            openai_api_key if openai_api_key else os.getenv("OPENAI_API_KEY")
        )
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        try:
            self.client = MongoClient(db_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()
            self.db = self.client["task_db"]
            self.tasks_collection = self.db["tasks"]
            self.agents_collection = self.db["agents"]
            self.task_history_collection = self.db["task_history"]
            self.scheduler = AgentScheduler(
                self.tasks_collection, self.agents_collection
            )

        except ServerSelectionTimeoutError as err:
            logger.error(f"Error: Could not connect to MongoDB server: {err}")
            raise

        self.result_consumer = KafkaConsumer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
            group_id="boss_group",  # Add this line
        )

        # Subscribe to all agent result topics
        agent_result_topics = [
            f"results_from_{agent['agent_id']}"
            for agent in self.agents_collection.find()
        ]

        print(f"Subscribing to topics: {agent_result_topics}")
        self.result_consumer.subscribe(agent_result_topics)

    def consume_agent_results(self):
        logger.info("BOSS started listening for agent results.")
        while self.running:
            try:
                for message in self.result_consumer:
                    if not self.running:
                        break
                    result = message.value
                    logger.info(f"Received result from agent: {result}")
                    self.handle_agent_result(result)
            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Error in consume_agent_results: {e}")
            self.shutdown_event.wait(timeout=self.check_interval)

    def handle_agent_result(self, result: Dict[str, Any]):
        """Handle the result received from an agent via Kafka"""
        try:
            task_id = result.get("task_id")
            if not task_id:
                logger.error("Received result without task_id.")
                return

            task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})
            if not task:
                logger.error(f"Task {task_id} not found in database.")
                return

            # Update the current step based on the result
            current_step_index = task.get("current_step_index")
            if current_step_index is None:
                logger.error(f"No current_step_index for task {task_id}")
                return

            steps = task.get("steps", [])
            if current_step_index >= len(steps):
                logger.error(f"Invalid current_step_index for task {task_id}")
                return

            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")
            agent_result = result.get("result", "")

            # Evaluate the step result using LLM
            evaluation_success = self._evaluate_step_result_with_llm(
                step_description, agent_result
            )

            if evaluation_success:
                current_step["state"] = TaskState.COMPLETED_STEP.value
                current_step["result"] = agent_result
                current_step["updated_at"] = datetime.now(timezone.utc)
            else:
                current_step["state"] = TaskState.FAILED.value
                current_step["error"] = "Result does not satisfy the step description."
                current_step["updated_at"] = datetime.now(timezone.utc)

            # Update the task in the database
            self.tasks_collection.update_one(
                {"_id": ObjectId(task_id)},
                {
                    "$set": {
                        "steps": steps,
                        "current_step_index": None,
                        "current_agent": None,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            logger.info(f"Updated task {task_id} with result from agent.")

            # Re-fetch the task to get the updated state
            task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})

            # Handle next steps or completion
            if evaluation_success:
                self._process_task_with_inference(task)
            else:
                # Handle failure, possibly generate additional steps
                self._handle_failed_completion(task, result)

        except Exception as e:
            logger.error(f"Error handling agent result: {e}")

    def _generate_steps_from_description(self, description: str) -> List[Dict]:
        """Generate steps from task description using OpenAI LLM"""
        try:
            # Get agent capabilities
            active_agents = list(self.agents_collection.find({"status": "active"}))
            if not active_agents:
                logger.warning("No active agents available for step estimation.")
                return []

            capabilities_list = ", ".join(
                [
                    f"{agent['agent_id']}: {', '.join(agent['capabilities'])}"
                    for agent in active_agents
                ]
            )
            logger.debug(f"Aggregated Agent Capabilities: {capabilities_list}")

            # Format messages properly for OpenAI API
            messages = self.prompts.format_step_generation(
                task_description=description, capabilities_list=capabilities_list
            )

            logger.debug(f"Generated messages: {messages}")

            # Call OpenAI API with structured response
            estimation = self.call_openai_api_structured(
                messages=messages,
                response_model=StepEstimationResponse,
                model="gpt-4o",
            )

            logger.debug(f"Received estimation: {estimation}")

            steps = []
            for idx, estimate in enumerate(estimation.estimated_steps):
                step = {
                    "step_number": idx + 1,
                    "step_description": estimate.step_description,
                    "state": TaskState.CREATED.value,
                    "estimated_duration": estimate.estimated_duration_minutes,
                    "confidence_score": estimate.confidence_score,
                    "expected_outcome": estimate.expected_outcome,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                steps.append(step)

            return steps

        except Exception as e:
            logger.error(f"Error generating steps from description: {e}")
            return []

    def _generate_additional_steps(
        self, task: Dict, step_result: StepResult
    ) -> List[Dict]:
        """
        Generate additional steps to handle a failed step using OpenAI LLM
        """
        try:
            messages = self.prompts.format_step_generation(
                task_description=f"""Original task: {task.get('description', '')}
                Failed step: {step_result.step_description}
                Error: {step_result.error}
                Generate additional steps to resolve the issue.""",
                capabilities_list=self._get_agent_capabilities(),
            )

            # Call OpenAI API with structured response
            estimation = self.call_openai_api_structured(
                messages=messages,
                response_model=StepEstimationResponse,
                model="gpt-4o-mini",
            )

            additional_steps = []
            for idx, estimate in enumerate(estimation.estimated_steps):
                step = {
                    "step_number": len(task.get("steps", [])) + idx + 1,
                    "step_description": estimate.step_description,
                    "state": TaskState.CREATED.value,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                additional_steps.append(step)

            return additional_steps

        except Exception as e:
            logger.error(f"Error generating additional steps: {e}")
            return []

    def _evaluate_step_result_with_llm(
        self, step_description: str, step_result: str
    ) -> bool:
        """Evaluate step results using LLM"""
        try:
            # Use new prompt system
            messages = self.prompts.format_step_evaluation(
                step_description=step_description, step_result=step_result
            )

            # Call OpenAI API with structured response
            evaluation = self.call_openai_api_structured(
                messages=messages,
                response_model=TaskEvaluationResponse,
                model="gpt-4o-mini",
            )

            return evaluation.success

        except Exception as e:
            logger.error(f"Error evaluating step result with LLM: {e}")
            return False

    def _process_task_with_inference(self, task: Dict):
        """Process task with dynamic step expansion based on LLM predictions"""

        logger.debug("_process_task_with_inference")
        try:
            task_description = task.get("description", "")

            task_id = str(task["_id"])
            if not task_description:
                logger.error(f"Task {task_id} has no description.")
                self._request_human_intervention(task, "Task missing description.")
                return

            # Get the steps from the task
            steps = task.get("steps", [])
            # Find the current step index
            current_step_index = task.get("current_step_index")

            previous_steps = (
                steps[:current_step_index]
                if current_step_index and current_step_index > 0
                else []
            )
            previous_step_results = "\n".join(
                [step.get("result", "") for step in previous_steps]
            )

            task["previous_step_results"] = previous_step_results

            if not steps:
                # No steps have been generated yet, generate steps
                logger.info(f"No steps found for task {task_id}, generating steps.")
                steps = self._generate_steps_from_description(task_description)
                if not steps:
                    logger.error(f"Failed to generate steps for task {task_id}")
                    self._request_human_intervention(task, "Failed to generate steps.")
                    return

                # Update task in database with generated steps
                task["steps"] = steps
                task["current_step_index"] = None
                self.tasks_collection.update_one(
                    {"_id": task["_id"]},
                    {"$set": {"steps": steps, "current_step_index": None}},
                )

            # If current_step_index is None, find the next pending step
            if current_step_index is None:
                for idx, step in enumerate(steps):
                    if step.get("state") == TaskState.CREATED.value:
                        current_step_index = idx
                        break
                else:
                    # No pending steps, task is completed
                    logger.info(f"All steps completed for task {task_id}")
                    self._perform_final_evaluation(task)
                    return

            # Get the current step
            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")
            if not step_description:
                logger.error(f"No step description found for task {task_id}")
                self._request_human_intervention(task, "No step description found.")
                return

            # Assign task to suitable agent
            agent_details = self._get_agent_details()
            messages = self.prompts.format_agent_selection(
                step_description=step_description, agent_details=agent_details
            )

            agent_analysis = self.call_openai_api_structured(
                messages=messages,
                response_model=AgentSelectionAnalysis,
                model="gpt-4o-mini",
            )

            agent_id = (
                agent_analysis.agent_id
                if agent_analysis.overall_match_score > 0
                else None
            )

            if agent_id:
                logger.info(f"Task {task_id} assigned to agent {agent_id}")
                # Update current agent and step state in task
                steps[current_step_index]["state"] = TaskState.IN_PROGRESS.value
                steps[current_step_index]["updated_at"] = datetime.now(timezone.utc)
                task["current_agent"] = agent_id
                task["current_step_index"] = current_step_index
                task["steps"] = steps
                task["status"] = TaskState.IN_PROGRESS.value
                task["updated_at"] = datetime.now(timezone.utc)

                # Update task in database
                self.tasks_collection.update_one(
                    {"_id": ObjectId(task["_id"])},
                    {
                        "$set": {
                            "current_agent": agent_id,
                            "current_step_index": current_step_index,
                            "steps": steps,
                            "status": TaskState.IN_PROGRESS.value,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )
                # Send updated task to agent
                self._send_task_to_agent_with_retry(agent_id, task)
            else:
                logger.warning("No suitable agents found for step description")

        except Exception as e:
            logger.error(f"Error in task processing: {e}")
            self._handle_task_failure(task, None, str(e))

    def _perform_final_evaluation(self, task: Dict):
        """
        Perform a final evaluation of the task based on the evaluation_criteria using the LLM.
        """
        try:
            task_id = str(task["_id"])
            evaluation_criteria = task.get("evaluation_criteria")
            if not evaluation_criteria:
                # No evaluation criteria set, mark task as completed
                self.tasks_collection.update_one(
                    {"_id": task["_id"]},
                    {"$set": {"status": TaskState.COMPLETED_WORKFLOW.value}},
                )
                logger.info(
                    f"Task {task_id} completed successfully without evaluation criteria."
                )
                return

            # Aggregate results from all successful steps
            steps = task.get("steps", [])
            successful_steps = [
                step
                for step in steps
                if step.get("state") == TaskState.COMPLETED_STEP.value
            ]
            combined_results = "\n".join(
                [step.get("result", "") for step in successful_steps]
            )

            # Use LLM to evaluate combined results against evaluation criteria
            evaluation_success = self._evaluate_task_result_with_llm(
                evaluation_criteria, combined_results
            )

            if evaluation_success:
                # Mark task as completed successfully
                self.tasks_collection.update_one(
                    {"_id": task["_id"]},
                    {"$set": {"status": TaskState.COMPLETED_WORKFLOW.value}},
                )
                logger.info(
                    f"Task {task_id} passed final evaluation and completed successfully."
                )
            else:
                # Mark task as failed
                self.tasks_collection.update_one(
                    {"_id": task["_id"]}, {"$set": {"status": TaskState.FAILED.value}}
                )
                logger.info(f"Task {task_id} failed final evaluation.")
                # Optionally, request human intervention
                self._request_human_intervention(
                    task, "Task failed final evaluation against evaluation criteria."
                )

        except Exception as e:
            logger.error(f"Error during final task evaluation: {e}")
            self._handle_task_failure(task, None, str(e))

    def _evaluate_task_result_with_llm(
        self, evaluation_criteria: str, combined_results: str
    ) -> bool:
        """Evaluate final task results using LLM"""
        try:
            # Use new prompt system
            messages = self.prompts.format_final_evaluation(
                task_description="",  # Need to pass task description from context
                evaluation_criteria=evaluation_criteria,
                step_results=combined_results,
            )

            # Call OpenAI API with structured response
            evaluation = self.call_openai_api_structured(
                messages=messages,
                response_model=TaskEvaluationResponse,
                model="gpt-4o-mini",
            )

            return evaluation.success

        except Exception as e:
            logger.error(f"Error evaluating task result with LLM: {e}")
            return False

    def _handle_successful_completion(self, task: Dict, evaluation: Dict):
        """Handle successful completion of a step"""
        task_id = str(task["_id"])
        current_step_index = task.get("current_step_index")
        if current_step_index is None:
            logger.error(f"No current_step_index for task {task_id}")
            return

        steps = task.get("steps", [])
        if current_step_index >= len(steps):
            logger.error(f"Invalid current_step_index for task {task_id}")
            return

        current_step = steps[current_step_index]
        # Update the step with the result
        current_step["state"] = TaskState.COMPLETED_STEP.value
        current_step["result"] = evaluation.get("result", "")
        current_step["updated_at"] = datetime.now(timezone.utc)

        # Clear current_step_index
        self.tasks_collection.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "steps": steps,
                    "current_step_index": None,
                    "current_agent": None,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        logger.info(f"Step {current_step_index} completed for task {task_id}")

    def _handle_failed_completion(self, task: Dict, evaluation: Dict):
        """Handle failed completion of a step and generate additional steps if necessary"""
        task_id = str(task["_id"])
        current_step_index = task.get("current_step_index")
        if current_step_index is None:
            logger.error(f"No current_step_index for task {task_id}")
            return

        steps = task.get("steps", [])
        if current_step_index >= len(steps):
            logger.error(f"Invalid current_step_index for task {task_id}")
            return

        current_step = steps[current_step_index]
        # Update the step with the error
        current_step["state"] = TaskState.FAILED.value
        current_step["error"] = evaluation.get("error", "")
        current_step["updated_at"] = datetime.now(timezone.utc)

        # Clear current_step_index
        self.tasks_collection.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "steps": steps,
                    "current_step_index": None,
                    "current_agent": None,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        logger.info(f"Step {current_step_index} failed for task {task_id}")

        # Generate additional steps to handle the failure
        logger.info(f"Generating additional steps for task {task_id} due to failure.")
        step_result = StepResult(
            step_description=current_step.get("step_description", ""),
            success=False,
            result=None,
            error=evaluation.get("error", ""),
            execution_time=None,
            metadata=evaluation,
        )
        additional_steps = self._generate_additional_steps(task, step_result)

        if additional_steps:
            logger.info(f"Adding additional steps to task {task_id}")
            steps.extend(additional_steps)
            self.tasks_collection.update_one(
                {"_id": task["_id"]}, {"$set": {"steps": steps}}
            )
        else:
            logger.warning(f"No additional steps generated for task {task_id}")
            self._request_human_intervention(
                task, "No additional steps generated to handle failure"
            )

    def _send_task_to_agent_with_retry(
        self, agent_id: str, task: Dict, max_retries: int = 3
    ):
        """Send task to agent via Kafka with retry logic"""
        topic = f"tasks_for_{agent_id}"
        serialized_task = serialize_task(task)

        try:
            # Send with a timeout
            future = self.producer.send(topic, serialized_task)
            future.get(timeout=10)  # Wait for send to complete with timeout
            self.producer.flush()
            return True
        except Exception:
            logger.error(
                f"Failed to send task {task['_id']} to agent {agent_id} after {max_retries} attempts"
            )
            # Revert the task assignment
            # self._revert_task_assignment(task, agent_id)
            return False

    # works as expected
    def _record_task_history(
        self,
        task_id: ObjectId,
        action: str,
        agent_id: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        """Record task history with detailed tracking"""
        history_entry = {
            "task_id": task_id,
            "action": action,
            "timestamp": datetime.now(timezone.utc),
            "agent_id": agent_id,
            "details": details,
        }

        self.task_history_collection.insert_one(history_entry)

    # works as expected
    def _request_human_intervention(self, task: Dict, reason: str):
        """Request human intervention with detailed context"""
        self.tasks_collection.update_one(
            {"_id": ObjectId(task["_id"])},
            {
                "$set": {
                    "status": TaskState.AWAITING_HUMAN.value,
                    "human_intervention_request": {
                        "reason": reason,
                        "timestamp": datetime.now(timezone.utc),
                    },
                }
            },
        )

    def _handle_invalid_task(self, task: Dict, error_msg: str):
        """Handle tasks with invalid structure or missing required fields"""
        try:
            task_id = task.get("_id", "UNKNOWN")
            update_data = {
                "status": TaskState.FAILED.value,
                "error": f"Invalid task structure: {error_msg}",
                "updated_at": datetime.now(timezone.utc),
                "validation_failure": {
                    "timestamp": datetime.now(timezone.utc),
                    "error": error_msg,
                },
            }

            if isinstance(task_id, ObjectId):
                self.tasks_collection.update_one(
                    {"_id": task_id}, {"$set": update_data}
                )

                self._record_task_history(
                    task_id, "VALIDATION_FAILURE", None, {"error": error_msg}
                )

                self._request_human_intervention(
                    task, f"Task validation failed: {error_msg}"
                )
        except Exception as e:
            logger.error(f"Error handling invalid task: {e}")

    def check_tasks(self):
        while self.running:
            tasks = list(
                self.tasks_collection.find(
                    {
                        "status": {
                            "$in": [
                                TaskState.CREATED.value,
                            ]
                        }
                    }
                )
            )
            for task in tasks:
                self._process_task_with_inference(task)

            self.shutdown_event.wait(timeout=self.check_interval)
            time.sleep(self.check_interval)

    def _create_daemon_thread(self, target, name):
        thread = threading.Thread(target=target, name=name, daemon=True)
        self.threads.append(thread)
        return thread

    # works as expected
    def start(self):
        """Start the BOSS service with enhanced thread management"""
        self.running = True
        self.shutdown_event.clear()

        # Create threads using the new helper method
        self.task_thread = self._create_daemon_thread(self.check_tasks, "TaskProcessor")
        self.result_thread = self._create_daemon_thread(
            self.consume_agent_results, "ResultConsumer"
        )
        self.monitor_thread = self._create_daemon_thread(
            self._monitor_system_health, "SystemMonitor"
        )

        # Start threads
        for thread in self.threads:
            thread.start()

        logger.info("BOSS service started with enhanced thread management")

    def __del__(self):
        """Ensure proper cleanup on object destruction"""
        try:
            self.stop()
        except Exception as e:
            logger.error(f"Error during cleanup in __del__: {e}")

    # works as expected
    def stop(self):
        self.running = False
        self.shutdown_event.set()
        for thread in self.threads:
            if thread.is_alive():
                thread.join()
        self.client.close()
        self.producer.close()
        logger.info("Boss stopped.")

    # works as expected
    def _monitor_system_health(self):
        """Monitor system health and performance"""
        while self.running:
            try:
                # Check MongoDB connection
                self.client.admin.command("ping")

                # Check Kafka producer
                if not self.producer:
                    logger.error("Kafka producer not available")

                # Monitor task processing metrics
                metrics = self._calculate_system_metrics()
                logger.debug(f"System metrics: {metrics}")

            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")

            time.sleep(60)  # Check every minute

    # works as expected
    def _calculate_system_metrics(self) -> Dict:
        """Calculate system performance metrics"""
        try:
            return {
                "active_tasks": self.tasks_collection.count_documents(
                    {"status": TaskState.IN_PROGRESS.value}
                ),
                "waiting_tasks": self.tasks_collection.count_documents(
                    {"status": TaskState.CREATED.value}
                ),
                "active_agents": self.agents_collection.count_documents(
                    {"status": "active"}
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    # Works as expected
    def call_openai_api_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: type[BaseModel],
        model: str = "gpt-4o-mini",
    ) -> BaseModel:
        """
        Make a structured call to OpenAI API with Pydantic model parsing
        """
        try:
            # Validate message format
            if not isinstance(messages, list):
                raise ValueError("Messages must be a list")

            for msg in messages:
                if not isinstance(msg, dict):
                    raise ValueError("Each message must be a dict")
                if "role" not in msg or "content" not in msg:
                    raise ValueError("Messages must have 'role' and 'content' keys")

            # Make API call
            completion = self.openai_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=0,
                response_format=response_model,
                max_tokens=2000,
            )

            # Parse response into model
            response_text = completion.choices[0].message.content
            return response_model.model_validate_json(response_text)

        except Exception as e:
            logger.error(f"Error in structured OpenAI API call: {e}")
            raise

    # Works as expected
    def _handle_task_failure(
        self, task: Dict, last_action: Optional[Dict], error_message: str
    ):
        """Handle failed task execution with detailed error tracking"""
        retry_count = task.get("retry_count", 0)

        # Record the failure details
        failure_details = {
            "timestamp": datetime.now(timezone.utc),
            "error_message": error_message,
            "last_action": last_action,
            "retry_count": retry_count,
        }

        # Mark as failed and requiring human intervention
        self.tasks_collection.update_one(
            {"_id": ObjectId(task["_id"])},
            {
                "$set": {
                    "status": TaskState.FAILED.value,
                    "final_failure_details": failure_details,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$push": {"failure_history": failure_details},
            },
        )

        # Request human intervention
        self._request_human_intervention(
            task,
            f"Final error: {error_message}",
        )

    def _get_agent_capabilities(self) -> str:
        """Get formatted string of agent capabilities"""
        try:
            active_agents = list(self.agents_collection.find({"status": "active"}))
            return ", ".join(
                [
                    f"{agent['agent_id']}: {', '.join(agent['capabilities'])}"
                    for agent in active_agents
                ]
            )
        except Exception as e:
            logger.error(f"Error getting agent capabilities: {e}")
            return ""

    def _get_agent_details(self) -> str:
        """Get detailed agent information including workload"""
        try:
            active_agents = list(self.agents_collection.find({"status": "active"}))
            agent_details = []
            for agent in active_agents:
                # Get current workload
                current_tasks = self.tasks_collection.count_documents(
                    {
                        "current_agent": agent["agent_id"],
                        "status": TaskState.IN_PROGRESS.value,
                    }
                )

                details = (
                    f"Agent: {agent['agent_id']}\n"
                    f"Capabilities: {', '.join(agent['capabilities'])}\n"
                    f"Current workload: {current_tasks} tasks\n"
                    f"Status: {agent['status']}\n"
                    "---"
                )
                agent_details.append(details)

            return "\n".join(agent_details)
        except Exception as e:
            logger.error(f"Error getting agent details: {e}")
            return ""
