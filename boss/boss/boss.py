from dotenv import load_dotenv

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

from boss.boss_prompts import BossPrompts
from boss.events import shutdown_event
from boss.models import (
    AgentSelectionAnalysis,
    NecessityCheckResponse,
    StepEstimationResponse,
    TaskEvaluationResponse,
    TaskState,
)
from boss.utils import serialize_task, serialize_task_to_string

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BOSS:
    def __init__(
        self,
        db_uri="mongodb://localhost:27017/",
        kafka_bootstrap_servers="localhost:9092",
        check_interval=1,
        openai_api_key=None,
    ):
        self.prompts = BossPrompts()
        self.threads = []

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

        except ServerSelectionTimeoutError as err:
            logger.error(f"Error: Could not connect to MongoDB server: {err}")
            raise

        self.result_consumer = KafkaConsumer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
            group_id="boss_group",
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
            shutdown_event.wait(timeout=self.check_interval)

    def _add_additional_steps(
        self, task: Dict, evaluation_response: TaskEvaluationResponse
    ) -> None:
        """
        Determine if additional steps are necessary and add them to the task only if they are truly needed.
        """
        try:
            task_id = str(task["_id"])
            current_steps = task.get("steps", [])

            if not evaluation_response.additional_steps_needed:
                logger.debug(f"No additional steps needed for task {task_id}")
                return

            # Gather all steps and their states
            all_steps_info = "\n".join(
                [
                    f"Step {step['step_number']}: {step['step_description']} - State: {step['state']}"
                    for step in current_steps
                ]
            )

            # Gather results from completed steps
            completed_steps = [
                step
                for step in current_steps
                if step.get("state") == TaskState.COMPLETED_STEP.value
            ]
            completed_steps_results = "\n".join(
                [
                    f"Step {step['step_number']} Result: {step.get('result', '')}"
                    for step in completed_steps
                ]
            )

            # Include current progress and context in the prompt
            task_description = task.get("description", "")
            previous_step_results = completed_steps_results

            # Step 1: Check if additional steps are truly necessary
            messages = self.prompts.format_additional_steps_necessity_check(
                task_description=(
                    f"Original task: {task_description}\n\n"
                    f"Current steps and their states:\n{all_steps_info}\n\n"
                    f"Results of completed steps:\n{previous_step_results}\n\n"
                    f"Evaluation explanation: {evaluation_response.explanation}\n\n"
                    f"Based on the above, do we truly need additional steps to complete this task? Answer 'Yes' or 'No' and provide a brief justification."
                )
            )

            necessity_response = self.call_openai_api_structured(
                messages=messages,
                response_model=NecessityCheckResponse,
                model="gpt-4o-mini",  # noqa: F821
            )

            if necessity_response.is_additional_steps_needed.strip().lower() != "yes":
                logger.info(
                    f"No additional steps needed for task {task_id} according to LLM."
                )
                return

            # Step 2: Generate additional steps
            capabilities_list = self._get_agent_capabilities()

            messages = self.prompts.format_step_generation(
                task_description=(
                    f"Original task: {task_description}\n\n"
                    f"Current steps and their states:\n{all_steps_info}\n\n"
                    f"Results of completed steps:\n{previous_step_results}\n\n"
                    f"Additional steps needed as per evaluation: {evaluation_response.additional_steps_needed}\n\n"
                    f"Please generate only the minimal set of additional steps that are absolutely necessary to complete the task, without duplicating any existing steps."
                ),
                capabilities_list=capabilities_list,
            )

            estimation = self.call_openai_api_structured(
                messages=messages,
                response_model=StepEstimationResponse,
                model="gpt-4o-mini",
            )

            # Handle the case where no steps are needed
            if not estimation.estimated_steps:
                logger.info(f"No new steps were generated for task {task_id}")
                return

            # Create new step entries
            new_steps = []
            start_index = len(current_steps)

            for idx, estimate in enumerate(estimation.estimated_steps):
                step = {
                    "step_number": start_index + idx + 1,
                    "step_description": estimate.step_description,
                    "overall_plan": estimation.overall_plan,
                    "state": TaskState.CREATED.value,
                    "estimated_duration": estimate.estimated_duration_minutes,
                    "confidence_score": estimate.confidence_score,
                    "expected_outcome": estimate.expected_outcome,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                new_steps.append(step)

            # Update the task with new steps
            all_steps = current_steps + new_steps

            update_result = self.tasks_collection.update_one(
                {"_id": ObjectId(task_id)},
                {
                    "$set": {
                        "steps": all_steps,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )

            if update_result.modified_count == 0:
                logger.error(f"Failed to update task {task_id} with new steps")
                return

            logger.info(f"Added {len(new_steps)} new steps to task {task_id}")

            # Record the addition of new steps in task history
            self._record_task_history(
                task_id=ObjectId(task_id),
                action="added_steps",
                details={
                    "num_steps_added": len(new_steps),
                    "reason": necessity_response.justification,
                    "new_step_numbers": [step["step_number"] for step in new_steps],
                },
            )

        except Exception as e:
            logger.error(f"Error adding additional steps to task: {e}", exc_info=True)
            self._handle_task_failure(
                task, None, f"Failed to add additional steps: {str(e)}"
            )

    def handle_agent_result(self, result: Dict[str, Any]):
        """
        Handle the result received from an agent via Kafka with improved step validation and error handling.

        Args:
            result (Dict[str, Any]): The result dictionary containing task_id and result data
        """
        try:
            # Initialize variables that will be used in multiple scopes
            next_step_index = None
            task_id = result.get("task_id")

            # Validate basic requirements
            if not task_id:
                logger.error("Received result without task_id.")
                return

            # Fetch and validate task
            task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})
            if not task:
                logger.error(f"Task {task_id} not found in database.")
                return

            current_step_index = task.get("current_step_index")
            if current_step_index is None:
                logger.error(f"No current_step_index for task {task_id}")
                return

            steps = task.get("steps", [])
            if current_step_index >= len(steps):
                logger.error(f"Invalid current_step_index for task {task_id}")
                return

            # Get current step details
            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")
            agent_result = result.get("result", "")
            expected_outcome = current_step.get("expected_outcome", "")

            # Evaluate step result
            evaluation: TaskEvaluationResponse = self._evaluate_step_result_with_llm(
                step_description=step_description,
                step_result=agent_result,
                expected_outcome=expected_outcome,
            )

            logger.info(f"\n\nEvaluation: {evaluation}\n\n")

            if evaluation.additional_steps_needed:
                logger.info(
                    f"\n\nAdditional steps: {evaluation.additional_steps_needed}\n\n"
                )
                # self._add_additional_steps(task, evaluation)

            state = (
                TaskState.COMPLETED_STEP.value
                if evaluation.success
                else TaskState.FAILED.value
            )

            update_fields = {
                f"steps.{current_step_index}.updated_at": datetime.now(timezone.utc),
                f"steps.{current_step_index}.result": agent_result,
                f"steps.{current_step_index}.state": state,
                f"steps.{current_step_index}.metadata": result.get("metadata", {}),
            }

            if not evaluation.success:
                update_fields[f"steps.{current_step_index}.error"] = (
                    "Result does not satisfy the step description and expected outcome."
                )

            if evaluation.success:
                for idx, step in enumerate(
                    steps[current_step_index + 1 :], start=current_step_index + 1
                ):
                    if step.get("state") == TaskState.CREATED.value:
                        next_step_index = idx
                        break

            update_fields["current_step_index"] = next_step_index

            result = self.tasks_collection.update_one(
                {"_id": ObjectId(task_id)}, {"$set": update_fields}
            )

            if result.modified_count == 0:
                logger.error(f"Failed to update task {task_id} in database.")

            updated_task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})

            if next_step_index is not None:
                self._process_task_with_inference(updated_task)
            else:
                self._perform_final_evaluation(updated_task)

        except Exception as e:
            logger.error(f"Error handling agent result: {str(e)}", exc_info=True)

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
                    "overall_plan": estimation.overall_plan,
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

    def _evaluate_step_result_with_llm(
        self, step_description: str, step_result: str, expected_outcome: str = ""
    ) -> bool:
        """Evaluate step results using LLM with enhanced context"""
        try:
            messages = self.prompts.format_step_evaluation(
                step_description=f"""
                Step Description: {step_description}
                Expected Outcome: {expected_outcome}
                
                The step result should address at least some of the step requirements, but sometimes might not be exact. An attempt at solving the step or task can be considered successful.

                """,
                step_result=step_result,
            )

            evaluation = self.call_openai_api_structured(
                messages=messages,
                response_model=TaskEvaluationResponse,
                model="gpt-4o-mini",
            )

            return evaluation

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

            # Add tracking of completed and failed steps
            completed_steps = [
                step
                for step in steps
                if step.get("state") == TaskState.COMPLETED_STEP.value
            ]
            failed_steps = [
                step for step in steps if step.get("state") == TaskState.FAILED.value
            ]
            pending_steps = [
                step for step in steps if step.get("state") == TaskState.CREATED.value
            ]

            # If we have any failed steps and no more pending steps, perform final evaluation
            if failed_steps and not pending_steps:
                logger.info(
                    f"Task {task_id} has failed steps and no pending steps. Performing final evaluation."
                )
                self._perform_final_evaluation(task)
                return

            previous_steps = (
                steps[:current_step_index]
                if current_step_index and current_step_index > 0
                else []
            )

            logger.debug(f"Previous steps: {previous_steps}")

            previous_step_results = "\n".join(
                [serialize_task_to_string(step) for step in previous_steps]
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
        """Perform final evaluation of the task based on completed and failed steps."""
        try:
            task_id = str(task["_id"])
            steps = task.get("steps", [])

            # Gather all completed step results
            completed_steps = [
                step
                for step in steps
                if step.get("state") == TaskState.COMPLETED_STEP.value
            ]
            failed_steps = [
                step for step in steps if step.get("state") == TaskState.FAILED.value
            ]

            # Combine results from successful steps
            combined_results = "\n".join(
                [
                    f"Step {step.get('step_number')}: {step.get('result', '')}"
                    for step in completed_steps
                ]
            )

            # Add information about failed steps
            failed_steps_info = "\n".join(
                [
                    f"Failed Step {step.get('step_number')}: {step.get('error', '')}"
                    for step in failed_steps
                ]
            )

            evaluation_criteria = task.get("evaluation_criteria", "")
            if not evaluation_criteria:
                # If no specific criteria, use default based on step completion
                if failed_steps:
                    default_message = (
                        f"Task completed with {len(completed_steps)} successful steps "
                        f"and {len(failed_steps)} failed steps. Failed steps: {failed_steps_info}"
                    )
                    self.tasks_collection.update_one(
                        {"_id": task["_id"]},
                        {
                            "$set": {
                                "status": TaskState.FAILED.value,
                                "completion_message": default_message,
                                "updated_at": datetime.now(timezone.utc),
                            }
                        },
                    )
                    logger.info(f"Task {task_id} marked as failed due to failed steps.")
                    return
                else:
                    evaluation_criteria = "Verify that all steps completed successfully and produced expected results."

            # Use LLM to evaluate results against criteria
            messages = self.prompts.format_final_evaluation(
                task_description=task.get("description", ""),
                evaluation_criteria=evaluation_criteria,
                step_results=f"Completed Steps:\n{combined_results}\n\nFailed Steps:\n{failed_steps_info}",
            )

            evaluation = self.call_openai_api_structured(
                messages=messages,
                response_model=TaskEvaluationResponse,
                model="gpt-4o-mini",
            )

            # Update task status based on evaluation
            new_status = (
                TaskState.COMPLETED_WORKFLOW.value
                if evaluation.success
                else TaskState.FAILED.value
            )
            self.tasks_collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": new_status,
                        "completion_message": evaluation.explanation,
                        "final_evaluation": {
                            "success": evaluation.success,
                            "explanation": evaluation.explanation,
                            "completed_steps": len(completed_steps),
                            "failed_steps": len(failed_steps),
                            "evaluation_timestamp": datetime.now(timezone.utc),
                        },
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )

            logger.info(
                f"Task {task_id} final evaluation completed. Success: {evaluation.success}"
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

    def check_tasks(self):
        logger.debug("Checking tasks")
        logger.debug(f"Running: {self.running}")

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

            logger.debug(f"Tasks: {tasks}")

            for task in tasks:
                self._process_task_with_inference(task)

            shutdown_event.wait(timeout=self.check_interval)
            time.sleep(self.check_interval)

    def _create_daemon_thread(self, target, name):
        thread = threading.Thread(target=target, name=name, daemon=True)
        self.threads.append(thread)
        return thread

    # works as expected
    def start(self):
        """Start the BOSS service with enhanced thread management and health checks"""

        try:
            self.running = True
            shutdown_event.clear()

            # Verify connections before starting threads
            try:
                # Test MongoDB connection
                self.client.admin.command("ping")

                # Test Kafka connection
                self.producer.flush(timeout=5)

            except Exception as e:
                logger.error(f"Failed to verify connections: {e}")
                self.running = False
                raise

            # Create threads with enhanced error handling
            self.threads = []
            thread_targets = [
                (self.check_tasks, "TaskProcessor"),
                (self.consume_agent_results, "ResultConsumer"),
                (self._monitor_system_health, "SystemMonitor"),
            ]

            for target, name in thread_targets:
                thread = self._create_daemon_thread(target, name)
                thread.start()
        except Exception as e:
            logger.error(f"Error starting BOSS service: {e}")
            raise

        logger.info("BOSS service started successfully")

    # works as expected
    def stop(self):
        self.running = False
        shutdown_event.set()

        for thread in self.threads:
            # force kill threads after 5 seconds
            thread.join(timeout=1)

        if self.client:
            self.client.close()
        if self.producer:
            self.producer.close()
        if self.result_consumer:
            self.result_consumer.close()

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
        model: str = "gpt-4o",
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
