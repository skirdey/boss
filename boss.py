import signal
import traceback

from dotenv import load_dotenv

from agent_scheduler import AgentScheduler
from models import (
    StepEstimationResponse,
    StepResult,
    TaskState,
)
from workflow_state_manager import WorkflowStateManager

load_dotenv()

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from bson import ObjectId
from kafka import KafkaProducer
from openai import OpenAI
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from utils import ensure_timezone_aware, serialize_task

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BOSS:
    def __init__(
        self,
        db_uri="mongodb://localhost:27017/",
        kafka_bootstrap_servers="localhost:9092",
        check_interval=10,
        openai_api_key=None,
        max_retries=3,
        retry_delay=5,
    ):
        self.threads = []
        self.shutdown_event = threading.Event()

        self.is_shutting_down = False

        def signal_handler(signum, frame):
            import sys

            if self.is_shutting_down:
                logger.warning("Received additional shutdown signal, forcing exit...")
                os._exit(1)  # Force exit if we get a second signal

            logger.info(f"Received signal {signum}")
            self.shutdown_event.set()
            self.stop()
            # Exit the program after clean shutdown
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.task_thread = None
        self.monitor_thread = None
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.check_interval = check_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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

        try:
            self.scheduler = AgentScheduler(
                self.tasks_collection, self.agents_collection
            )
            self.workflow_manager = WorkflowStateManager(
                self.tasks_collection,
                self.task_history_collection,
                self.agents_collection,
            )
            logger.debug("WorkflowStateManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowStateManager: {e}")
            raise

    def _generate_step_estimates(self, task: Dict) -> Dict:
        """
        Generate detailed step estimates for task execution based on available agents and their capabilities.
        Utilizes structured output parsing with OpenAI to ensure steps are aligned with agent capabilities.
        """
        logger.debug(
            "Starting step estimation with consideration of agent capabilities."
        )

        # Step 1: Retrieve Available Agents and Their Capabilities
        try:
            active_agents = list(self.agents_collection.find({"status": "active"}))
            if not active_agents:
                logger.warning("No active agents available for step estimation.")
                return self._generate_fallback_estimation("No active agents available.")

            capabilities_list = ", ".join(
                [
                    f"{agent['agent_id']}: {', '.join(agent['capabilities'])}"
                    for agent in active_agents
                ]
            )
            logger.debug(f"Aggregated Agent Capabilities: {capabilities_list}")

        except Exception as e:
            logger.error(f"Error retrieving agent capabilities: {e}")
            return self._generate_fallback_estimation(
                "Error retrieving agent capabilities."
            )

        # Step 2: Modify the OpenAI Prompt to Include Agent Capabilities
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that generates detailed step estimates for agentic LLM workflow tasks. "
                    "Your job is to keep it simple and to the minimum number of steps required. "
                    "Assume that the agent will be running in a command line or some other agent interface, "
                    "so the steps should be things that a human would do in a command line, like 'ping', 'curl', 'ssh', 'scp', etc. "
                    "Each step must include 'fallback_steps' and 'validation_criteria'. "
                    "Provide the response in JSON format matching the StepEstimationResponse model."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the following task and estimate the sequence of steps needed based on the available agents and their capabilities.\n\n"
                    f"Task Description: {task['description']}\n"
                    f"Task Complexity: {task.get('complexity', 'unknown')}\n"
                    f"Context: {task.get('context', 'No additional context provided')}\n"
                    f"Available Agents and Capabilities: {capabilities_list}"
                ),
            },
        ]

        model = "gpt-4o"

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=StepEstimationResponse,
                temperature=0,
            )

            estimation = completion.choices[0].message.parsed

            # logger.debug(f"Step Estimation: {pformat(estimation.model_dump())}")

            # Add metadata
            estimation_dict = estimation.model_dump()
            estimation_dict["estimation_model_version"] = model
            estimation_dict["last_updated"] = datetime.now(timezone.utc)

            return estimation_dict

        except Exception as e:
            logger.error(f"Unexpected error in step estimation: {e}")
            return self._generate_fallback_estimation("Error during step estimation.")

    def _generate_fallback_estimation(self, reason: Optional[str] = None) -> Dict:
        """
        Generate a basic fallback estimation when main estimation fails or no agents are available.
        Optionally include a reason for the fallback.
        """
        fallback_steps = {
            "conversation": {
                "step_description": "Conduct a conversation to gather more information",
                "estimated_duration_minutes": 5,
                "confidence_score": 0.9,
                "expected_outcome": "Gathered necessary information through conversation.",
                "fallback_steps": ["Review task description for missing details"],
                "validation_criteria": [
                    "Conversation logs are comprehensive and relevant."
                ],
            },
            # Add more fallback steps as needed
        }

        fallback_estimation = {
            "estimated_steps": [
                fallback_steps.get("conversation", fallback_steps["conversation"])
            ],
            "critical_path_steps": [],
            "additional_steps": [],
            "estimation_model_version": "fallback",
            "last_updated": datetime.now(timezone.utc),
        }

        if reason:
            fallback_estimation["fallback_reason"] = reason

        # logger.debug(f"Fallback Step Estimation: {pformat(fallback_estimation)}")
        return fallback_estimation

    def _process_new_task(self, task: Dict):
        # logger.debug(f"_process_new_task: {pformat(task)}")
        try:
            # Generate step estimates
            step_estimation = self._generate_step_estimates(task)
            # logger.debug(f"Step Estimation: {pformat(step_estimation)}")

            # Create initial workflow state
            initial_workflow_state = {
                "completed_steps": [],
                "remaining_steps": step_estimation["estimated_steps"],
                "current_agent": None,
            }

            # Just update the essential fields for task processing
            task_update = {
                "workflow_state": initial_workflow_state,
                "status": TaskState.PENDING_NEXT_STEP.value,
            }

            # Update task in database
            self.tasks_collection.update_one(
                {"_id": task["_id"]}, {"$set": task_update}
            )

            # Handle additional steps if needed
            additional_steps = step_estimation.get("additional_steps", [])

            logger.debug(f"Additional steps: {additional_steps}")
            if additional_steps:
                for step in additional_steps:
                    self.workflow_manager.add_step(str(task["_id"]), step)

        except Exception as e:
            logger.error(f"Error in processing new task: {e}")
            self._handle_task_failure(task, None, str(e))

    def _process_task_with_inference(self, task: Dict):
        """Process task with appropriate agent selection and workflow state management"""

        logger.debug("_process_task_with_inference")
        try:
            task_description = task.get("description", "")
            task_id = str(task["_id"])
            if not task_description:
                logger.error(f"Task {task_id} has no description.")
                self._request_human_intervention(task, "Task missing description.")
                return

            # Get current step info from workflow manager
            current_step = self.workflow_manager.get_current_step(task_id)

            # logger.debug(
            # f"Current step in _process_task_with_inference: {pformat(current_step)}"
            # )

            if not current_step:
                logger.info(f"No remaining steps for task {task_id}")
                return

            # Assign task to suitable agent
            step_description = current_step.get("step_description", None)

            if step_description is None or step_description == "":
                logger.error(f"No step description found for task {task_id}")
                self._request_human_intervention(task, "No step description found.")
                return

            agent_id = self.scheduler.assign_task(task_id, step_description)

            logger.debug(
                f"Agent ID for task {task_id} at step {current_step['step_description']}: {agent_id}"
            )

            if agent_id:
                logger.info(f"Task {task_id} assigned to agent {agent_id}")
                self._send_task_to_agent_with_retry(agent_id, task)
            else:
                logger.warning("No suitable agents found for step description")
                self._handle_no_available_agents(task, current_step["step_description"])

        except Exception as e:
            logger.error(f"Error in task processing: {e}")
            self._handle_task_failure(task, None, str(e))

    def _handle_successful_completion(self, task: Dict, evaluation: Dict):
        """Delegate successful completion handling to workflow manager"""
        step_result = StepResult(
            step_description=task.get("description", "unknown"),
            success=True,
            result=evaluation.get("reasoning"),
            error=None,
            execution_time=None,
            metadata=evaluation,
        )

        success = self.workflow_manager.update_workflow_state(
            str(task["_id"]), step_result, task.get("current_agent")
        )

        if not success:
            logger.error(f"Failed to update workflow state for task {task['_id']}")
            self._handle_task_failure(task, None, "Failed to update workflow state")

    def _handle_failed_completion(self, task: Dict, evaluation: Dict):
        """Delegate failed completion handling to workflow manager"""
        step_result = StepResult(
            step_description=task.get("description", "unknown"),
            success=False,
            result=None,
            error=evaluation.get("reasoning"),
            execution_time=None,
            metadata=evaluation,
        )

        success = self.workflow_manager.update_workflow_state(
            str(task["_id"]), step_result, task.get("current_agent")
        )

        if not success:
            logger.error(f"Failed to update workflow state for task {task['_id']}")
            self._request_human_intervention(task, "Failed to update workflow state")

    def aggregate_results(self, task_id: str) -> str:
        """
        Aggregate and synthesize results from all completed steps into a final answer.
        Combines aggregation and synthesis into a single function for better efficiency.

        Args:
            task_id (str): The ID of the task to aggregate results for

        Returns:
            str: Synthesized final answer combining all step outcomes
        """
        task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})
        if not task:
            logger.error(f"Task {task_id} not found for aggregation.")
            return "Aggregation failed: Task not found."

        completed_steps = task.get("workflow_state", {}).get("completed_steps", [])
        if not completed_steps:
            return "No completed steps found for synthesis."

        # Gather context and outcomes
        step_outcomes = []
        for step in completed_steps:
            outcome = step.get("actual_outcome", "")
            context = {
                "step_number": step.get("step_number"),
                "evaluation": step.get("evaluation", {}),
                "completion_time": step.get("completion_time"),
            }
            if outcome:
                step_outcomes.append(f"Step {context['step_number']}: {outcome}")

        aggregated_text = "\n\n".join(step_outcomes)

        try:
            # Create a structured prompt for synthesis
            synthesis_prompt = f"""
            Synthesize the following step outcomes into a coherent final answer.
            Include key findings and ensure all important details are preserved.
            
            Original Task Description: {task.get('description', 'No description available')}
            
            Step Outcomes:
            {aggregated_text}
            
            Please provide a clear, concise synthesis that:
            1. Summarizes the main results
            2. Highlights key findings
            3. Maintains important technical details
            4. Presents information in a logical order
            """

            # Get synthesis using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise assistant that synthesizes technical results clearly and accurately.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=8192,
            )

            synthesized_result = response.choices[0].message.content.strip()

            # Update both 'final_synthesis' and 'result' fields
            self.tasks_collection.update_one(
                {"_id": ObjectId(task_id)},
                {
                    "$set": {
                        "final_synthesis": {
                            "content": synthesized_result,
                            "generated_at": datetime.now(timezone.utc),
                            "source_steps": len(completed_steps),
                        },
                        "result": synthesized_result,  # Add this line
                    }
                },
            )

            return synthesized_result

        except Exception as e:
            logger.error(f"Error in result synthesis for task {task_id}: {e}")
            # Fallback to returning aggregated text if synthesis fails
            return f"Synthesis failed. Raw aggregated results:\n\n{aggregated_text}"

    # works as expected
    def _get_task_state(self, task: Dict) -> Dict:
        """Get comprehensive current state of the task"""
        return {
            "status": task.get("status"),
            "current_step": task.get("current_step"),
            "completed_steps": task.get("workflow_state", {}).get(
                "completed_steps", []
            ),
            "remaining_steps": task.get("workflow_state", {}).get(
                "remaining_steps", []
            ),
            "last_action": self.task_history_collection.find_one(
                {"task_id": ObjectId(task["_id"])}, sort=[("timestamp", -1)]
            ),
        }

    def _handle_no_available_agents(self, task: Dict, step_description: str):
        """Handle situation when no suitable agents are available"""
        self.tasks_collection.update_one(
            {"_id": ObjectId(task["_id"])},
            {
                "$set": {
                    "status": TaskState.AWAITING_HUMAN.value,
                    "updated_at": datetime.now(timezone.utc),
                    "notes": f"No available agents for step_type: {step_description}",
                    "retry_scheduled": datetime.now(timezone.utc)
                    + timedelta(minutes=self.retry_delay),
                }
            },
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

    def _revert_task_assignment(self, task: Dict, agent_id: str):
        """Revert task assignment if sending to agent fails"""
        try:
            logger.info(
                f"Reverting assignment of task {task['_id']} from agent {agent_id}"
            )

            # Update task status back to pending
            self.tasks_collection.update_one(
                {"_id": ObjectId(task["_id"])},
                {
                    "$set": {
                        "status": TaskState.PENDING_NEXT_STEP.value,
                        "updated_at": datetime.now(timezone.utc),
                    },
                    "$unset": {"agent_id": "", "assignment_details": ""},
                },
            )

            # Update agent load
            self.agents_collection.update_one(
                {"agent_id": agent_id},
                {"$pull": {"active_tasks": {"task_id": ObjectId(task["_id"])}}},
            )

            # Record in history
            self._record_task_history(
                task["_id"],
                "ASSIGNMENT_REVERTED",
                agent_id,
                {"reason": "Failed to send task to agent"},
            )

        except Exception as e:
            logger.error(f"Error reverting task assignment: {e}")

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
                        "context": self._get_task_state(task),
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

    def _handle_stale_task(self, task: Dict):
        """Handle tasks that have been in the system too long"""
        try:
            self.tasks_collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": TaskState.AWAITING_HUMAN.value,
                        "updated_at": datetime.now(timezone.utc),
                        "stale_task": {
                            "detected_at": datetime.now(timezone.utc),
                            "age_hours": (
                                datetime.now(timezone.utc)
                                - ensure_timezone_aware(task["created_at"])
                            ).total_seconds()
                            / 3600,
                        },
                    }
                },
            )

            self._record_task_history(
                task["_id"],
                "MARKED_STALE",
                None,
                {
                    "age_hours": (
                        datetime.now(timezone.utc)
                        - ensure_timezone_aware(task["created_at"])
                    ).total_seconds()
                    / 3600
                },
            )

            self._request_human_intervention(task, "Task has become stale")
        except Exception as e:
            logger.error(f"Error handling stale task: {e}")

    def check_tasks(self):
        """
        Main task processing loop that handles different task states and prioritizes processing.
        Implements task validation, prioritization, parallel processing, and adaptive scheduling.
        """
        logger.info("Starting task processing loop...")

        while not self.shutdown_event.is_set():
            try:
                # Process tasks in priority order for each state
                for task_state in [
                    TaskState.CREATED.value,
                    TaskState.PENDING_NEXT_STEP.value,
                    TaskState.COMPLETED_WORKFLOW.value,
                ]:
                    # Fixed query structure
                    if self.shutdown_event.is_set():
                        break

                    query = {"status": task_state}

                    tasks = list(
                        self.tasks_collection.find(query).sort([("created_at", 1)])
                    )

                    # logger.debug(
                    #     f"check_tasks: Processing tasks: {pformat(tasks)} for state: {task_state}"
                    # )

                    for task in tasks:
                        # logger.debug(f"task: {pformat(task)}")

                        if not self.running:
                            return

                        try:
                            # Process based on task state

                            if task_state == TaskState.CREATED.value:
                                # logger.debug(
                                #     f"task state is: {task_state} and TaskState is {TaskState.CREATED.value}"
                                # )

                                logger.debug("Task is just created")
                                self._process_new_task(task)
                            elif task_state == TaskState.PENDING_NEXT_STEP.value:
                                # Check if task has necessary step information
                                if not task.get("workflow_state", {}).get(
                                    "remaining_steps"
                                ):
                                    logger.error(
                                        f"Task {task['_id']} missing step information"
                                    )
                                    self._request_human_intervention(
                                        task, "Missing step information"
                                    )
                                    continue
                                self._process_task_with_inference(task)

                        except Exception as e:
                            logger.error(
                                f"Error processing task {task['_id']}: {e}\n{traceback.format_exc()}"
                            )
                            self._handle_task_failure(task, None, str(e))
                            continue

            except Exception as e:
                if not self.shutdown_event.is_set():
                    logger.error(f"Error in task processing loop: {e}")
                    continue
                else:
                    logger.info("Task processing loop shutting down...")
                    break

            self.shutdown_event.wait(timeout=self.check_interval)

        logger.info("Task processing loop ended")

    def _evaluate_final_task(self, task: Dict):
        """Perform final evaluation of the task after all steps are completed."""
        try:
            logger.info(f"Performing final evaluation for task {task['_id']}")

            # Aggregate results from all completed steps
            final_result = self.workflow_manager.aggregate_results(task["_id"])

            # Update task with final result
            self.tasks_collection.update_one(
                {"_id": ObjectId(task["_id"])},
                {
                    "$set": {
                        "final_evaluation": {
                            "result": final_result,
                            "evaluated_at": datetime.now(timezone.utc),
                        },
                        "status": TaskState.FINAL_COMPLETION.value,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )

            # Record in task history
            self.workflow_manager._record_task_history(
                task_id=str(task["_id"]),
                action="TASK_COMPLETED",
                details={"final_result": final_result},
            )

            logger.info(f"Final evaluation completed for task {task['_id']}")

        except Exception as e:
            logger.error(f"Error in final task evaluation for {task['_id']}: {e}")
            self._handle_task_failure(task, None, f"Final evaluation failed: {str(e)}")

    def _create_daemon_thread(self, target, name):
        thread = threading.Thread(target=target, name=name, daemon=True)
        self.threads.append(thread)
        return thread

    # works as expected
    def start(self):
        """Start the BOSS service with improved thread management"""
        self.running = True
        self.shutdown_event.clear()

        # Create threads using the new helper method
        self.task_thread = self._create_daemon_thread(self.check_tasks, "TaskProcessor")
        self.monitor_thread = self._create_daemon_thread(
            self._monitor_system_health, "SystemMonitor"
        )

        # Start threads
        for thread in self.threads:
            thread.start()

        logger.info("BOSS service started with enhanced thread management")

    def __del__(self):
        """Ensure proper cleanup on object destruction"""
        if not self.is_shutting_down:
            try:
                self.stop()
            except Exception as e:
                logger.error(f"Error during cleanup in __del__: {e}")

    # works as expected
    def stop(self):
        """Stop the BOSS service gracefully with improved shutdown handling"""
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress, skipping...")
            return

        self.is_shutting_down = True
        logger.info("Initiating BOSS service shutdown...")
        self.running = False
        self.shutdown_event.set()

        def cleanup_resources():
            if hasattr(self, "producer") and self.producer:
                try:
                    logger.info("Closing Kafka producer...")
                    self.producer.flush(timeout=5)
                    self.producer.close(timeout=5)
                except Exception as e:
                    logger.error(f"Error closing Kafka producer: {e}")

            if hasattr(self, "client") and self.client:
                try:
                    logger.info("Closing MongoDB client...")
                    self.client.close()
                except Exception as e:
                    logger.error(f"Error closing MongoDB client: {e}")

        try:
            # Set timeout for thread completion
            thread_timeout = 10  # seconds

            # Stop task processing threads
            if self.task_thread and self.task_thread.is_alive():
                logger.info("Waiting for task thread to complete...")
                self.task_thread.join(timeout=thread_timeout)
                if self.task_thread.is_alive():
                    logger.warning("Task thread did not terminate within timeout")

            if self.monitor_thread and self.monitor_thread.is_alive():
                logger.info("Waiting for monitor thread to complete...")
                self.monitor_thread.join(timeout=thread_timeout)
                if self.monitor_thread.is_alive():
                    logger.warning("Monitor thread did not terminate within timeout")

            # Stop all remaining threads
            for thread in self.threads:
                if thread.is_alive():
                    logger.info(f"Waiting for thread {thread.name} to complete...")
                    thread.join(timeout=thread_timeout)
                    if thread.is_alive():
                        logger.warning(
                            f"Thread {thread.name} did not terminate within timeout"
                        )

        except Exception as e:
            logger.error(f"Error during thread shutdown: {e}")
        finally:
            cleanup_resources()
            logger.info("BOSS service stopped")

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
        self, prompt: str, response_model: type[BaseModel]
    ) -> BaseModel:
        """
        Make a structured call to OpenAI API with Pydantic model parsing

        Args:
            prompt (str): The prompt to send to OpenAI
            response_model (type[BaseModel]): The Pydantic model class to parse the response into

        Returns:
            BaseModel: Instance of the provided response_model containing the parsed response
        """
        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that provides structured responses.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=response_model,
                temperature=0,
            )

            return completion.choices[0].message.parsed

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

        if retry_count < self.max_retries:
            # Attempt retry with exponential backoff
            next_retry_delay = self.retry_delay * (2**retry_count)
            retry_time = datetime.now(timezone.utc) + timedelta(
                seconds=next_retry_delay
            )

            self.tasks_collection.update_one(
                {"_id": ObjectId(task["_id"])},
                {
                    "$set": {
                        "status": TaskState.CREATED.value,
                        "retry_count": retry_count + 1,
                        "next_retry_time": retry_time,
                        "updated_at": datetime.now(timezone.utc),
                    },
                    "$push": {"failure_history": failure_details},
                },
            )

            # Record in history
            self._record_task_history(
                task["_id"], "FAILURE_WITH_RETRY", None, failure_details
            )

        else:
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

            # Record in history
            self._record_task_history(
                task["_id"], "FINAL_FAILURE", None, failure_details
            )

            # Request human intervention
            self._request_human_intervention(
                task,
                f"Max retries ({self.max_retries}) exceeded. Final error: {error_message}",
            )
