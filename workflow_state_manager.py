import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bson import ObjectId

from models import StepResult, TaskState

logger = logging.getLogger(__name__)


class WorkflowStateManager:
    def __init__(self, tasks_collection, task_history_collection, agents_collection):
        self.tasks_collection = tasks_collection
        self.task_history_collection = task_history_collection
        self.agents_collection = agents_collection
        self.logger = logging.getLogger(__name__)

    def get_current_step(self, task_id: str) -> Optional[Dict]:
        """Get the current step for a task"""
        try:
            task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})
            if not task:
                return None

            workflow_state = task.get("workflow_state", {})
            remaining_steps = workflow_state.get("remaining_steps", [])
            current_step_index = int(task.get("current_step", 0))

            if current_step_index >= len(remaining_steps):
                logger.warning(f"No remaining steps for task {task_id}")
                return None

            current_step = remaining_steps[current_step_index]
            return current_step
        except Exception as e:
            self.logger.error(f"Error getting current step: {str(e)}")
            return None

    def add_step(self, task_id: str, step: Dict) -> bool:
        """Add a new step to the workflow"""
        try:
            result = self.tasks_collection.update_one(
                {"_id": ObjectId(task_id)},
                {"$push": {"workflow_state.remaining_steps": step}},
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error adding step: {str(e)}")
            return False

    def get_workflow_state(self, task_id: str) -> Optional[Dict]:
        """Get the workflow state for a task"""
        task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})
        return task.get("workflow_state", {})

    def decrement_agent_task_count(self, agent_id: str, decrement: int = 1) -> bool:
        """Decrement agent's task count"""
        try:
            result = self.agents_collection.update_one(
                {"agent_id": agent_id},
                {
                    "$inc": {"current_tasks": -decrement},
                    "$set": {"last_updated": datetime.now(timezone.utc)},
                },
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error decrementing agent task count: {e}")
            return False

    def update_workflow_state(
        self, task_id: str, step_result: StepResult, agent_id: Optional[str] = None
    ) -> bool:
        """
        Update the workflow state of a task with the results of a completed step.

        Args:
            task_id: The MongoDB ObjectId of the task as a string
            step_result: StepResult object containing the step execution results
            agent_id: Optional ID of the agent that executed the step

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Fetch current task state
            task = self.tasks_collection.find_one({"_id": ObjectId(task_id)})
            if not task:
                self.logger.error(f"Task {task_id} not found")
                return False

            workflow_state = task.get("workflow_state", {})
            remaining_steps = workflow_state.get("remaining_steps", [])
            completed_steps = workflow_state.get("completed_steps", [])
            current_step_index = int(task.get("current_step", 0))

            if current_step_index >= len(remaining_steps):
                self.logger.error("No remaining steps to update.")
                return False

            # Get the current step
            current_step = remaining_steps[current_step_index]
            step_description = current_step["step_description"]

            # Prepare completed step data
            completed_step_data = {
                **current_step,
                "completed_at": datetime.now(timezone.utc),
                "actual_outcome": step_result.result,
                "success": step_result.success,
                "execution_time": step_result.execution_time,
                "error": step_result.error,
                "metadata": step_result.metadata,
            }

            # Determine next task status
            next_status = self._determine_next_status(
                task, remaining_steps, step_result.success
            )

            # Prepare update operation
            update_operation = {
                "$set": {
                    "status": next_status,
                    "updated_at": datetime.now(timezone.utc),
                    "current_step": current_step_index + 1,
                    "workflow_state": {
                        "completed_steps": completed_steps + [completed_step_data],
                        "remaining_steps": remaining_steps,  # Updated below
                        "current_agent": agent_id,
                        "last_inference": datetime.now(timezone.utc),
                    },
                },
                "$push": {
                    "metadata.modification_history": {
                        "timestamp": datetime.now(timezone.utc),
                        "action": "step_completed",
                        "step_description": step_description,
                        "agent_id": agent_id,
                        "status": next_status,
                    }
                },
            }

            # Remove the completed step from remaining_steps
            updated_remaining_steps = remaining_steps.copy()
            updated_remaining_steps.pop(current_step_index)

            # Update remaining_steps in the workflow_state
            update_operation["$set"]["workflow_state"]["remaining_steps"] = (
                updated_remaining_steps
            )

            # Add result if step was successful
            if step_result.success:
                self.tasks_collection.update_one(
                    {"_id": ObjectId(task_id)},
                    {"$set": {"result": step_result.result}},
                )
            else:
                self.tasks_collection.update_one(
                    {"_id": ObjectId(task_id)},
                    {"$set": {"error": step_result.error}},
                )

            # Update task document
            result = self.tasks_collection.update_one(
                {"_id": ObjectId(task_id)}, update_operation
            )

            if agent_id:
                self.decrement_agent_task_count(agent_id)

            if result.modified_count == 0:
                self.logger.warning(f"No documents were updated for task {task_id}")
                return False

            # Record in task history
            self._record_task_history(
                task_id=task_id,
                action="STEP_COMPLETED",
                agent_id=agent_id,
                details={
                    "step_description": step_description,
                    "success": step_result.success,
                    "result": step_result.result,
                    "error": step_result.error,
                    "next_status": next_status,
                },
            )

            return True

        except Exception as e:
            self.logger.error(f"Error updating workflow state: {str(e)}")
            return False

    def _determine_next_status(
        self, task: Dict, remaining_steps: List[Dict], step_success: bool
    ) -> str:
        """Determine the next task status based on current state and step result.

        Args:
            task: The task document
            remaining_steps: List of remaining workflow steps
            step_success: Whether the current step executed successfully

        Returns:
            str: The next task state from TaskState enum
        """
        if not step_success:
            retry_count = int(task.get("retry_count", 0))
            max_retries = int(task.get("max_retries", 3))

            if retry_count >= max_retries:
                return TaskState.AWAITING_HUMAN.value
            return TaskState.FAILED.value

        # Check if this was the last step
        current_step_index = int(task.get("current_step", 0))
        if (
            current_step_index >= len(remaining_steps) - 1
        ):  # Current step is the last step
            return TaskState.COMPLETED_WORKFLOW.value

        # If there are more steps, set to pending next step
        return TaskState.PENDING_NEXT_STEP.value

    def _record_task_history(
        self,
        task_id: str,
        action: str,
        agent_id: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Record an entry in the task history collection."""
        try:
            history_entry = {
                "task_id": ObjectId(task_id),
                "action": action,
                "timestamp": datetime.now(timezone.utc),
                "agent_id": agent_id,
                "details": details,
            }

            self.task_history_collection.insert_one(history_entry)
        except Exception as e:
            self.logger.error(f"Error recording task history: {str(e)}")
