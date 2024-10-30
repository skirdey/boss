from bson import ObjectId
from dotenv import load_dotenv

from models import StepResult, TaskState

load_dotenv()

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from pymongo import MongoClient

from wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConversationResult(BaseModel):
    """Model for conversation results"""

    success: bool
    response: Optional[str]
    error: Optional[str]
    model: Optional[str]
    timestamp: datetime
    metrics: Dict = Field(default_factory=dict)


class WrapperConversation(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_conversation",
        db_uri="mongodb://localhost:27017/",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, db_uri, kafka_bootstrap_servers)
        self.agent_id = agent_id
        self.client = MongoClient(db_uri)
        self.db = self.client["agent_db"]
        self.collection = self.db["tasks"]
        self.task_history = self.db["task_history"]

    def process_message(self, messages: List[Dict]) -> ConversationResult:
        """Process a conversation message using Claude with improved error handling and metrics"""
        start_time = datetime.now(timezone.utc)

        try:
            # Call Claude API
            response = self.anthropic.messages.create(
                max_tokens=8192,
                messages=[*messages],
                model="claude-3-5-sonnet-20241022",
            )

            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()

            # self.task_logger.info(f"Conversation response: {response.content}")

            return ConversationResult(
                success=True,
                response=" ".join([con.text for con in response.content]),
                model="claude-3-5-sonnet-20241022",
                timestamp=end_time,
                metrics={
                    "processing_time_seconds": processing_time,
                    "token_count": sum(len(msg["content"].split()) for msg in messages),
                    "response_length": len(response.content[0].text),
                    "model_used": "claude-3-5-sonnet-20241022",
                },
                error=None,
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            self.task_logger.error(f"Error in process_message: {str(e)}")
            return ConversationResult(
                success=False,
                response=None,
                error=str(e),
                timestamp=end_time,
                metrics={
                    "processing_time_seconds": (end_time - start_time).total_seconds(),
                    "error_type": type(e).__name__,
                },
                model=None,
            )

    def process_task(self, task: Dict) -> Dict:
        """Process a task with consistent workflow state management"""
        if not isinstance(task, dict) or "_id" not in task:
            self.task_logger.error("Invalid task format")
            return {"success": False, "note": "Invalid task format"}

        try:
            # Fetch current workflow state
            workflow_state = task.get("workflow_state", {})
            remaining_steps = workflow_state.get("remaining_steps", [])
            current_step_index = int(task.get("current_step", 0))

            if current_step_index >= len(remaining_steps):
                self.task_logger.info("All steps have been completed.")
                return {
                    "task_id": str(task["_id"]),
                    "result": "All steps completed.",
                    "success": True,
                    "metrics": {},
                }

            # Get the current step
            current_step = remaining_steps[current_step_index]
            step_description = current_step["step_description"]

            self.task_logger.info(
                f"Processing step {current_step_index + 1}: {step_description}"
            )

            # Process the step (e.g., as a conversation step)
            conversation_result = self.process_message(
                [{"role": "user", "content": step_description}],
            )

            # self.task_logger.info(f"Conversation result: {conversation_result}")

            # Create step result for workflow state manager
            step_result = StepResult(
                step_description=step_description,
                success=conversation_result.success,
                result=conversation_result.response
                if conversation_result.success
                else None,
                error=conversation_result.error
                if not conversation_result.success
                else None,
                execution_time=conversation_result.metrics.get(
                    "processing_time_seconds"
                ),
                metadata={
                    "model": conversation_result.model,
                    "metrics": conversation_result.metrics,
                    "timestamp": conversation_result.timestamp.isoformat(),
                    "conversation_history": [
                        {"role": "user", "content": step_description}
                    ],
                    "task_context": task.get("context", ""),
                },
            )

            # Update workflow state
            workflow_update_success = self.workflow_state_manager.update_workflow_state(
                task_id=str(task["_id"]),
                step_result=step_result,
                agent_id=self.agent_id,
            )

            if not workflow_update_success:
                logger.error(f"Failed to update workflow state for task {task['_id']}")
                return {
                    "task_id": str(task["_id"]),
                    "error": "Failed to update workflow state.",
                    "success": False,
                }

            # If there are more steps, set status to PENDING_NEXT_STEP
            if current_step_index + 1 < len(remaining_steps):
                new_status = TaskState.PENDING_NEXT_STEP
            else:
                new_status = TaskState.COMPLETED_WORKFLOW

            self.collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": new_status,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )

            # If the workflow is completed, perform final evaluation if needed
            if new_status == TaskState.COMPLETED_WORKFLOW:
                try:
                    final_result = self.aggregate_results(task["_id"])
                    self.collection.update_one(
                        {"_id": ObjectId(task["_id"])},
                        {"$set": {"result": final_result}},  # Ensure 'result' is set
                    )
                    logger.info(f"Final result updated for task {task['_id']}")
                except Exception as e:
                    logger.error(
                        f"Failed to update final result for task {task['_id']}: {e}"
                    )

            return {
                "task_id": str(task["_id"]),
                "result": conversation_result.response
                if conversation_result.success
                else conversation_result.error,
                "success": conversation_result.success,
                "metrics": conversation_result.metrics,
            }
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")

    def start(self):
        """Start the conversation agent with proper logging"""
        logger.info(f"Starting {self.agent_id}")
        super().start()
