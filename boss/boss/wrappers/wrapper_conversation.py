import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from boss.wrappers.wrapper_agent import WrapperAgent

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
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.agent_id = agent_id

        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def process_message(self, messages: List[Dict]) -> ConversationResult:
        """Process a conversation message using Claude with improved error handling and metrics"""
        start_time = datetime.now(timezone.utc)

        try:
            # Call Claude API
            response = self.anthropic.messages.create(
                max_tokens=8192,
                messages=[*messages],
                model="claude-3-5-haiku-20241022",
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
            # Fetch steps and current step index
            steps = task.get("steps", [])
            current_step_index = task.get("current_step_index")

            # Build context from previous steps
            previous_steps = (
                steps[:current_step_index] if current_step_index > 0 else []
            )
            previous_step_results = "\n".join(
                [
                    f"Step {i+1} Result: {step.get('result', '')}"
                    for i, step in enumerate(previous_steps)
                    if step.get("result")
                ]
            )

            # Get current step
            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")

            # Build message with context
            messages = []

            messages.append(
                {
                    "role": "user",
                    "content": f"Previous step results:\n{previous_step_results}\n---\nCurrent step task:\n{step_description}\n\nProvide a new response that builds upon previous results and addresses this specific step.",
                }
            )

            # Process the step
            conversation_result = self.process_message(messages)

            # Create step result
            step_result = {
                "task_id": str(task["_id"]),
                "agent_id": self.agent_id,
                "step_description": step_description,
                "success": conversation_result.success,
                "result": conversation_result.response
                if conversation_result.success
                else None,
                "error": conversation_result.error
                if not conversation_result.success
                else None,
                "execution_time": conversation_result.metrics.get(
                    "processing_time_seconds"
                ),
                "metadata": {
                    "model": conversation_result.model,
                    "metrics": conversation_result.metrics,
                    "timestamp": conversation_result.timestamp.isoformat(),
                    "conversation_history": messages,
                    "task_context": task.get("context", ""),
                },
            }

            return step_result

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return {"success": False, "error": str(e)}
