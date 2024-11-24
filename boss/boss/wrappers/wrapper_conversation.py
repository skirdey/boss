from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from boss.wrappers.wrapper_agent import WrapperAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class ConversationResult(BaseModel):
    success: bool
    response: Optional[str]
    error: Optional[str]
    model: Optional[str]
    timestamp: datetime
    metrics: Dict = Field(default_factory=dict)


# Works as expected
class WrapperConversation(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_conversation",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.agent_id = agent_id
        self.anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def process_message(self, messages: List[Dict]) -> ConversationResult:
        start_time = datetime.now(timezone.utc)

        try:
            # Call Claude API using run_in_executor if not async
            response = await self.anthropic.messages.create(
                max_tokens=8192,
                messages=messages,
                model="claude-3-5-haiku-latest",
            )

            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()

            return ConversationResult(
                success=True,
                response=" ".join([con.text for con in response.content]),
                model="claude-3-5-haiku-latest",
                timestamp=end_time,
                metrics={
                    "processing_time_seconds": processing_time,
                    "token_count": sum(len(msg["content"].split()) for msg in messages),
                    "response_length": len(response.content[0].text),
                    "model_used": "claude-3-5-haiku-latest",
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

    async def process_task(self, task: Dict) -> Dict:
        if not isinstance(task, dict) or "task_id" not in task:
            self.task_logger.error("Invalid task format")
            return self._create_task_result(
                task.get("task_id", "unknown"), error="Invalid task format"
            )

        if not task.get("description"):
            self.task_logger.error("Description not found in task")
            return self._create_task_result(
                task.get("task_id", "unknown"), error="Description not found in task"
            )

        try:
            messages = [
                {
                    "role": "user",
                    "content": (f"{task.get('description')}"),
                }
            ]

            conversation_result = await self.process_message(messages)

            step_result = self._create_task_result(
                task.get("task_id"),
                step_id=task.get("step_id", "unknown"),
                step_description=task.get("description"),
                result=conversation_result.response
                if conversation_result.success
                else None,
                error=conversation_result.error
                if not conversation_result.success
                else None,
                metadata={
                    "model": conversation_result.model,
                    "metrics": conversation_result.metrics,
                    "timestamp": conversation_result.timestamp.isoformat(),
                    "conversation_history": messages,
                    "task_context": task.get("context", ""),
                },
            )

            return step_result

        except Exception as e:
            self.task_logger.error(f"Error processing task: {str(e)}")
            step_result = self._create_task_result(
                task.get("task_id"), error=f"Error processing task: {str(e)}"
            )
            return step_result
