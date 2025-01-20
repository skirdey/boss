# boss/boss/wrappers/wrapper_browser_use_agent.py
import asyncio
import logging
import os
from typing import Any, Dict

from browser_use import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from boss.wrappers.wrapper_agent import WrapperAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False

load_dotenv()

class BrowserUseCommand(BaseModel):
    """Model for browser use command parameters"""

    task: str = Field(
        description="The task to be performed by the browser agent. This should clearly define the steps for the agent to execute using the browser."
    )

class WrapperBrowserUseAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_browser_use",
        kafka_bootstrap_servers="localhost:9092",
        max_iterations: int = 5,
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()
        self.max_iterations = max_iterations

    def setup_task_logger(self):
        """Setup task-specific logging."""
        self.task_logger = logging.getLogger(f"{self.agent_id}_task")
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.task_logger.addHandler(handler)
        self.task_logger.setLevel(logging.INFO)

    async def _call_openai_api(self, prompt: str) -> BrowserUseCommand:
        """
        Call OpenAI to determine the task for the browser agent based on the prompt.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that generates a task for a browser agent. The task should be clearly defined and executable by the browser-use library.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=BrowserUseCommand,
            )
            logger.info(f"Response from LLM: {completion}")
            return completion.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    async def execute_browser_task(self, task_description: str) -> str:
        """
        Execute a task using the browser-use library.
        """
        try:
            self.task_logger.info(f"Executing browser task: {task_description}")
            llm = ChatOpenAI(model="gpt-4o-mini")
            browser_agent = Agent(task=task_description, llm=llm)
            result = (await browser_agent.run()).model_dump()
            self.task_logger.info(f"Browser task completed: {result}")
            return result
        except Exception as e:
            error_message = f"Error executing browser task: {e}"
            self.task_logger.error(error_message)
            return error_message

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the task using the browser agent.
        """
        self.task_logger.info("**************** BROWSER USE AGENT ****************")
        self.task_logger.info(f"{task}\n\n")

        task_id = task.get("task_id")
        step_id = task.get("step_id")
        if not task_id or not step_id:
            error_message = "Invalid task format: missing task_id or step_id"
            self.task_logger.error(error_message)
            return self._create_task_result(task_id, step_id, error=error_message)

        try:
            step_description = task.get("description", "")
            if targets := task.get("targets", []):
                step_description += f"\n\nTargets: {targets} \n\n"

            if context := task.get("context", ""):
                step_description += f"\n\nPrevious step execution context: {context} \n\n"

            # Get the command from LLM
            parsed_command = await self._call_openai_api(step_description)
            self.task_logger.info(f"Parsed browser command: {parsed_command}")

            # Execute the browser task
            browser_result = await self.execute_browser_task(parsed_command.task)

            # Prepare the result
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                step_description=step_description,
                result=browser_result,
                metadata={"task": parsed_command.task},
            )

        except Exception as e:
            error_message = f"Error processing task: {e}"
            self.task_logger.error(error_message)
            return self._create_task_result(task_id, step_id, error=error_message)