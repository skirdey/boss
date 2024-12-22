import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from boss.utils import serialize_task_to_string
from boss.wrappers.network_utils.dig import PythonDig
from boss.wrappers.wrapper_agent import WrapperAgent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class DigLookupCommand(BaseModel):
    """Model for dig lookup command parameters"""

    target: str = Field(description="Domain name to be used for DNS lookup.")


# Works as expected
class WrapperDigAgent(WrapperAgent):
    def __init__(
        self,
        agent_id: str = "agent_network_dig_lookup",
        kafka_bootstrap_servers: str = "localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()
        self.dig = PythonDig()  # Initialize the dig implementation

    async def _call_openai_api(self, prompt: str) -> DigLookupCommand:
        """Asynchronously call OpenAI API with structured output parsing for DIG lookup command"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the DNS lookup command parameters from the task description.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=DigLookupCommand,
            )

            parsed_command = completion.choices[0].message.parsed
            logger.info(f"Parsed command parameters: {parsed_command}")
            return parsed_command

        except ValidationError as ve:
            logger.error(f"Validation error when parsing command: {ve}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            raise

    async def execute_dig_lookup(self, target: str) -> Dict[str, Any]:
        """Asynchronously execute DNS lookup for the given domain using PythonDig"""
        try:
            self.task_logger.info(f"Executing DIG lookup with target={target}")
            start_time = datetime.now(timezone.utc)

            # Execute dig command asynchronously
            dig_result = await asyncio.to_thread(self.dig.dig, target)
            formatted_result = self.dig.format_results(dig_result)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.task_logger.info(
                f"DIG lookup completed in {execution_time:.2f} seconds."
            )

            return formatted_result

        except Exception as e:
            self.task_logger.error(f"An error occurred during DIG lookup: {str(e)}")
            return {"error": f"An error occurred during DIG lookup: {str(e)}"}

    def is_valid_target(self, target: str) -> bool:
        """Validate if the target is a valid domain or hostname"""
        hostname_regex = r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$"
        return bool(re.match(hostname_regex, target))

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously process a DIG lookup task with tree structure support"""
        self.task_logger.info("**************** DIG AGENT ****************")

        if not isinstance(task, dict) or "task_id" not in task or "step_id" not in task:
            logger.error("Invalid task format")
            return self._create_task_result(
                task.get("task_id"), task.get("step_id"), error="Invalid task format"
            )

        try:
            task_id = task.get("task_id")
            step_id = task.get("step_id")
            self.task_logger.info(
                f"Processing task with ID: {task_id} and step ID: {step_id}"
            )

            # Extract step description
            step_description = task.get("description", "")

            task_prompt = f"Current step:\n{step_description}"

            targets = task.get("targets", [])
            if targets:
                task_prompt += f"\n\nTargets: {targets} \n\n"

            # Parse the command using structured output
            parsed_command = await self._call_openai_api(task_prompt)

            self.task_logger.info(
                f"Using command parameters: target={parsed_command.target}"
            )

            # Validate the target
            if not self.is_valid_target(parsed_command.target):
                return self._create_task_result(
                    task_id=task_id,
                    step_id=step_id,
                    error=f"Invalid target: {parsed_command.target}",
                )

            # Execute the DIG lookup
            dig_result = await self.execute_dig_lookup(target=parsed_command.target)

            # Prepare the result to send back to BOSS
            result = self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                step_description=step_description,
                result=serialize_task_to_string(dig_result),
                metadata={
                    "target": parsed_command.target,
                },
            )

            return result

        except ValidationError as ve:
            self.task_logger.error(f"Validation error: {ve}")
            return self._create_task_result(
                task_id=task.get("task_id"),
                step_id=task.get("step_id"),
                error=f"Validation error: {ve}",
            )
        except Exception as e:
            self.task_logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(
                task_id=task.get("task_id"),
                step_id=task.get("step_id"),
                error=str(e),
            )
