import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

# Import the pure Python dig implementation
from boss.utils import serialize_task_to_string
from boss.wrappers.network_utils.dig import PythonDig
from boss.wrappers.wrapper_agent import WrapperAgent

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


class DigWrapperAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_network_dig_lookup",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()
        self.dig = PythonDig()  # Initialize the dig implementation

    def _call_openai_api(self, prompt: str) -> DigLookupCommand:
        """Call OpenAI API with structured output parsing for DIG lookup command"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = self.openai_client.beta.chat.completions.parse(
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

    def execute_dig_lookup(self, target: str) -> Dict[str, Any]:
        """Execute DNS lookup for the given domain using PythonDig"""
        try:
            self.task_logger.info(f"Executing DIG lookup with target={target}")
            start_time = datetime.now(timezone.utc)

            # Use the PythonDig implementation
            dig_result = self.dig.format_results(self.dig.dig(target))

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.task_logger.info(
                f"DIG lookup completed in {execution_time:.2f} seconds."
            )

            return dig_result

        except Exception as e:
            self.task_logger.error(f"An error occurred during DIG lookup: {str(e)}")
            return {"error": f"An error occurred during DIG lookup: {str(e)}"}

    def is_valid_target(self, target: str) -> bool:
        """Validate if the target is a valid domain or hostname"""
        import re

        hostname_regex = r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$"
        return bool(re.match(hostname_regex, target))

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(task, dict) or "_id" not in task:
            logger.error("Invalid task format")
            return {"success": False, "note": "Invalid task format"}

        try:
            task_id = task.get("_id")
            self.task_logger.info(f"Processing task with ID: {task_id}")

            # Extract current step index and steps
            current_step_index = task.get("current_step_index")
            steps = task.get("steps", [])
            if current_step_index is None or current_step_index >= len(steps):
                logger.error("Invalid current_step_index")
                return {"task_id": task_id, "error": "Invalid current_step_index"}

            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")

            task_prompt = f"Current step:\n{step_description}"

            # Parse the command using structured output
            parsed_command = self._call_openai_api(task_prompt)

            logger.info(f"Using command parameters: target={parsed_command.target}")

            # Validate the target
            if not self.is_valid_target(parsed_command.target):
                return {
                    "task_id": str(task["_id"]),
                    "result": f"Invalid target: {parsed_command.target}",
                    "note": "Validation failed",
                }

            # Execute the DIG lookup
            dig_result = self.execute_dig_lookup(target=parsed_command.target)

            # Prepare the result to send back to BOSS
            result = {
                "task_id": str(task["_id"]),
                "agent_id": self.agent_id,
                "result": serialize_task_to_string(dig_result),
                "metadata": {
                    "target": parsed_command.target,
                },
            }

            return result

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return {
                "task_id": str(task.get("_id", "unknown")),
                "agent_id": self.agent_id,
                "error": f"Validation error: {ve}",
                "note": "Invalid command parameters",
            }
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return {
                "task_id": str(task.get("_id", "unknown")),
                "agent_id": self.agent_id,
                "error": str(e),
                "note": "Exception occurred during task execution",
            }
