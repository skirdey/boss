# wrappers/wrapper_ping.py

import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field

from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PingCommand(BaseModel):
    """Model for ping command parameters"""

    target: str = Field(description="The domain or IP address to ping")


class WrapperPing(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_network_ping",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    def _call_openai_api(self, prompt):
        """Call OpenAI API with structured output parsing"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the ping command parameters from the task description.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=PingCommand,
            )

            parsed_command = completion.choices[0].message.parsed
            logger.info(f"Parsed command parameters: {parsed_command}")
            return parsed_command

        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            raise

    def execute_ping(self, target: str) -> str:
        """Execute ping command with given parameters"""
        try:
            # Adjust ping command based on OS
            if os.name == "nt":  # Windows
                command = ["ping", "-n", "4", target]
            else:  # Unix/Linux/MacOS
                command = ["ping", "-c", "4", target]

            self.task_logger.info(f"Executing ping command: {' '.join(command)}")

            result = subprocess.run(command, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                output = result.stdout.strip()
                self.task_logger.info(f"Ping successful: {output}")
                return output
            else:
                error_output = result.stderr.strip()
                self.task_logger.error(f"Ping failed: {error_output}")
                return f"Ping failed: {error_output}"

        except subprocess.TimeoutExpired:
            self.task_logger.error("Ping command timed out.")
            return "Ping command timed out."
        except Exception as e:
            self.task_logger.error(f"An error occurred while executing ping: {str(e)}")
            return f"An error occurred while executing ping: {str(e)}"

    def is_valid_domain_or_ip(self, target: str) -> bool:
        """Validate if the target is a valid domain or IP address"""
        import ipaddress
        import re

        # Validate IP address
        try:
            ipaddress.ip_address(target)
            return True
        except ValueError:
            pass

        # Validate domain name
        domain_regex = r"^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$"
        if re.match(domain_regex, target):
            return True

        return False

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

            # Collect previous steps' results
            previous_steps = steps[:current_step_index]
            previous_step_results = "\n".join(
                [
                    step.get("result", "")
                    for step in previous_steps
                    if step.get("result")
                ]
            )

            # Combine previous results with the current step description if available
            if previous_step_results:
                task_prompt = f"Previous step results:\n{previous_step_results}\n\nCurrent step:\n{step_description}"
            else:
                task_prompt = f"Current step:\n{step_description}"

            # Parse the command using structured output
            parsed_command = self._call_openai_api(task_prompt)

            logger.info(f"Using command parameters: target={parsed_command.target}")

            # Validate the target
            if not self.is_valid_domain_or_ip(parsed_command.target):
                return {
                    "task_id": str(task["_id"]),
                    "result": f"Invalid target: {parsed_command.target}",
                    "note": "Validation failed",
                }

            # Execute the ping command
            start_time = datetime.now(timezone.utc)
            ping_result = self.execute_ping(target=parsed_command.target)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Prepare the result to send back to BOSS
            result = {
                "task_id": str(task["_id"]),
                "agent_id": self.agent_id,
                "result": ping_result,
                "execution_time": execution_time,
                "metadata": {
                    "target": parsed_command.target,
                    "command_executed": f"ping {parsed_command.target}",
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return {
                "task_id": str(task["_id"]),
                "agent_id": self.agent_id,
                "error": str(e),
                "note": "Exception occurred during task execution",
            }
