import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

from boss.wrappers.network_utils.ssl_certificate import get_ssl_certificate
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GetSSLCertificateCommand(BaseModel):
    """Model for get SSL certificate command parameters"""

    target: str = Field(
        description="The domain, hostname, or IP address to retrieve the SSL certificate from. Avoid providing protocols like https or http."
    )


class WrapperGetSSLCertificateAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_network_get_ssl_certificate",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    def _call_openai_api(self, prompt: str) -> GetSSLCertificateCommand:
        """Call OpenAI API with structured output parsing for get SSL certificate command"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the get SSL certificate command parameters from the task description.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=GetSSLCertificateCommand,
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

    def execute_get_ssl_certificate(
        self,
        target: str,
        port: int = 443,
    ) -> Dict[str, Any]:
        """Execute get_ssl_certificate function with given parameters"""
        try:
            self.task_logger.info(
                f"Executing get_ssl_certificate with target={target}, port={port}"
            )

            start_time = datetime.now(timezone.utc)
            cert_info = get_ssl_certificate(host=target, port=port)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.task_logger.info(
                f"SSL certificate retrieval completed in {execution_time:.2f} seconds."
            )
            return cert_info

        except Exception as e:
            self.task_logger.error(
                f"An error occurred during SSL certificate retrieval: {str(e)}"
            )
            return {
                "error": f"An error occurred during SSL certificate retrieval: {str(e)}"
            }

    def is_valid_target(self, target: str) -> bool:
        """Validate if the target is a valid domain, hostname, or IP address"""
        import ipaddress
        import re

        # Validate IP address
        try:
            ipaddress.ip_address(target)
            return True
        except ValueError:
            pass

        # Validate hostname (simpler regex for hostname validation)
        hostname_regex = r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$"
        if re.match(hostname_regex, target):
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
            if not self.is_valid_target(parsed_command.target):
                return {
                    "task_id": str(task["_id"]),
                    "result": f"Invalid target: {parsed_command.target}",
                    "note": "Validation failed",
                }

            # Execute the SSL certificate retrieval
            cert_result = self.execute_get_ssl_certificate(target=parsed_command.target)

            # Prepare the result to send back to BOSS
            result = {
                "task_id": str(task["_id"]),
                "agent_id": self.agent_id,
                "result": cert_result,
                "metadata": {"target": parsed_command.target},
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