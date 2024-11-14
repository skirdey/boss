# wrappers/wrapper_scan_port.py

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from boss.wrappers.network_utils.scan_ports import scan_ports
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ScanPortsCommand(BaseModel):
    """Model for scan ports command parameters"""

    target: str = Field(
        description=(
            "The domain, hostname, or IP address to scan. Avoid providing https or http protocols, "
            "they are not required."
        )
    )


class WrapperScanPortAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_network_scan_ports",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    def _call_openai_api(self, prompt: str) -> ScanPortsCommand:
        """Call OpenAI API with structured output parsing for scan ports command"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the scan ports command parameters from the task description.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=ScanPortsCommand,
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

    def get_ports_to_scan(self, previous_steps_info: str) -> List[int]:
        """Use LLM to determine which ports to scan based on previous steps"""
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "As an expert network engineer, given the previous steps information, "
                            "suggest a list of TCP ports that should be scanned. "
                            "Provide the list as comma-separated numbers."
                        ),
                    },
                    {"role": "user", "content": previous_steps_info},
                ],
            )

            # Parse the output to get the list of ports
            port_list_str = completion.choices[0].message.content.strip()
            # Extract integers from the string
            ports = [int(port) for port in re.findall(r"\b\d+\b", port_list_str)]
            logger.info(f"Ports to scan determined by LLM: {ports}")
            return ports

        except Exception as e:
            self.task_logger.error(
                f"An error occurred while determining ports to scan: {str(e)}"
            )
            # Return a default port list in case of error
            default_ports = list(range(1, 1025))  # Common ports
            logger.info(f"Using default ports to scan: {default_ports}")
            return default_ports

    def execute_scan_ports(
        self,
        target: str,
        ports_to_scan: List[int],
        timeout: float = 0.5,
        max_concurrent: int = 10,
    ) -> str:
        """Execute scan_ports function with given parameters"""
        try:
            self.task_logger.info(
                f"Executing scan_ports with target={target}, timeout={timeout}"
            )

            start_time = datetime.now(timezone.utc)
            scan_result = scan_ports(
                target=target,
                ports_to_scan=ports_to_scan,
                timeout=timeout,
                max_concurrent=max_concurrent,
            )
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.task_logger.info(
                f"Port scan completed in {execution_time:.2f} seconds."
            )
            return scan_result

        except Exception as e:
            self.task_logger.error(f"An error occurred during port scan: {str(e)}")
            return f"An error occurred during port scan: {str(e)}"

    def is_valid_target(self, target: str) -> bool:
        """Validate if the target is a valid domain, hostname, or IP address"""
        import ipaddress

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
                previous_steps_info = (
                    f"Previous step results:\n{previous_step_results}\n\n"
                    f"Current step:\n{step_description}"
                )
            else:
                previous_steps_info = f"Current step:\n{step_description}"

            # Parse the command using structured output
            parsed_command = self._call_openai_api(previous_steps_info)

            logger.info(f"Using command parameters: target={parsed_command.target}")

            # Validate the target
            if not self.is_valid_target(parsed_command.target):
                return {
                    "task_id": str(task["_id"]),
                    "result": f"Invalid target: {parsed_command.target}",
                    "note": "Validation failed",
                }

            # Use LLM to determine ports to scan
            ports_to_scan = self.get_ports_to_scan(previous_steps_info)
            if not ports_to_scan:
                self.task_logger.error("Failed to get ports to scan from LLM")
                return {
                    "task_id": str(task["_id"]),
                    "result": "Failed to determine ports to scan",
                    "note": "LLM did not provide any ports to scan",
                }

            # Execute the port scan
            scan_result = self.execute_scan_ports(
                target=parsed_command.target, ports_to_scan=ports_to_scan
            )

            # Prepare the result to send back to BOSS
            result = {
                "task_id": str(task["_id"]),
                "agent_id": self.agent_id,
                "result": scan_result,
                "metadata": {
                    "target": parsed_command.target,
                    "ports_scanned": ports_to_scan,
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
