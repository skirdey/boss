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
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


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

    async def _call_openai_api(self, prompt: str) -> ScanPortsCommand:
        """Call OpenAI API with structured output parsing for scan ports command"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
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

    async def get_ports_to_scan(self, previous_steps_info: str) -> List[int]:
        """Use LLM to determine which ports to scan based on previous steps"""
        try:
            completion = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "As an expert network engineer, given the previous steps information, "
                            "suggest a list of TCP ports that should be scanned for security testing. Try to avoid duplicate work and repeated scans. "
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

    async def execute_scan_ports(
        self,
        target: str,
        ports_to_scan: List[int],
        timeout: float = 0.5,
        max_concurrent: int = 10,
    ) -> str:
        """Execute scan_ports function with given parameters asynchronously"""
        try:
            self.task_logger.info(
                f"Executing scan_ports with target={target}, timeout={timeout}"
            )

            start_time = datetime.now(timezone.utc)
            # Assuming scan_ports can be run in a thread pool if it's synchronous
            scan_result = await scan_ports(
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

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.task_logger.info("**************** SCAN PORT AGENT ****************")
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

            # Extract current step description
            step_description = task.get("description", "")

            # Combine previous results with the current step description if available
            task_prompt = f"Current step:\n{step_description}"

            targets = task.get("targets", [])
            if targets:
                task_prompt += f"\n\nTargets: {targets} \n\n"

            context = task.get("context", "")

            if context:
                task_prompt += f"\n\nPrevious step execution context(Use to guide yourself): {context} \n\n"

            # Parse the command using structured output
            parsed_command = await self._call_openai_api(task_prompt)

            logger.info(f"Using command parameters: target={parsed_command.target}")

            # Use LLM to determine ports to scan
            ports_to_scan = await self.get_ports_to_scan(
                task_prompt
            )  # Using task_prompt for context
            if not ports_to_scan:
                self.task_logger.error("Failed to get ports to scan from LLM")
                return self._create_task_result(
                    task_id=task_id,
                    step_id=step_id,
                    error="Failed to determine ports to scan",
                )

            # Execute the port scan
            scan_result = await self.execute_scan_ports(
                target=parsed_command.target, ports_to_scan=ports_to_scan
            )

            # Prepare the result to send back to BOSS
            result = self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                step_description=step_description,
                result=scan_result,
                metadata={
                    "target": parsed_command.target,
                    "ports_scanned": ports_to_scan,
                },
            )

            return result

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return self._create_task_result(
                task_id=task.get("task_id", "unknown"),
                step_id=task.get("step_id", "unknown"),
                error=f"Validation error: {ve}",
            )
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(
                task_id=task.get("task_id", "unknown"),
                step_id=task.get("step_id", "unknown"),
                error=str(e),
            )
