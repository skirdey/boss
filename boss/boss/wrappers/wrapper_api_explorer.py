# wrappers/wrapper_rest_explorer.py

import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from boss.utils import serialize_task_to_string
from boss.wrappers.network_utils.scan_apis import scan_api
from boss.wrappers.wrapper_agent import WrapperAgent

# Assuming the APIScanner code is available in the same module or can be imported
# from apiscanner import APIScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class RestExplorerCommand(BaseModel):
    """Model for REST API explorer command parameters"""

    target: str = Field(description="FQDN or IP address to scan for API endpoints.")
    ports: Optional[List[int]] = Field(
        description="List of ports to scan for API endpoints. If not provided, default ports will be used."
    )


class WrapperAPIExplorer(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_api_explorer",
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
                        "content": (
                            "Extract the FQDN or IP address and ports from the provided text to be used "
                            "as a target for scanning REST API endpoints. Remove http or https or www from the target."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=RestExplorerCommand,
            )

            parsed_command = completion.choices[0].message.parsed
            logger.info(f"Parsed command parameters: {parsed_command}")
            return parsed_command

        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            return None

    def execute_scan(self, target: str, ports: Optional[List[int]] = None):
        """Execute API scan with given parameters"""
        try:
            self.task_logger.info(
                f"Executing API scan on target: {target} with ports {ports}"
            )

            # Instantiate the APIScanner with the target
            report = scan_api(target, ports)

            self.task_logger.info("API scan completed.")
            return report

        except Exception as e:
            self.task_logger.error(f"An error occurred while scanning: {str(e)}")
            return {"error": str(e)}

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(task, dict) or "_id" not in task:
            logger.error("Invalid task format")
            return self._create_task_result(
                task.get("_id"), error="Invalid task format"
            )

        try:
            task_id = task.get("_id")
            self.task_logger.info(f"Processing task with ID: {task_id}")

            # Extract current step index and steps
            current_step_index = task.get("current_step_index")
            steps = task.get("steps", [])
            if current_step_index is None or current_step_index >= len(steps):
                logger.error("Invalid current_step_index")
                return self._create_task_result(
                    task.get("_id"), error="Invalid current_step_index"
                )

            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")

            # Collect previous steps' results
            previous_steps = steps[:current_step_index]
            previous_step_results = "\n".join(
                [serialize_task_to_string(step) for step in previous_steps]
            )

            # Combine previous results with the current step description if available
            if previous_step_results:
                task_prompt = f"Previous step results:\n{previous_step_results}\n\nCurrent step:\n{step_description}"
            else:
                task_prompt = f"Current step:\n{step_description}"

            task_prompt += "\n\nChoose the most probable target and ports to scan for API endpoints based on the previous steps description."

            # Parse the command using structured output
            parsed_command = self._call_openai_api(task_prompt)

            logger.info(f"Using command parameters: target={parsed_command.target}")

            scan_result = self.execute_scan(
                target=parsed_command.target, ports=parsed_command.ports
            )

            # Prepare the result to send back to BOSS
            result = self._create_task_result(
                task.get("_id"),
                result=serialize_task_to_string(scan_result),
                step_description=step_description,
            )

            return result

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            result = self._create_task_result(
                task.get("_id"), error=f"Error processing task: {str(e)}"
            )
            return result
