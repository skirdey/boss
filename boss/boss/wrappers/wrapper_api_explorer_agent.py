import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from boss.utils import serialize_task_to_string
from boss.wrappers.network_utils.scan_apis import scan_api
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class RestExplorerCommand(BaseModel):
    """Model for REST API explorer command parameters"""

    target: str = Field(
        description="FQDN or IP address to scan for API endpoints based on targets or step description"
    )
    ports: Optional[List[int]] = Field(
        default=None,
        description="List of ports to scan for API endpoints. If not provided, default ports will be used.",
    )


class WrapperAPIExplorerAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_api_explorer",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    async def _call_openai_api(self, prompt: str) -> Optional[RestExplorerCommand]:
        """Asynchronously call OpenAI API with structured output parsing."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the FQDN or IP address and ports from the provided text to be used based on targets or step description only,"
                            "as a target for scanning REST API endpoints. Remove http or https or www from the target."
                            "Use valid targets based on the provided text. Do not come up with your own targets that not mentioned in the provided text."
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

    async def execute_scan(
        self, target: str, ports: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Asynchronously execute the API scan with given parameters."""
        try:
            self.task_logger.info(
                f"Executing API scan on target: {target} with ports {ports}"
            )

            # Run the synchronous scan in a thread pool
            report = await scan_api(target, ports)

            self.task_logger.info("API scan completed.")
            return report
        except Exception as e:
            self.task_logger.error(f"An error occurred while scanning: {str(e)}")
            return {"error": str(e)}

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously process the API exploration task."""
        self.task_logger.info("**************** API EXPLORER AGENT ****************")
        self.task_logger.info(f"{task}\n\n")

        # Validate required fields
        if not isinstance(task, dict) or "task_id" not in task or "step_id" not in task:
            logger.error("Invalid task format")
            return self._create_task_result(
                task_id=task.get("task_id", "unknown"),
                step_id=task.get("step_id", "unknown"),
                error="Invalid task format",
            )

        task_id = task.get("task_id")
        step_id = task.get("step_id")
        description = task.get("description", "")

        targets = task.get("targets", [])

        if targets:
            description += f"\n\nTargets: {targets} \n\n"

        try:
            # Parse the command using the task's description
            parsed_command = await self._call_openai_api(description)

            if not parsed_command:
                return self._create_task_result(
                    task_id=task_id,
                    step_id=step_id,
                    error="Failed to parse command from the description",
                )

            logger.info(
                f"Using command parameters: target={parsed_command.target}, ports={parsed_command.ports}"
            )

            # Execute the API scan
            scan_result = await self.execute_scan(
                target=parsed_command.target, ports=parsed_command.ports
            )

            # Prepare and return the result
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                step_description=description,
                result=serialize_task_to_string(scan_result),
                metadata={
                    "target": parsed_command.target,
                    "ports": parsed_command.ports if parsed_command.ports else None,
                },
            )

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                error=str(e),
            )
