from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import os
from typing import Any, Dict

from pydantic import BaseModel, Field

from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class PingCommand(BaseModel):
    """Model for ping command parameters"""

    target: str = Field(description="FQDN or IP address to ping.")


class WrapperPing(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_network_ping",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    async def _call_openai_api(self, prompt):
        """Call OpenAI API with structured output parsing asynchronously"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the FQDN or IP address from the provided text to be "
                            "used as a target for 'ping' command. Remove http or https or www from the target."
                        ),
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

    async def execute_ping(self, target: str) -> str:
        """Execute ping command asynchronously with given parameters"""
        try:
            # Adjust ping command based on OS
            if os.name == "nt":  # Windows
                command = ["ping", "-n", "4", target]
            else:  # Unix/Linux/MacOS
                command = ["ping", "-c", "4", target]

            self.task_logger.info(f"Executing ping command: {' '.join(command)}")

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=360
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                self.task_logger.error("Ping command timed out.")
                return "Ping command timed out."

            output = stdout.decode().strip() + "\n" + stderr.decode().strip()
            self.task_logger.info(f"Ping result: {output}")
            return output

        except Exception as e:
            self.task_logger.error(f"An error occurred while executing ping: {str(e)}")
            return f"An error occurred while executing ping: {str(e)}"

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.task_logger.info("**************** PING AGENT ****************")

        if not isinstance(task, dict) or "task_id" not in task:
            logger.error("Invalid task format")
            return self._create_task_result(
                task.get("task_id"), task.get("step_id"), error="Invalid task format"
            )

        try:
            task_id = task.get("task_id")
            step_id = task.get("step_id") or task.get("current_step_id")
            self.task_logger.info(
                f"Processing task with ID: {task_id} and step ID: {step_id} of task {task}"
            )
            # Parse the command using structured output
            parsed_command = await self._call_openai_api(task.get("description"))

            self.task_logger.info(
                f"Using command parameters: target={parsed_command.target}"
            )

            # Execute the ping command
            ping_result = await self.execute_ping(target=parsed_command.target)

            # Prepare the result to send back to BOSS
            result = self._create_task_result(
                task.get("task_id"),
                step_id=step_id,
                step_description=task.get("description"),
                result=ping_result,
                metadata={
                    "target": parsed_command.target,
                    "command_executed": f"ping {parsed_command.target}",
                },
            )

            logger.info(f"WrapperPing result: {result}")

            return result

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(
                task.get("task_id"), task.get("step_id"), error=str(e)
            )
