import logging
import os
import subprocess
from datetime import datetime

from pydantic import BaseModel, Field
from pymongo import MongoClient

from workflow_state_manager import StepResult
from wrappers.wrapper_agent import WrapperAgent

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
        db_uri="mongodb://localhost:27017/",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, db_uri, kafka_bootstrap_servers)
        self.setup_task_logger()
        self.client = MongoClient(db_uri)
        self.db = self.client["agent_db"]
        self.tasks_collection = self.db["tasks"]  # Renamed from self.collection

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

            result = subprocess.run(command, capture_output=True, text=True, timeout=10)

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

    def process_task(self, task: dict) -> dict:
        if not isinstance(task, dict) or "_id" not in task:
            logger.error("Invalid task format")
            return {"success": False, "note": "Invalid task format"}

        try:
            task_description = task.get("description", "")
            self.task_logger.info(
                f"Processing task with description: {task_description}"
            )

            # Parse the command using structured output
            parsed_command = self._call_openai_api(task_description)
            logger.info(f"Using command parameters: target={parsed_command.target}")

            # Validate the target
            if not self.is_valid_domain_or_ip(parsed_command.target):
                step_result = StepResult(
                    step_description="ping",
                    success=False,
                    result=f"Invalid target: {parsed_command.target}",
                    error="Validation failed",
                )
                self.workflow_state_manager.update_workflow_state(
                    task_id=str(task["_id"]),
                    step_result=step_result,
                    agent_id=self.agent_id,
                )
                return {
                    "task_id": str(task["_id"]),
                    "result": step_result.result,
                    "success": False,
                    "note": "Validation failed",
                }

            # Execute the ping command
            start_time = datetime.now()
            ping_result = self.execute_ping(target=parsed_command.target)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Create step result
            step_result = StepResult(
                step_description="ping",
                success=True,
                result=ping_result,
                execution_time=execution_time,
                metadata={
                    "target": parsed_command.target,
                    "command_executed": f"ping {parsed_command.target}",
                },
                error=None,
            )

            # Update workflow state
            self.workflow_state_manager.update_workflow_state(
                task_id=str(task["_id"]),
                step_result=step_result,
                agent_id=self.agent_id,
            )

            return {"task_id": str(task["_id"]), "result": ping_result, "success": True}

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            step_result = StepResult(
                step_description="ping",
                success=False,
                result=None,
                error=str(e),
                metadata={"exception_type": type(e).__name__},
                execution_time=None,
            )
            self.workflow_state_manager.update_workflow_state(
                task_id=str(task["_id"]),
                step_result=step_result,
                agent_id=self.agent_id,
            )
            return {
                "task_id": str(task["_id"]),
                "error": str(e),
                "success": False,
                "note": "Exception occurred during task execution",
            }

    def start(self):
        """Start the agent with proper logging"""
        logger.info(f"Starting {self.agent_id}")
        super().start()
