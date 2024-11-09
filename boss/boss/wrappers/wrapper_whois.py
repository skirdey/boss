import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class WhoisLookupCommand(BaseModel):
    """Model for whois lookup command parameters"""

    target: str = Field(description="Domain name to be used for WHOIS lookup.")


class WhoisWrapperAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_network_whois_lookup",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    def _call_openai_api(self, prompt: str) -> WhoisLookupCommand:
        """Call OpenAI API with structured output parsing for WHOIS lookup command"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the WHOIS lookup command parameters from the task description.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=WhoisLookupCommand,
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

    def execute_whois_lookup(
        self,
        target: str,
    ) -> Dict[str, Any]:
        """Execute WHOIS lookup for the given domain"""
        tld = target.split(".")[-1]
        if tld in ["com", "net", "org", "info", "biz"]:
            server = "whois.verisign-grs.com"
        elif tld == "io":
            server = "whois.nic.io"
        else:
            server = "whois.iana.org"

        port = 43
        query = target + "\r\n"

        try:
            self.task_logger.info(f"Executing WHOIS lookup with target={target}")
            start_time = datetime.now(timezone.utc)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((server, port))
            sock.sendall(query.encode())
            response = b""
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                response += data
            sock.close()

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.task_logger.info(
                f"WHOIS lookup completed in {execution_time:.2f} seconds."
            )
            return {"whois_data": response.decode()}

        except Exception as e:
            self.task_logger.error(f"An error occurred during WHOIS lookup: {str(e)}")
            return {"error": f"An error occurred during WHOIS lookup: {str(e)}"}

    def is_valid_target(self, target: str) -> bool:
        """Validate if the target is a valid domain or hostname"""
        import re

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

            # Execute the WHOIS lookup
            whois_result = self.execute_whois_lookup(target=parsed_command.target)

            # Prepare the result to send back to BOSS
            result = {
                "task_id": str(task["_id"]),
                "agent_id": self.agent_id,
                "result": whois_result,
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
