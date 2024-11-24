import logging
import os
import re
import socket
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class WhoisLookupCommand(BaseModel):
    """Model for WHOIS lookup command parameters"""

    target: str = Field(description="Domain name to be used for WHOIS lookup.")


class WrapperWhoisAgent(WrapperAgent):
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
        """Execute WHOIS lookup for the given domain and return a formatted report"""
        tld = target.split(".")[-1].lower()
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

            # Parse the WHOIS response
            parsed_report = self.parse_whois_response(response.decode())

            return {"whois_report": parsed_report}

        except Exception as e:
            self.task_logger.error(f"An error occurred during WHOIS lookup: {str(e)}")
            return {"error": f"An error occurred during WHOIS lookup: {str(e)}"}

    def parse_whois_response(self, whois_data: str) -> str:
        """Parse the raw WHOIS data and format it into a readable report"""

        # Define regex patterns for desired fields
        patterns = {
            "Registrar": r"Registrar:\s*(.+)",
            "Creation Date": r"Creation Date:\s*(.+)",
            "Updated Date": r"Updated Date:\s*(.+)",
            "Expiration Date": r"Expiration Date:\s*(.+)",
            "Registrant Name": r"Registrant Name:\s*(.+)",
            "Registrant Organization": r"Registrant Organization:\s*(.+)",
            "Registrant Email": r"Registrant Email:\s*(.+)",
            "Registrant Phone": r"Registrant Phone:\s*(.+)",
        }

        extracted_data = {}

        for field, pattern in patterns.items():
            match = re.search(pattern, whois_data, re.IGNORECASE)
            if match:
                extracted_data[field] = match.group(1).strip()
            else:
                extracted_data[field] = "Not Available"

        # Format the extracted data into a report string
        report = (
            f"WHOIS Report for {extracted_data.get('Registrar', 'N/A')}\n"
            f"{'-'*60}\n"
            f"Registrar: {extracted_data.get('Registrar')}\n"
            f"Creation Date: {extracted_data.get('Creation Date')}\n"
            f"Updated Date: {extracted_data.get('Updated Date')}\n"
            f"Expiration Date: {extracted_data.get('Expiration Date')}\n\n"
            f"Registrant Information:\n"
            f"Name: {extracted_data.get('Registrant Name')}\n"
            f"Organization: {extracted_data.get('Registrant Organization')}\n"
            f"Email: {extracted_data.get('Registrant Email')}\n"
            f"Phone: {extracted_data.get('Registrant Phone')}\n"
        )

        logger.info("Formatted WHOIS report created.")
        return report

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

            task_prompt = f"Current step:\n{step_description}"

            # Parse the command using structured output
            parsed_command = self._call_openai_api(task_prompt)

            logger.info(f"Using command parameters: target={parsed_command.target}")

            # Validate the target
            if not self.is_valid_target(parsed_command.target):
                return self._create_task_result(
                    task.get("_id"),
                    error=f"Invalid target: {parsed_command.target}",
                )

            # Execute the WHOIS lookup
            whois_result = self.execute_whois_lookup(target=parsed_command.target)

            if "whois_report" in whois_result:
                # Prepare the result to send back to BOSS
                result = self._create_task_result(
                    task.get("_id"),
                    step_description=step_description,
                    result=whois_result["whois_report"],
                    metadata={"target": parsed_command.target},
                )
            else:
                # Handle errors
                result = self._create_task_result(
                    task.get("_id"), error=whois_result.get("error", "Unknown error")
                )

            return result

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return self._create_task_result(
                task.get("_id"), error=f"Validation error: {ve}"
            )
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(
                task.get("_id"), error=f"Error processing task: {str(e)}"
            )
