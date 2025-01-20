import os
import re

os.environ["PYTHONIOENCODING"] = "utf-8"

import asyncio
import logging
import os
import shlex
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class WapitiCliCommand(BaseModel):
    """Model for Wapiti CLI command parameters"""

    # Basic options
    url: str = Field(description="Target URL for the Wapiti scan, can only be extracted from the provided description, targets or context")


    max_attack_time: Optional[int] = Field(
        description="Maximum time to spend on attacks (in seconds), maximum of 180 seconds"
    )


class WrapperWapitiCliAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_wapiti_cli_scanner",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    def setup_task_logger(self):
        """Setup task-specific logging"""
        self.task_logger = logging.getLogger(f"{self.agent_id}_task")
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.task_logger.addHandler(handler)
        self.task_logger.setLevel(logging.INFO)

    async def _call_openai_api(self, prompt: str) -> WapitiCliCommand:
        """Call OpenAI API with structured output parsing for Wapiti CLI parameters"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            system_prompt = """
            Extract Wapiti CLI parameters from the task description. Your goal is to construct a valid Wapiti command. Consider:
            1. Identify the target URL from the task description, targets or context, it needs to be compatible with Wapiti CLI.
            Map these to the corresponding Wapiti CLI options.
            """

            completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=WapitiCliCommand,
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

    async def execute_wapiti_scan(self, command: WapitiCliCommand) -> Dict[str, Any]:
        """Execute Wapiti scan using the command line interface and capture output"""
        try:
            self.task_logger.info(f"Executing Wapiti scan on: {command.url}")
            start_time = datetime.now(timezone.utc)

            wapiti_command = ["wapiti", "-u", command.url]

            # Mapping model fields to CLI arguments
            field_mappings = {
                "output_format": ("-f", "json"),
                "max_attack_time_seconds": (
                    "--max-attack-time",
                    str(60)
                ),
            }

            # Dynamically add arguments based on field mappings
            for field, (cli_arg, value) in field_mappings.items():
                if isinstance(value, bool):
                    if value:
                        wapiti_command.append(cli_arg)
                elif value is not None:
                    wapiti_command.extend([cli_arg, value])

            self.task_logger.info(
                f"Constructed Wapiti command: {' '.join(wapiti_command)}"
            )

            # Quote arguments with spaces
            quoted_command = [shlex.quote(arg) for arg in wapiti_command]
            command_string = " ".join(quoted_command)
            self.task_logger.info(f"Executing command: {command_string}")

            process = await asyncio.create_subprocess_shell(
                command_string,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True, 
                encoding=None,

            )

            stdout, stderr = await process.communicate()
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            if process.returncode == 0:
                self.task_logger.info(
                    f"Wapiti scan completed successfully in {execution_time:.2f} seconds."
                )

                report_text = stdout.decode(encoding="utf-8", errors="replace")
                self.task_logger.info(f"Report text: {report_text}")

                # Extract report path using regex
                report_path_match = re.search(
                    r"A report has been generated in the file (.+\.json)", report_text
                )
                if report_path_match:
                    report_path = report_path_match.group(1)
                    if os.path.exists(report_path):
                        with open(report_path, "r", encoding="utf-8") as report_file:
                            report_content = report_file.read()
                    else:
                        report_content = "Report file not found."

                else:
                    report_content = "Report path not found in Wapiti output."

                return {
                    "success": True,
                    "execution_time": execution_time,
                    "report": report_content,
                }
            else:
                self.task_logger.error(
                    f"Wapiti scan failed with return code: {process.returncode, stderr.decode(), }"
                )
                return {
                    "success": False,
                    "error": f"Wapiti scan failed with return code: {process.returncode}, stderr: {stderr.decode()}",
                    "execution_time": execution_time,
                }

        except FileNotFoundError:
            self.task_logger.error(
                "Wapiti executable not found. Ensure it's in your PATH."
            )
            return {
                "success": False,
                "error": "Wapiti executable not found. Ensure it's in your PATH.",
            }
        except Exception as e:
            self.task_logger.error(f"Error during Wapiti execution: {str(e)}")
            return {
                "success": False,
                "error": f"Error during Wapiti execution: {str(e)}",
            }

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task asynchronously"""
        self.task_logger.info("**************** WAPITI CLI AGENT ****************")

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

            step_description = task.get("description", "")
            task_prompt = f"Run a Wapiti scan with the following requirements:\n{step_description}\nPlease provide the report in JSON format if possible."

            targets = task.get("targets", [])
            if targets:
                task_prompt += f"\nTargets: {targets}\n"

            context = task.get("context", "")
            if context:
                task_prompt += f"\nContext: {context}\n"

            # Generate scan parameters using LLM
            parsed_command = await self._call_openai_api(task_prompt)

            self.task_logger.info(f"Parsed Wapiti command: {parsed_command}")

            # Execute the Wapiti scan using the CLI
            scan_result = await self.execute_wapiti_scan(parsed_command)

            self.task_logger.info(f"Scan result: {scan_result} for parsed command: {parsed_command}")

            # Prepare the result
            result = self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                result=f"Scan completed in {scan_result.get('execution_time', 0):.2f} seconds.\n{scan_result.get('report', '')}",
                step_description=step_description,
            )

            return result

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return self._create_task_result(
                task_id=task.get("task_id"),
                step_id=task.get("step_id"),
                error=f"Validation error: {ve}",
            )
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(
                task_id=task.get("task_id"),
                step_id=task.get("step_id"),
                error=str(e),
            )
