import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError

from boss.wrappers.network_utils.ssl_certificate import get_ssl_certificate
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class GetSSLCertificateCommand(BaseModel):
    """Model for get SSL certificate command parameters"""

    target: str = Field(
        description="Extract hostname for server_hostname parameter in get_ssl_certificate function."
    )


class WrapperGetSSLCertificateAnalysisAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_network_get_ssl_certificate",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    async def _call_openai_api(self, prompt: str) -> GetSSLCertificateCommand:
        """Asynchronously call OpenAI API with structured output parsing for get SSL certificate command"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            completion = await self.openai_client.beta.chat.completions.parse(
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

    async def execute_get_ssl_certificate(
        self, target: str, port: int = 443
    ) -> Dict[str, Any]:
        """Asynchronously execute get_ssl_certificate function with given parameters"""
        try:
            self.task_logger.info(
                f"Executing get_ssl_certificate with target={target}, port={port}"
            )
            start_time = datetime.now(timezone.utc)

            # Run synchronous I/O in a separate thread
            cert_info = await asyncio.to_thread(
                get_ssl_certificate, host=target, port=port
            )

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

        # Validate hostname
        hostname_regex = r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$"
        return bool(re.match(hostname_regex, target))

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously process the SSL certificate retrieval task."""
        self.task_logger.info("**************** SSL CERTIFICATE AGENT ****************")
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

        try:
            # Parse the command using the task's description
            parsed_command = await self._call_openai_api(description)
            logger.info(f"Using command parameters: target={parsed_command.target}")

            # Validate the target
            if not self.is_valid_target(parsed_command.target):
                return self._create_task_result(
                    task_id=task_id,
                    step_id=step_id,
                    error=f"Invalid target: {parsed_command.target}",
                )

            # Execute the SSL certificate retrieval
            cert_result = await self.execute_get_ssl_certificate(
                target=parsed_command.target
            )

            # Prepare the result
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                step_description=description,
                result=json.dumps(cert_result),
                metadata={"target": parsed_command.target},
            )

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                error=f"Validation error: {ve}",
            )
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                error=str(e),
            )
