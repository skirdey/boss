import os
import re
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic
from pydantic import BaseModel, Field, ValidationError

from boss.models import ScanFinding
from boss.utils import get_iso_timestamp, serialize_task_to_string
from boss.wrappers.network_utils.sql_injection import ScanTarget, SecurityScanner
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class ScanParameters(BaseModel):
    """Model for security scan parameters"""

    target_url: str = Field(..., description="Target URL to scan based on task description or targets")
    paths: List[str] = Field(..., description="List of paths to test for SQL injections")
    description: Optional[str] = Field(description="Scan description")


class WrapperSQLInjectionAgent(WrapperAgent):
    """
    Asynchronous Wrapper agent for SQL Injection scanning operations.

    Adheres to a similar data pattern as your 'WrapperDigAgent', returning:
    {
        "task_id": ...,
        "step_id": ...,
        "step_description": ...,
        "result": ...,
        "metadata": ...,
        "error": ...
    }
    """

    def __init__(
        self,
        agent_id: str = "agent_sql_injection_tester",
        kafka_bootstrap_servers: str = "localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

        self.async_anthropic = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def setup_task_logger(self):
        """Setup a dedicated task logger, similar to other agents."""
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


    async def _parse_scan_parameters_async(self, task_description: str) -> ScanParameters:
        """
        Extract scan parameters using the *async* Anthropics client.
        Converts the LLM response into a Pydantic model (ScanParameters).
        """
        try:
            system_prompt = (
                "You are an offensive security and penetration testing expert. "
                "Extract scan parameters from the user's task description as JSON: "
                " - target_url (string) \n"
                " - paths (list of strings)\n"
                " - description (string, optional)\n"
                "Remove credentials or sensitive data."
            )

            response = await self.async_anthropic.messages.create(
                model="claude-3-5-sonnet-latest",
                system=system_prompt,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": f"{task_description}\n\nPlease return valid JSON with the specified fields."
                    }
                ],
                temperature=0.2,
            )

            # Extract content from the response
            content = response.content[0].text
            parsed_data = json.loads(content)
            return ScanParameters(**parsed_data)

        except Exception as e:
            logger.error(f"[SQL Injection Tester] Parameter parsing failed: {str(e)}")
            raise

    async def _execute_scan_async(self, params: ScanParameters) -> Dict[str, Any]:
        """
        Execute the security scan with given parameters in an *async* context.
        """
        try:
            scanner = SecurityScanner(
                client=self.async_anthropic,
                scan_target=ScanTarget(
                    url=params.target_url,
                    paths=params.paths,
                    description=params.description,
                ),
            )

            # Await the async scan directly (instead of asyncio.run).
            all_results, all_errors = await scanner.scan()

            self.task_logger.info(
                "\n***************** SQL Injection Scan Results ***********************\n"
            )
            self.task_logger.info(f"Scan target: {params.target_url}")
            self.task_logger.info(f"Scan paths: {params.paths}")
            self.task_logger.info(f"Injection scan results: {all_results}")
            self.task_logger.info(f"Error scan results: {all_errors}")
            self.task_logger.info("\n****************************************************\n")

            return {
                "findings": all_results + all_errors,
                "paths_tested": len(all_results + all_errors),
                "total_parameters": sum(len(r.parameters_tested) for r in all_results + all_errors)
            }

        except Exception as e:
            logger.error(f"[SQL Injection Tester] Scan execution failed: {str(e)}")
            raise

    def _assess_severity(self, finding: Dict[str, Any]) -> str:
        """Assess finding severity with basic logic."""
        # Example logic. Adjust as needed.
        finding_str = json.dumps(finding).lower()
        if "error" in finding_str:
            return "high"
        elif "union" in finding_str:
            return "high"
        elif "blind" in finding_str:
            return "medium"
        else:
            return "low"

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
    
        # Validate that task has the required fields
        if not isinstance(task, dict) or "task_id" not in task or "step_id" not in task:
            logger.error("[SQL Injection Tester] Invalid task format")
            return self._create_task_result(
                task.get("task_id"),
                task.get("step_id"),
                error="Invalid task format",
            )

        task_id = task["task_id"]
        step_id = task["step_id"]
        step_description = task.get("description", "")

        if targets := task.get("targets"):
            step_description += f"\n\nTargets: {targets} \n\n"

        if context := task.get("context"):
            step_description += f"\n\nPrevious step execution context(Use to guide yourself): {context} \n\n"

        try:
            self.task_logger.info(
                f"[SQL Injection Tester] Processing task_id={task_id}, step_id={step_id}"
            )

            # 1. Parse scan parameters (async).
            scan_params = await self._parse_scan_parameters_async(step_description)

            # 2. Execute scan (async).
            start_time = datetime.now(timezone.utc)
            scan_results = await self._execute_scan_async(scan_params)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.task_logger.info("\n\n\nScan results: %s \n\n\n", scan_results)


            # 4. Prepare standardized response.
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                result=f"{scan_results}",
                metadata={
                    "target": scan_params.target_url,
                    "paths_tested": len(scan_params.paths),
                    "timestamp": start_time.isoformat(),
                    "execution_time": execution_time,
                },
            )

        except Exception as e:
            logger.error(f"[SQL Injection Tester] Error processing task: {str(e)}")
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                error=str(e),
            )
