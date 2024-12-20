import asyncio  # Added to handle async calls in sync context
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic
from pydantic import BaseModel, Field

from boss.utils import serialize_task_to_string
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

    target_url: str = Field(..., description="Target URL to scan")
    paths: List[str] = Field(
        ..., description="List of paths to test for SQL injections"
    )
    description: Optional[str] = Field(description="Scan description")


class ScanFinding(BaseModel):
    """Model for individual scan findings"""

    parameter: str
    injection_type: str
    details: Dict[str, Any]
    timestamp: str
    severity: str


class WrapperSQLInjectionAgent(WrapperAgent):
    """Wrapper agent for security scanning operations"""

    def __init__(
        self,
        agent_id="agent_sql_injection_tester",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

        self.async_anthropic = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _parse_scan_parameters(self, task_description: str) -> ScanParameters:
        """Extract scan parameters using LLM"""
        try:
            response = self.anthropic.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=8192,
                system="You are expert in offensive security and penetration testing. You are extracting scan parameters from task description",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                    Extract scan parameters from the following task description.
                    Remove any credentials or sensitive data.
                    Task: {task_description}

                    Return as JSON with fields:
                    - target_url (string)
                    - paths (list of strings)
                    - description (optional string)
                    """,
                    }
                ],
            )

            # Parse response
            parsed = response.content[0].text
            return ScanParameters(**parsed)

        except Exception as e:
            logger.error(f"Parameter parsing failed: {str(e)}")
            raise

    def _execute_scan(self, params: ScanParameters) -> Dict[str, Any]:
        """Execute security scan with given parameters"""
        try:
            scanner = SecurityScanner(
                client=self.async_anthropic,
                scan_target=ScanTarget(
                    url=params.target_url,
                    paths=params.paths,
                    description=params.description,
                ),
            )

            # Run the async scan method synchronously
            results = asyncio.run(scanner.scan())
            logger.info(
                "\n***************** Injection SQL Results ***********************\n"
            )
            logger.info(f"Scan target: {params.target_url}")
            logger.info(f"Injection scan results: {results}")
            logger.info(f"Scan paths: {params.paths}")
            logger.info("\n****************************************************\n")

            # Process and structure results
            findings = []
            for result in results:
                for finding in result.findings:
                    findings.append(
                        ScanFinding(
                            parameter=finding["parameter"],
                            injection_type=finding["injection_type"],
                            details=finding["details"],
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            severity=self._assess_severity(finding),
                        )
                    )

            return {
                "findings": [finding.dict() for finding in findings],
                "paths_tested": len(results),
                "total_parameters": sum(len(r.parameters_tested) for r in results),
                "scan_duration": sum(1 for r in results if r.success),
            }

        except Exception as e:
            logger.error(f"Scan execution failed: {str(e)}")
            raise

    def _assess_severity(self, finding: Dict[str, Any]) -> str:
        """Assess finding severity"""
        # Basic severity assessment logic
        if "error" in str(finding).lower():
            return "high"
        elif "union" in str(finding).lower():
            return "high"
        elif "blind" in str(finding).lower():
            return "medium"
        else:
            return "low"

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming task"""
        if not isinstance(task, dict) or "_id" not in task:
            logger.error("Invalid task format")
            return {"success": False, "note": "Invalid task format"}

        try:
            task_id = task.get("_id")
            self.task_logger.info(f"Processing task with ID: {task_id}")

            # Get current step
            current_step_index = task.get("current_step_index")
            steps = task.get("steps", [])
            if current_step_index is None or current_step_index >= len(steps):
                logger.error("Invalid current_step_index")
                return {"task_id": task_id, "error": "Invalid current_step_index"}

            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")

            # Parse scan parameters
            scan_params = self._parse_scan_parameters(step_description)

            # Execute scan
            start_time = datetime.now(timezone.utc)
            scan_results = self._execute_scan(scan_params)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Prepare result
            result = {
                "task_id": str(task_id),
                "agent_id": self.agent_id,
                "result": serialize_task_to_string(scan_results),
                "execution_time": execution_time,
                "metadata": {
                    "target": scan_params.target_url,
                    "paths_tested": len(scan_params.paths),
                    "timestamp": start_time.isoformat(),
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return {
                "task_id": str(task_id),
                "agent_id": self.agent_id,
                "error": str(e),
                "note": "Exception occurred during scan execution",
            }
