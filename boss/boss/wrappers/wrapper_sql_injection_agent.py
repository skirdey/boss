import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic
from pydantic import BaseModel, Field

from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ScanParameters(BaseModel):
    """Model for security scan parameters"""

    target_url: str = Field(..., description="Target URL to scan")
    paths: List[str] = Field(..., description="List of paths to test")
    description: Optional[str] = Field(description="Scan description")


class ScanFinding(BaseModel):
    """Model for individual scan findings"""

    parameter: str
    injection_type: str
    details: Dict[str, Any]
    timestamp: str
    severity: str


class WrapperSecurityScanner(WrapperAgent):
    """Wrapper agent for security scanning operations"""

    def __init__(
        self,
        agent_id="agent_security_scanner",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

        self.anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _parse_scan_parameters(self, task_description: str) -> ScanParameters:
        """Extract scan parameters using LLM"""
        try:
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                temperature=0,
                system="Security researcher extracting scan parameters from task description",
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
            parsed = json.loads(response.content)
            return ScanParameters(**parsed)

        except Exception as e:
            logger.error(f"Parameter parsing failed: {str(e)}")
            raise

    def _validate_target(self, target_url: str) -> bool:
        """Validate target URL"""
        from urllib.parse import urlparse

        try:
            # Basic URL validation
            result = urlparse(target_url)
            return all([result.scheme, result.netloc])

            # Check if target is in allowed domains/ranges
            allowed_targets = ["127.0.0.1", "localhost", ".local", ".test", ".example"]

            return any(target in target_url for target in allowed_targets)

        except Exception:
            return False

    async def _execute_scan(self, params: ScanParameters) -> Dict[str, Any]:
        """Execute security scan with given parameters"""
        try:
            scanner = SecurityScanner(
                client=self.anthropic_client,
                scan_target=ScanTarget(
                    url=params.target_url,
                    paths=params.paths,
                    description=params.description,
                ),
            )

            results = await scanner.scan()

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

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
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

            # Validate target
            if not self._validate_target(scan_params.target_url):
                return {
                    "task_id": str(task_id),
                    "result": f"Invalid target: {scan_params.target_url}",
                    "note": "Target validation failed",
                }

            # Execute scan
            start_time = datetime.now(timezone.utc)
            scan_results = await self._execute_scan(scan_params)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Prepare result
            result = {
                "task_id": str(task_id),
                "agent_id": self.agent_id,
                "result": scan_results,
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


def create_scanner_agent(
    kafka_servers: str = "localhost:9092", agent_id: str = "agent_security_scanner"
) -> WrapperSecurityScanner:
    """Factory function to create scanner agent"""
    return WrapperSecurityScanner(
        agent_id=agent_id, kafka_bootstrap_servers=kafka_servers
    )
