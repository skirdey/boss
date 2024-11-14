import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic
from pydantic import BaseModel, Field

from boss.utils import serialize_task_to_string
from boss.wrappers.network_utils.subdomain_scanner import SubdomainScanner
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SubdomainScanParameters(BaseModel):
    """Model for subdomain scan parameters"""

    target_domain: str = Field(..., description="Target domain to scan")
    wordlist: Optional[List[str]] = Field(description="Wordlist for subdomain enumeration")
    description: Optional[str] = Field(description="Scan description")


class SubdomainFinding(BaseModel):
    """Model for individual subdomain findings"""

    subdomain: str
    ip_address: Optional[str]
    timestamp: str


class WrapperSubdomainAgent(WrapperAgent):
    """Wrapper agent for subdomain exploration operations"""

    def __init__(
        self,
        agent_id="agent_subdomain_explorer",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

        self.async_anthropic = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _parse_scan_parameters(self, task_description: str) -> SubdomainScanParameters:
        """Extract scan parameters using LLM"""
        try:
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                system="You are expert in offensive security and penetration testing. You are extracting scan parameters from task description",
                messages=[{
                    "role": "user",
                    "content": f"""
                    Extract scan parameters from the following task description.
                    Remove any credentials or sensitive data.
                    Task: {task_description}

                    Return as JSON with fields:
                    - target_domain (string)
                    - wordlist (optional list of strings)
                    - description (optional string)
                    """,
                }]
            )

            # Parse response
            parsed = response.content <sup> </sup>.text
            return SubdomainScanParameters(**parsed)

        except Exception as e:
            logger.error(f"Parameter parsing failed: {str(e)}")
            raise


    def _execute_scan(self, params: SubdomainScanParameters) -> Dict[str, Any]:
        """Execute subdomain exploration with given parameters"""
        try:
            scanner = SubdomainScanner(
                client=self.async_anthropic,
                target_domain=params.target_domain,
                wordlist=params.wordlist,
                description=params.description,
            )

            # Run the async scan method synchronously
            results = asyncio.run(scanner.scan())
            logger.info(
                "\n***************** Subdomain Exploration Results ***********************\n"
            )
            logger.info(f"Target domain: {params.target_domain}")
            logger.info(f"Subdomain scan results: {results}")
            logger.info(f"Wordlist used: {params.wordlist}")
            logger.info("\n****************************************************\n")

            # Process and structure results
            findings = []
            for result in results:
                findings.append(
                    SubdomainFinding(
                        subdomain=result["subdomain"],
                        ip_address=result.get("ip_address"),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )

            return {
                "findings": [finding.dict() for finding in findings],
                "subdomains_found": len(findings),
                "scan_duration": sum(1 for r in results if r["success"]),
            }

        except Exception as e:
            logger.error(f"Scan execution failed: {str(e)}")
            raise

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
                    "target": scan_params.target_domain,
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