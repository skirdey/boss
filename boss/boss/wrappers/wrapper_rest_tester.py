import asyncio
import sys
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.request import Request

import aiohttp
import jwt
from pydantic import BaseModel, Field

from boss.utils import serialize_task_to_string
from boss.wrappers.wrapper_agent import WrapperAgent


JWT_INSTANCE = jwt.JWT()


if sys.platform.startswith("win"):
    # Force asyncio to use SelectorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class TestScenario(BaseModel):
    """Model for test scenario"""

    scenario: str = Field(..., description="Test scenario name")
    description: str = Field(..., description="Test scenario description")
    url: str = Field(..., description="URL to test (must be based on description or targets)")
    method: str = Field(..., description="HTTP method to use")
    headers: Optional[Dict[str, str]] = Field(description="HTTP headers")
    body: str = Field(..., description="HTTP body")
    auth_type: str = Field(..., description="Authentication type")
    auth_params: Optional[Dict[str, str]] = Field(
        description="Authentication parameters"
    )
    test_scenario: str = Field(..., description="Test scenario name")


class RESTRequestCommand(BaseModel):
    """Model for REST request command parameters"""

    test_scenarios: List[TestScenario]


class WrapperRESTTesterAgent(WrapperAgent):
    """
    Asynchronous REST Tester agent that attempts to find vulnerabilities
    (SQL injection, XSS, path traversal, etc.) rather than merely checking pass/fail.
    """

    def __init__(
        self,
        agent_id: str = "agent_rest_tester",
        kafka_bootstrap_servers: str = "localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    def setup_task_logger(self):
        """Setup a dedicated task logger."""
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

    async def _call_openai_api_for_rest(self, prompt: str) -> RESTRequestCommand:
        """
        Asynchronously call OpenAI (or similar) API with structured output parsing for
        vulnerability test scenarios generation.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            # Example instructions that explicitly ask for vulnerability tests.
            system_prompt = """
                You are a security-focused assistant. Given a user description of the endpoint or API,
                generate 3-10 REST API test scenarios to attempt to discover common vulnerabilities:
                - SQL injection
                - XSS (Cross-Site Scripting)
                - Path Traversal
                - Authentication/Authorization bypass
                - etc.

                For each scenario:
                1. Provide a descriptive name and scenario ID (test_scenario).
                2. Include any malicious parameters or payload in the body or query string, if applicable.
                3. Set the appropriate HTTP method (GET, POST, PUT, etc.).
                4. If authentication is needed, indicate auth type (JWT, Basic, Bearer, etc.) and put dummy or invalid tokens.
            """

            completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format=RESTRequestCommand,
            )

            parsed_command: RESTRequestCommand = completion.choices[0].message.parsed
            logger.info(f"[REST Vuln Tester] Parsed command parameters: {parsed_command}")
            return parsed_command

        except Exception as e:
            logger.error(f"[REST Vuln Tester] LLM request failed: {str(e)}")
            raise

    def generate_test_token(
        self,
        auth_type: str,
        test_scenario: str,
        auth_params: Dict[str, str],
    ) -> str:
        """
        Generate test tokens based on the scenario, including invalid or malformed tokens for vulnerability testing.
        """
        if auth_type.lower() in ["jwt", "bearer"]:
            if "malformed" in test_scenario.lower():
                return "malformed.jwt.token"
            elif "expired" in test_scenario.lower():
                # Generate an expired JWT token
                payload = {
                    "exp": datetime.now(timezone.utc).timestamp() - 3600,
                    **auth_params,
                }
                return JWT_INSTANCE.encode(payload, jwt.jwk.OctetJWK(b"secret"), alg="HS256")
            else:
                # Generate a valid JWT token
                payload = {
                    "exp": datetime.now(timezone.utc).timestamp() + 3600,
                    **auth_params,
                }
                return JWT_INSTANCE.encode(payload, jwt.jwk.OctetJWK(b"secret"), alg="HS256")

        elif auth_type.lower() == "oauth":
            if "malformed" in test_scenario.lower():
                return "invalid_oauth_token"
            return auth_params.get("token", "default_oauth_token")

        elif auth_type.lower() == "basic":
            # Potentially encode dummy username/password, if required
            return auth_params.get("token", "dGVzdDp0ZXN0")  # base64("test:test")

        return auth_params.get("token", "")

    async def execute_rest_request(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Asynchronously execute the REST request for a vulnerability scenario
        and analyze the response for signs of vulnerabilities.
        """
        try:
            self.task_logger.info(
                f"Executing REST vulnerability test: {scenario.method} {scenario.url}"
            )

            # Prepare headers
            headers = dict(scenario.headers) if scenario.headers else {}

            # Handle authentication
            if scenario.auth_type:
                token = self.generate_test_token(
                    scenario.auth_type, scenario.test_scenario, scenario.auth_params or {}
                )
                if scenario.auth_type.lower() in ["bearer", "jwt", "oauth"]:
                    headers["Authorization"] = f"Bearer {token}"
                elif scenario.auth_type.lower() == "basic":
                    headers["Authorization"] = f"Basic {token}"
                # Add more auth schemes as needed

            # Prepare the request body
            data = json.dumps(scenario.body).encode("utf-8") if scenario.body else None
            request = Request(
                scenario.url, data=data, headers=headers, method=scenario.method
            )

            start_time = datetime.now(timezone.utc)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=request.get_method(),
                        url=request.full_url,
                        headers=dict(request.headers),
                        data=request.data,
                    ) as response:
                        resp_body = await response.text()
                        result_data = {
                            "response_body": resp_body,
                            "status_code": response.status,
                            "response_headers": dict(response.headers),
                        }
            except aiohttp.ClientError as e:
                return {
                    "error": f"Connection error: {str(e)}",
                    "scenario": scenario.test_scenario,
                }

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Basic vulnerability analysis: 
            #   Check if the response body contains suspicious stack traces, DB error messages, 
            #   or reflection of our malicious payload.
            response_body = result_data["response_body"]
            status_code = result_data["status_code"]
            response_headers = result_data["response_headers"]

            # A naive check for suspicious patterns (illustrative only!)
            vulnerability_findings = []
            suspicious_indicators = [
                "syntax error",
                "unclosed quotation mark",
                "you have an error in your sql syntax",
                "<script>",
                "../etc/passwd",
                "ORA-",
                "Traceback (most recent call last):",
            ]
            for indicator in suspicious_indicators:
                if indicator.lower() in response_body.lower():
                    vulnerability_findings.append(
                        f"Detected possible vulnerability indicator: '{indicator}'"
                    )

            result = {
                "status_code": status_code,
                "response_headers": response_headers,
                "response_body": response_body,
                "execution_time": execution_time,
                "scenario": scenario.test_scenario,
                "vulnerability_findings": vulnerability_findings,
            }

            self.task_logger.info(
                f"Request completed in {execution_time:.2f}s with status {status_code}"
            )
            return result

        except Exception as e:
            self.task_logger.error(
                f"An error occurred during request execution: {str(e)}"
            )
            return {
                "error": f"An error occurred during request execution: {str(e)}",
                "scenario": scenario.test_scenario,
            }

    def summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize a list of vulnerability test results.
        """
        summary = {
            "total_tests": len(results),
            "passed": 0,
            "failed": 0,
            "potential_vulns": 0,
            "error_details": [],
        }

        for result in results:
            logger.info(f"[Result] {result}")

            # If there's a connection error or HTTP >= 400, label as "failed"
            if "error" in result or result.get("status_code", 0) >= 400:
                summary["failed"] += 1
                summary["error_details"].append(
                    {
                        "scenario": result.get("scenario"),
                        "error": result.get("error"),
                        "status_code": result.get("status_code"),
                        "response_body": result.get("response_body"),
                        "response_headers": result.get("response_headers"),
                        "vulnerability_findings": result.get("vulnerability_findings", []),
                    }
                )
            else:
                summary["passed"] += 1

            # If any vulnerability indicators are found, mark it
            findings = result.get("vulnerability_findings", [])
            if findings:
                summary["potential_vulns"] += 1

        self.task_logger.info(f"Summary of test results: {summary}")
        return summary

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously process a vulnerability-focused REST testing task,
        returning results in the same structure as the Dig agent.
        """
        self.task_logger.info("************ REST VULN TESTER AGENT ************")

        if not isinstance(task, dict) or "task_id" not in task or "step_id" not in task:
            logger.error("[REST Vuln Tester] Invalid task format")
            return self._create_task_result(
                task.get("task_id"),
                task.get("step_id"),
                error="Invalid task format",
            )

        try:
            task_id = task["task_id"]
            step_id = task["step_id"]
            step_description = task.get("description", "")

            self.task_logger.info(
                f"[REST Vuln Tester] Processing task_id={task_id}, step_id={step_id}"
            )

            if targets := task.get("targets", []):
                step_description += f"\n\nTargets: {targets} \n\n"

            if context := task.get("context", ""):
                step_description += f"\n\nPrevious step execution context(Use to guide yourself): {context} \n\n"

            # 1. Call LLM to generate vulnerability scenarios.
            parsed_command = await self._call_openai_api_for_rest(step_description)

            # 2. Execute each scenario in sequence (could also be parallel).
            results = []
            for scenario in parsed_command.test_scenarios:
                self.task_logger.info(
                    f"[REST Vuln Tester] Executing scenario: {scenario.test_scenario}"
                )
                request_result = await self.execute_rest_request(scenario)
                results.append(request_result)

            # 3. Summarize results (including vulnerability indicators).
            summary = self.summarize_results(results)

            # 5. Return standardized response.
            return self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                step_description=step_description,
                result=serialize_task_to_string(summary),
                metadata={
                    "summary": serialize_task_to_string(summary),
                    "request_results": serialize_task_to_string(results),
                },
            )

        except Exception as e:
            logger.error(f"[REST Vuln Tester] Error processing task: {str(e)}")
            return self._create_task_result(
                task.get("task_id"),
                task.get("step_id"),
                error=str(e),
            )
