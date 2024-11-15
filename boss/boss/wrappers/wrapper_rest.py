import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import jwt
from pydantic import BaseModel, Field, ValidationError

from boss.utils import serialize_task_to_string
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TestScenario(BaseModel):
    """Model for test scenario"""

    scenario: str = Field(..., description="Test scenario name")
    description: str = Field(..., description="Test scenario description")
    url: str = Field(..., description="URL to test")
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


class WrapperRESTTestAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_rest_tester",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.agent_id = agent_id
        self.setup_logging()

    def setup_logging(self):
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

    def _call_openai_api(self, prompt: str) -> RESTRequestCommand:
        """Call OpenAI API with structured output parsing for REST request generation"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            system_prompt = """
                Extract REST API test parameters from the task description. Make sure to:
                1. **Identify the exact endpoint URL and HTTP method.**
                2. Determine authentication requirements.
                3. Generate between 1 to 10 appropriate test scenarios based on the HTML response and common REST patterns.
                4. Include relevant headers and body parameters.

                Common test scenarios:
                - normal: Standard API call with valid authentication.
                - malformed_token: Test with incorrectly formatted authentication token.
                - expired_token: Test with an expired authentication token.
                - missing_auth: Test without required authentication.
                - invalid_params: Test with invalid request parameters.
                - ... (add more as needed)

                **Ensure that the extracted URL is valid and includes the correct hostname and path.**
                """

            completion = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=RESTRequestCommand,
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

    def summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize the list of test results."""
        summary = {
            "total_tests": len(results),
            "passed": 0,
            "failed": 0,
            "error_details": [],
        }

        for result in results:
            logger.info(f"Result: {result}")
            if "error" in result or result.get("status_code", 0) >= 400:
                summary["failed"] += 1
                summary["error_details"].append(
                    {
                        "scenario": result.get("scenario"),
                        "error": result.get("error"),
                        "status_code": result.get("status_code"),
                        "response_body": result.get("response_body"),
                        "response_headers": result.get("response_headers"),
                    }
                )
            else:
                summary["passed"] += 1

        self.task_logger.info(f"Summary of test results: {summary}")
        return summary

    def generate_test_token(
        self, auth_type: str, test_scenario: str, auth_params: Dict[str, str]
    ) -> str:
        """Generate test tokens based on the scenario"""
        if auth_type == "JWT":
            if test_scenario == "malformed_token":
                return "malformed.jwt.token"
            elif test_scenario == "expired_token":
                # Generate an expired JWT token
                payload = {
                    "exp": datetime.now(timezone.utc).timestamp()
                    - 3600,  # Expired 1 hour ago
                    **auth_params,
                }
                return jwt.encode(payload, "secret", algorithm="HS256")
            else:
                # Generate a valid JWT token
                payload = {
                    "exp": datetime.now(timezone.utc).timestamp()
                    + 3600,  # Valid for 1 hour
                    **auth_params,
                }
                return jwt.encode(payload, "secret", algorithm="HS256")

        elif auth_type == "OAuth":
            if test_scenario == "malformed_token":
                return "invalid_oauth_token"
            else:
                return auth_params.get("token", "default_oauth_token")

        return auth_params.get("token", "")

    def execute_request(self, command: TestScenario) -> Dict[str, Any]:
        """Execute the REST request with the given parameters"""
        try:
            self.task_logger.info(
                f"Executing REST request: {command.method} {command.url}"
            )

            # headers is a list of tuples
            if command.headers:
                headers = command.headers
            else:
                headers = {}

            # Handle authentication
            if command.auth_type:
                token = self.generate_test_token(
                    command.auth_type, command.test_scenario, command.auth_params or {}
                )

                if command.auth_type in ["Bearer", "JWT", "OAuth"]:
                    headers["Authorization"] = f"Bearer {token}"
                elif command.auth_type == "Basic":
                    # Handle Basic auth if needed
                    pass

            # Prepare the request
            data = json.dumps(command.body).encode("utf-8") if command.body else None
            request = Request(
                command.url, data=data, headers=headers, method=command.method
            )

            start_time = datetime.now(timezone.utc)

            try:
                with urlopen(request) as response:
                    response_data = response.read().decode("utf-8")
                    status_code = response.status
                    response_headers = dict(response.headers)
            except HTTPError as e:
                response_data = e.read().decode("utf-8")
                status_code = e.code
                response_headers = dict(e.headers)
            except URLError as e:
                return {
                    "error": f"Connection error: {str(e)}",
                    "scenario": command.test_scenario,
                }

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = {
                "status_code": status_code,
                "response_headers": response_headers,
                "response_body": response_data,
                "execution_time": execution_time,
                "scenario": command.test_scenario,
            }

            self.task_logger.info(
                f"Request completed in {execution_time:.2f} seconds with status {status_code}"
            )
            return result

        except Exception as e:
            self.task_logger.error(
                f"An error occurred during request execution: {str(e)}"
            )
            return {
                "error": f"An error occurred during request execution: {str(e)}",
                "scenario": command.test_scenario,
            }

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(task, dict) or "_id" not in task:
            logger.error("Invalid task format")
            return {"success": False, "note": "Invalid task format"}

        try:
            task_id = task.get("_id")
            self.task_logger.info(f"Processing task with ID: {task_id}")

            current_step_index = task.get("current_step_index")
            steps = task.get("steps", [])
            if current_step_index is None or current_step_index >= len(steps):
                logger.error("Invalid current_step_index")
                return {"task_id": task_id, "error": "Invalid current_step_index"}

            current_step = steps[current_step_index]
            step_description = current_step.get("step_description", "")

            # Generate request parameters using LLM
            parsed_command = self._call_openai_api(step_description)

            # Execute requests for all test scenarios
            results = []
            for scenario in parsed_command.test_scenarios:
                logger.info(f"Executing scenario: {scenario}")

                command = scenario
                request_result = self.execute_request(command)
                results.append(request_result)

            # Summarize the results
            summary = self.summarize_results(results)

            # Prepare the result
            result = {
                "task_id": str(task_id),
                "agent_id": self.agent_id,
                "result": serialize_task_to_string(summary),
                "metadata": {
                    "summary": serialize_task_to_string(summary),
                    "request_results": serialize_task_to_string(results),
                },
            }

            return result

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return {
                "task_id": str(task.get("_id", "unknown")),
                "error": f"Validation error: {ve}",
                "note": "Invalid command parameters",
            }
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return {
                "task_id": str(task.get("_id", "unknown")),
                "error": str(e),
                "note": "Exception occurred during task execution",
            }
