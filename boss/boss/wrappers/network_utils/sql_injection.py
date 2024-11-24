from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import anthropic
import httpx

logger = logging.getLogger(__name__)


# Set logs to pink color
class PinkFormatter(logging.Formatter):
    """Custom formatter to output logs in pink color"""

    PINK = "\033[95m"
    RESET = "\033[0m"

    def format(self, record):
        message = super().format(record)
        return f"{self.PINK}{message}{self.RESET}"


# Set up the custom pink formatter for the logger
handler = logging.StreamHandler()
handler.setFormatter(
    PinkFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class ScanTarget:
    """Target configuration"""

    url: str
    paths: List[str]
    description: Optional[str] = None

    def __post_init__(self):
        if not self.url:
            raise ValueError("URL cannot be empty")
        if not self.paths:
            raise ValueError("At least one path must be provided")


@dataclass
class TestParameter:
    """Parameter test configuration"""

    name: str
    value: str
    injection_type: str
    expected_pattern: str


@dataclass
class ScanResult:
    """Scan result data"""

    target: str
    path: str
    timestamp: str
    findings: List[Dict[str, Any]]
    parameters_tested: List[TestParameter]
    errors: List[str]
    success: bool


class InjectionType(Enum):
    """Types of SQL injection patterns to test"""

    ERROR_BASED = "error_based"
    BOOLEAN_BASED = "boolean_based"
    TIME_BASED = "time_based"
    UNION_BASED = "union_based"
    STACKED = "stacked_queries"


class SecurityScanner:
    """LLM-powered security scanner for educational environments"""

    def __init__(
        self,
        client: anthropic.Anthropic,
        scan_target: ScanTarget,
        concurrency: int = 3,
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        if concurrency < 1:
            raise ValueError("Concurrency must be positive")
        if timeout < 1:
            raise ValueError("Timeout must be positive")

        self.client = client
        self.target = scan_target
        self.concurrency = concurrency
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.http_client = httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, verify=verify_ssl
        )

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.http_client.aclose()

    async def scan(self) -> Tuple[List[ScanResult], List[str]]:
        """Run full scan across all paths"""
        all_results = []
        all_errors = []

        try:
            sem = asyncio.Semaphore(self.concurrency)
            tasks = []

            for path in self.target.paths:
                task = self._scan_path_with_semaphore(sem, path)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            logger.info(f"SQL Injection Scan results: {results}")

            for result in results:
                if isinstance(result, Exception):
                    error_msg = f"Scan error: {str(result)}"
                    self.logger.error(error_msg)
                    all_errors.append(error_msg)
                else:
                    all_results.append(result)

            return all_results, all_errors

        except Exception as e:
            error_msg = f"Scan failed: {str(e)}"
            self.logger.error(error_msg)
            return [], [error_msg]

    async def _scan_path_with_semaphore(
        self, sem: asyncio.Semaphore, path: str
    ) -> ScanResult:
        """Execute scan for single path with concurrency control"""
        async with sem:
            try:
                async with asyncio.timeout(self.timeout):
                    return await self._scan_path(path)
            except asyncio.TimeoutError:
                return ScanResult(
                    target=self.target.url,
                    path=path,
                    timestamp=datetime.now().isoformat(),
                    findings=[],
                    parameters_tested=[],
                    errors=[f"Scan timed out after {self.timeout} seconds"],
                    success=False,
                )

    async def _scan_path(self, path: str) -> ScanResult:
        """Scan single path for vulnerabilities"""
        timestamp = datetime.now().isoformat()
        findings = []
        errors = []
        parameters_tested = []

        try:
            test_params = await self._generate_test_parameters(path)
            parameters_tested.extend(test_params)

            for param in test_params:
                try:
                    result = await self._test_parameter(path, param)
                    if result:
                        findings.append(
                            {
                                "parameter": param.name,
                                "injection_type": param.injection_type,
                                "details": result,
                            }
                        )
                except Exception as e:
                    errors.append(f"Parameter test failed: {str(e)}")

        except Exception as e:
            errors.append(f"Path scan failed: {str(e)}")

        return ScanResult(
            target=self.target.url,
            path=path,
            timestamp=timestamp,
            findings=findings,
            parameters_tested=parameters_tested,
            errors=errors,
            success=len(errors) == 0,
        )

    async def _generate_test_parameters(self, path: str) -> List[TestParameter]:
        """Use LLM to generate intelligent test parameters"""
        try:
            prompt = self._build_parameter_prompt(path)
            response = await self.client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=8192,
                system="You are expert in offensive security and penetration testing. Generate test parameters for SQL injection scan.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Safely extract content from the response
            content = response.content[0].text if response.content else ""

            logger.info(f"Generated parameters: {content}")

            parameters = self._parse_parameters(content)
            return self._validate_parameters(parameters)

        except Exception as e:
            self.logger.error(f"Parameter generation failed: {str(e)}")
            return []

    def _build_parameter_prompt(self, path: str) -> str:
        """Build prompt for parameter generation"""
        return f"""
        Analyzing endpoint in controlled security lab: {path}

        Generate test parameters for understanding SQL injection patterns.
        Consider:
        1. Common parameter names and locations
        2. Different injection types
        3. Expected response patterns
        
        Format response as JSON with structure:
        {{
            "parameters": [
                {{
                    "name": "<param_name>",
                    "value": "<test_value>",
                    "injection_type": "<type>",
                    "expected_pattern": "<pattern>"
                }}
            ]
        }}
        """

    def _parse_parameters(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into parameter objects"""
        try:
            # Extract JSON block from response
            parsed = json.loads(llm_response)
            return parsed.get("parameters", [])
        except Exception as e:
            self.logger.error(f"Parameter parsing failed: {str(e)}")
            return []

    def _validate_parameters(
        self, parameters: List[Dict[str, Any]]
    ) -> List[TestParameter]:
        """Validate and convert parameter dictionaries"""

        validated = []
        required_fields = {"name", "value", "injection_type", "expected_pattern"}

        for param in parameters:
            try:
                if not all(field in param for field in required_fields):
                    continue

                validated.append(
                    TestParameter(
                        name=str(param["name"]),
                        value=str(param["value"]),
                        injection_type=str(param["injection_type"]),
                        expected_pattern=str(param["expected_pattern"]),
                    )
                )
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Invalid parameter: {str(e)}")
        return validated

    async def _test_parameter(
        self, path: str, parameter: TestParameter
    ) -> Optional[Dict[str, Any]]:
        """Test single parameter for vulnerability patterns"""

        logger.info(f"Testing parameter: {parameter} and path: {path}")

        try:
            url = urljoin(self.target.url, path)

            logger.info(f"Testing URL: {url}")

            findings = {}

            # Test with GET
            get_response = await self.http_client.get(
                url, params={parameter.name: parameter.value}
            )

            logger.info(f"GET response: {get_response}")

            if self._analyze_response(get_response, parameter.expected_pattern):
                findings["get_vulnerable"] = True
                findings["get_details"] = self._extract_response_details(get_response)

            # Test with POST
            post_response = await self.http_client.post(
                url, json={parameter.name: parameter.value}
            )
            if self._analyze_response(post_response, parameter.expected_pattern):
                findings["post_vulnerable"] = True
                findings["post_details"] = self._extract_response_details(post_response)

            return findings if findings else None

        except Exception as e:
            self.logger.warning(f"Parameter test failed: {str(e)}")
            return None

    def _analyze_response(
        self, response: httpx.Response, expected_pattern: str
    ) -> bool:
        """Analyze response for vulnerability patterns"""
        try:
            content = response.text.lower()
            pattern = expected_pattern.lower()

            return (
                pattern in content
                or "sql" in content
                or "error" in content
                or "exception" in content
                or response.status_code >= 500
            )
        except Exception:
            return False

    def _extract_response_details(self, response: httpx.Response) -> Dict[str, Any]:
        """Extract relevant details from response"""
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body_preview": response.text[:500] if response.text else "",
            "response_time": response.elapsed.total_seconds(),
        }
