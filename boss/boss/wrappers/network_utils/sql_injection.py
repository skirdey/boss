import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import anthropic
import httpx
import yaml


@dataclass
class ScanTarget:
    """Target configuration"""

    url: str
    paths: List[str]
    description: Optional[str] = None


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
        client: anthropic.Client,
        scan_target: ScanTarget,
        concurrency: int = 3,
        timeout: int = 30,
    ):
        self.client = client
        self.target = scan_target
        self.concurrency = concurrency
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            verify=False,  # For testing internal labs only
        )

    async def scan(self) -> List[ScanResult]:
        """Run full scan across all paths"""
        try:
            all_results = []
            sem = asyncio.Semaphore(self.concurrency)
            tasks = []

            for path in self.target.paths:
                task = self._scan_path_with_semaphore(sem, path)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Scan error: {str(result)}")
                else:
                    all_results.append(result)

            return all_results

        except Exception as e:
            self.logger.error(f"Scan failed: {str(e)}")
            return []

    async def _scan_path_with_semaphore(
        self, sem: asyncio.Semaphore, path: str
    ) -> ScanResult:
        """Execute scan for single path with concurrency control"""
        async with sem:
            return await self._scan_path(path)

    async def _scan_path(self, path: str) -> ScanResult:
        """Scan single path for vulnerabilities"""
        timestamp = datetime.now().isoformat()
        findings = []
        errors = []
        parameters_tested = []

        try:
            # Generate test parameters using LLM
            test_params = await self._generate_test_parameters(path)
            parameters_tested.extend(test_params)

            # Execute tests for each parameter
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
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                temperature=0,
                system="Security researcher analyzing controlled test environment",
                messages=[{"role": "user", "content": prompt}],
            )

            parameters = self._parse_parameters(response.content)
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
        
        Format response as YAML with structure:
        parameters:
          - name: <param_name>
            value: <test_value>
            injection_type: <type>
            expected_pattern: <pattern>
        
        Focus on educational testing patterns.
        """

    def _parse_parameters(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into parameter objects"""
        try:
            # Extract YAML block from response
            yaml_block = llm_response.strip().split("```yaml")[-1].split("```")[0]
            parsed = yaml.safe_load(yaml_block)
            return parsed.get("parameters", [])
        except Exception as e:
            self.logger.error(f"Parameter parsing failed: {str(e)}")
            return []

    def _validate_parameters(
        self, parameters: List[Dict[str, Any]]
    ) -> List[TestParameter]:
        """Validate and convert parameter dictionaries"""
        validated = []
        for param in parameters:
            try:
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
        try:
            url = urljoin(self.target.url, path)

            # Test with GET and POST
            async with self.http_client as client:
                get_response = await client.get(
                    url, params={parameter.name: parameter.value}
                )

                post_response = await client.post(
                    url, json={parameter.name: parameter.value}
                )

            # Analyze responses
            findings = {}

            if self._analyze_response(get_response, parameter.expected_pattern):
                findings["get_vulnerable"] = True
                findings["get_details"] = self._extract_response_details(get_response)

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

            # Look for error patterns
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
            "body_preview": response.text[:500],
            "response_time": response.elapsed.total_seconds(),
        }


async def create_scanner(
    api_key: str, target_url: str, paths: List[str], description: Optional[str] = None
) -> SecurityScanner:
    """Factory function to create scanner instance"""
    client = anthropic.Client(api_key=api_key)
    target = ScanTarget(target_url, paths, description)
    return SecurityScanner(client, target)
