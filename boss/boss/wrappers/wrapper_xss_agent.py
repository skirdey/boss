import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from boss.utils import serialize_task_to_string
from boss.wrappers.wrapper_agent import WrapperAgent

################################################################################
# 1. Load environment, configure logging
################################################################################

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False

################################################################################
# 2. Define a helper function to fetch the URL text using aiohttp
################################################################################


async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    """
    Fetch the URL using the provided aiohttp session and return response text.
    Returns an empty string if there's an exception or non-200 status.
    """
    try:
        async with session.get(url) as response:
            return await response.text()

    except Exception as e:
        logger.error(f"Exception fetching {url}: {e}")
        return ""


################################################################################
# 2. Define a Pydantic model to parse the user's request
################################################################################


class XssScanCommand(BaseModel):
    """
    Model for an XSS scanning command.
    This is what we'll parse from the user prompt via structured output.
    """

    target_url: str = Field(
        description="Full target URL (https or http) to scan for potential XSS vulnerabilities."
    )
    num_payloads: int = Field(
        description="Number of XSS payloads to generate/test using the LLM (if applicable).",
    )
    use_browser: bool = Field(
        description="Whether to attempt a headless browser check (Playwright).",
    )


################################################################################
# 3. Define the XSS-scanning agent
################################################################################


class WrapperXssAgent(WrapperAgent):
    def __init__(
        self,
        agent_id: str = "agent_xss_scanner",
        kafka_bootstrap_servers: str = "localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    ############################################################################
    # 3a. Parse the user's request from the BOSS system using structured output
    ############################################################################
    async def _call_openai_api(self, prompt: str) -> XssScanCommand:
        """
        Asynchronously call an LLM (e.g., OpenAI) with structured output to parse
        the XSS scanning command.
        """
        try:
            chat_completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You're given a step description. "
                            "We want a JSON that fits the XssScanCommand schema: "
                            "the target_url should be extracted from the step description in the form of a FQDN URL, if there is no URL in the step description, return an empty object!"
                            "the num_payloads should be at least 1 and at most 10!"
                            "the use_browser you can determine based on the step description."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=XssScanCommand,
            )

            content = chat_completion.choices[0].message.parsed
            logger.info(f"Received from LLM: {content}")

            return content

        except Exception as e:
            logger.error(f"Error parsing XSS scan command: {e}")
            return None

    async def _generate_llm_xss_payloads(
        self, html_content: str, num_payloads: int
    ) -> List[str]:
        """
        Generates additional XSS payloads using an LLM, incorporating HTML content.
        """
        prompt = f"""
            Given the following HTML content of a webpage:
    ```html
    {html_content}
    ```

    Generate {num_payloads} unique XSS (Cross-Site Scripting) payloads that
    could potentially exploit reflected XSS vulnerabilities on this page.
    Consider different input types, attributes, event handlers, and encodings.
    Only provide the payloads, one per line, no extra commentary.
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            lines = response.choices[0].message.content.strip().split("\n")
            # Strip surrounding whitespace from each payload
            lines = [l.strip() for l in lines if l.strip()]
            return lines
        except Exception as e:
            logger.error(f"Error with LLM payload generation: {e}")
            return []

    def _check_security_headers(self, response) -> None:
        headers = response.headers
        csp = headers.get("Content-Security-Policy")
        x_xss = headers.get("X-XSS-Protection")
        strict_transport = headers.get("Strict-Transport-Security")

        self.task_logger.info("[*] Security Headers:")
        if csp:
            self.task_logger.info(f"    - CSP detected: {csp}")
        else:
            self.task_logger.info("    - No CSP detected.")

        if x_xss:
            self.task_logger.info(f"    - X-XSS-Protection: {x_xss}")
        else:
            self.task_logger.info("    - No X-XSS-Protection header.")

        if strict_transport:
            self.task_logger.info(
                f"    - Strict-Transport-Security: {strict_transport}"
            )

    ############################################################################
    # 3b. The main XSS scanning logic
    ############################################################################
    async def execute_xss_scan(
        self,
        target_url: str,
        num_payloads: int,
        use_browser: bool,
    ) -> Dict[str, Any]:
        self.task_logger.info(f"Starting XSS scan for URL={target_url}")
        start_time = datetime.now(timezone.utc)

        # Use a single session for this entire scan
        async with aiohttp.ClientSession() as session:
            try:
                # Non-blocking GET
                html_content = await fetch(session, target_url)
                if not html_content:
                    self.task_logger.error(f"Failed to retrieve page {target_url}.")
                    return {"error": f"Failed to retrieve page: {target_url}"}

                self.task_logger.info(
                    f"Response body: {html_content[:200]} ..."
                )  # just preview
            except Exception as e:
                self.task_logger.error(f"Failed to retrieve page {target_url}: {e}")
                return {"error": f"Failed to retrieve page: {e}"}

            # Known baseline XSS payloads
            known_payloads = [
                "<script>alert(1)</script>",
                '";alert(1);//',
                "<img src=x onerror=alert(1)>",
                "<svg><script>alert(1)</script>",
                "'><script>alert(document.domain)</script>",
                "</script><script>alert('XSS')</script>",
                "<iframe src=javascript:alert(1)>",
                "<body onload=alert(1)>",
            ]

            # Generate additional LLM-based payloads
            additional_payloads = await self._generate_llm_xss_payloads(
                html_content=html_content, num_payloads=num_payloads
            )

            all_payloads = list(set(known_payloads + additional_payloads))

            discovered_params = self._extract_parameters_from_forms(html_content)

            # Replacing synchronous calls to `_test_xss` with an async approach
            vulnerabilities = await self._test_xss(
                session, target_url, discovered_params, all_payloads, use_browser
            )

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.task_logger.info(
            f"XSS scanning completed in {execution_time:.2f} seconds."
        )

        return {
            "target": target_url,
            "discovered_params": discovered_params,
            "vulnerabilities_found": vulnerabilities,
            "scan_duration_seconds": execution_time,
        }

    def _extract_parameters_from_forms(self, html_content: str) -> List[str]:
        soup = BeautifulSoup(html_content, "html.parser")
        params = set()

        # Extract inputs from form elements
        for form in soup.find_all("form"):
            for tag in form.find_all(["input", "select", "textarea", "button"]):
                name = tag.get("name")
                if name:
                    params.add(name)
                # also check 'id'
                id_attr = tag.get("id")
                if id_attr:
                    params.add(id_attr)
                # data-* attributes
                for attr in tag.attrs:
                    if attr.startswith("data-"):
                        params.add(attr)

            # extract query params from 'action'
            action = form.get("action")
            if action:
                parsed_url = urlparse(action)
                query_params = parse_qs(parsed_url.query)
                for param in query_params:
                    params.add(param)

        # Extract parameters from <script> tags
        for script in soup.find_all("script"):
            if script.string:
                # naive approach to find ?key=value pairs
                matches = re.findall(r'\?([^#\'"]+)', script.string)
                for match in matches:
                    pairs = match.split("&")
                    for pair in pairs:
                        if "=" in pair:
                            key, _ = pair.split("=", 1)
                            params.add(key)

        # Extract parameters from links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            parsed_url = urlparse(href)
            query_params = parse_qs(parsed_url.query)
            for param in query_params:
                params.add(param)

        # Some common param names as heuristics
        common_params = [
            "id",
            "user",
            "uid",
            "token",
            "auth",
            "action",
            "type",
            "name",
            "page",
            "lang",
            "redirect",
            "callback",
        ]
        params.update(common_params)

        # Fallback
        if not params:
            params = {"test"}

        return list(params)

    def _encode_payloads(self, payload: str) -> List[str]:
        html_encoded = (
            payload.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        )
        url_encoded = requests.utils.quote(payload)
        return [payload, html_encoded, url_encoded]

    def _is_payload_reflected(self, html: str, payload: str) -> bool:
        # Plain check
        if payload in html:
            return True
        # Check HTML-encoded version
        html_encoded = (
            payload.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        )
        if html_encoded in html:
            return True
        return False

    async def _test_xss(
        self,
        session: aiohttp.ClientSession,
        target_url: str,
        parameters: List[str],
        payloads: List[str],
        use_browser: bool = False,
    ) -> Dict[str, List[Any]]:
        """
        Test each discovered parameter with each payload. We re-use the same
        aiohttp session to avoid overhead of multiple new connections.
        """
        vulnerable_params = {}
        parsed_url = urlparse(target_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        path = parsed_url.path or "/"

        for param_name in parameters:
            for payload in payloads:
                # We encode in multiple ways for better coverage
                for epayload in self._encode_payloads(payload):
                    # Create a query dict
                    test_params = {
                        p: (epayload if p == param_name else "normal")
                        for p in parameters
                    }
                    full_url = urljoin(base_url, path) + "?" + urlencode(test_params)

                    try:
                        async with session.get(full_url) as resp:
                            if resp.status == 200:
                                html = await resp.text()
                                if self._is_payload_reflected(html, epayload):
                                    executed = False
                                    # Use browser if needed, asynchronously
                                    if use_browser:
                                        # This is where you'd integrate playwright or similar
                                        pass

                                    if param_name not in vulnerable_params:
                                        vulnerable_params[param_name] = []
                                    vulnerable_params[param_name].append(
                                        (payload, epayload, executed)
                                    )
                                    self.task_logger.info(
                                        f"[+] Potential XSS in {param_name} with payload {payload} "
                                        f"(encoded: {epayload}). Executed: {executed}"
                                    )
                                else:
                                    self.task_logger.info(
                                        f"[-] No reflection for {param_name} with payload {payload}"
                                    )
                            else:
                                self.task_logger.info(
                                    f"[-] {full_url} returned {resp.status}, skipping."
                                )
                    except Exception as e:
                        self.task_logger.error(f"[-] Request error for {full_url}: {e}")

        return vulnerable_params

    ############################################################################
    # 3d. Validate user input
    ############################################################################
    def is_valid_target(self, target_url: str) -> bool:
        return target_url.lower().startswith(
            "http://"
        ) or target_url.lower().startswith("https://")

    ############################################################################
    # 3e. Orchestrate everything in process_task (as in the Dig example)
    ############################################################################
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.task_logger.info("**************** XSS SCANNER AGENT ****************")

        if not isinstance(task, dict) or "task_id" not in task or "step_id" not in task:
            logger.error("Invalid task format")
            return self._create_task_result(
                task.get("task_id"), task.get("step_id"), error="Invalid task format"
            )

        try:
            task_id = task["task_id"]
            step_id = task["step_id"]
            self.task_logger.info(
                f"Processing task with ID: {task_id} and step ID: {step_id}"
            )

            # Extract step description
            step_description = task.get("description", "")
            task_prompt = f"Current step:\n{step_description}"

            targets = task.get("targets", [])

            if targets:
                task_prompt += f"\nTargets:\n{targets}"

            # 1. Parse user command from the LLM / structured output
            parsed_command = await self._call_openai_api(task_prompt)
            if parsed_command is None:
                return self._create_task_result(
                    task_id=task_id,
                    step_id=step_id,
                    error="Error parsing XSS scan command",
                )

            self.task_logger.info(f"Parsed command parameters: {parsed_command}")

            # 2. Validate the target
            if not self.is_valid_target(parsed_command.target_url):
                return self._create_task_result(
                    task_id=task_id,
                    step_id=step_id,
                    error=f"Invalid target URL: {parsed_command.target_url}",
                )

            # 3. Execute the XSS scan
            scan_result = await self.execute_xss_scan(
                target_url=parsed_command.target_url,
                num_payloads=parsed_command.num_payloads,
                use_browser=parsed_command.use_browser,
            )

            # 4. Prepare and return the result to BOSS
            result = self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                step_description=step_description,
                result=serialize_task_to_string(scan_result),
                metadata={
                    "target_url": parsed_command.target_url,
                    "num_payloads": parsed_command.num_payloads,
                    "use_browser": parsed_command.use_browser,
                },
            )
            return result

        except ValidationError as ve:
            self.task_logger.error(f"Validation error: {ve}")
            return self._create_task_result(
                task_id=task.get("task_id"),
                step_id=task.get("step_id"),
                error=f"Validation error: {ve}",
            )
        except Exception as e:
            self.task_logger.error(f"Error processing task: {e}")
            return self._create_task_result(
                task_id=task.get("task_id"),
                step_id=task.get("step_id"),
                error=str(e),
            )
