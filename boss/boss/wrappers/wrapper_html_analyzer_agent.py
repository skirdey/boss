import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel

from boss.models import TaskState
from boss.utils import get_iso_timestamp, serialize_task_to_string
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class HTMLAnalysisResult(BaseModel):
    """Model for HTML analysis results"""

    raw_html: str
    potential_vulnerabilities: List[Dict]
    paths: List[str]
    success: bool
    error: Optional[str]
    timestamp: datetime
    metrics: Dict = Dict[str, str]


class HTMLScanResult(BaseModel):
    """Model for overall scan results"""

    target: str
    analysis: HTMLAnalysisResult


class WrapperHTMLAnalyzerAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_html_analyzer",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.agent_id = agent_id

    async def fetch_html(self, target: str) -> Tuple[str, Dict[str, str]]:
        """Fetch HTML content and headers from target"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{target}"
                async with session.get(url, timeout=10) as response:
                    html = await response.text()
                    headers = {k.lower(): v for k, v in response.headers.items()}
                    return html, headers
        except Exception as e:
            logger.error(f"Error fetching HTML from {target} {str(e)}")
            raise

    def analyze_vulnerabilities(
        self, html_content: str, headers: Dict[str, str], url: str
    ) -> Tuple[List[Dict], List[str]]:
        """Analyze HTML content and headers for potential vulnerabilities"""
        vulnerabilities = []
        soup = BeautifulSoup(html_content, "html.parser")

        # Check for forms without CSRF protection
        forms = soup.find_all("form")
        for form in forms:
            if not form.find("input", {"name": ["csrf_token", "_token", "_csrf"]}):
                vulnerabilities.append(
                    {
                        "type": "CSRF",
                        "severity": "Medium",
                        "description": "Form found without CSRF protection",
                        "location": str(form)[:200],
                    }
                )

        # Check for input fields with potential XSS vulnerabilities
        inputs = soup.find_all(["input", "textarea"])
        for input_field in inputs:
            if input_field.get("type") not in ["hidden", "submit", "button"]:
                if not any(
                    attr in input_field.attrs for attr in ["pattern", "maxlength"]
                ):
                    vulnerabilities.append(
                        {
                            "type": "XSS",
                            "severity": "High",
                            "description": "Input field without proper validation",
                            "location": str(input_field)[:200],
                        }
                    )

        # Check for sensitive information in HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            sensitive_patterns = ["password", "secret", "key", "token", "api"]
            if any(pattern in comment.lower() for pattern in sensitive_patterns):
                vulnerabilities.append(
                    {
                        "type": "Information Disclosure",
                        "severity": "Medium",
                        "description": "Potentially sensitive information in HTML comments",
                        "location": str(comment)[:200],
                    }
                )

        # Security Headers Analysis
        security_headers = {
            "content-security-policy": {
                "present": False,
                "description": "Missing Content Security Policy (CSP)",
                "severity": "High",
            },
            "x-frame-options": {
                "present": False,
                "description": "Missing X-Frame-Options header to prevent clickjacking",
                "severity": "Medium",
            },
            "x-xss-protection": {
                "present": False,
                "description": "Missing X-XSS-Protection header",
                "severity": "Medium",
            },
            "strict-transport-security": {
                "present": False,
                "description": "Missing Strict-Transport-Security (HSTS) header",
                "severity": "High",
            },
            "referrer-policy": {
                "present": False,
                "description": "Missing Referrer-Policy header",
                "severity": "Low",
            },
            "feature-policy": {
                "present": False,
                "description": "Missing Feature-Policy header",
                "severity": "Low",
            },
        }

        for header, info in security_headers.items():
            if header in headers:
                info["present"] = True
                # Additional checks can be implemented here (e.g., CSP strength)
            else:
                vulnerabilities.append(
                    {
                        "type": "Security Header Missing",
                        "severity": info["severity"],
                        "description": info["description"],
                        "location": "HTTP Response Headers",
                    }
                )

        # Check for mixed content (HTTPS pages loading HTTP resources)
        parsed_url = urlparse(url)
        if parsed_url.scheme == "https":
            mixed_content = False
            # Check for src attributes with http
            for tag in soup.find_all(src=True):
                src = tag["src"]
                if src.startswith("http://"):
                    mixed_content = True
                    vulnerabilities.append(
                        {
                            "type": "Mixed Content",
                            "severity": "High",
                            "description": f"Insecure resource loaded over HTTP: {src}",
                            "location": f"<{tag.name} src>",
                        }
                    )
            # Check for href attributes with http for resources like stylesheets
            for tag in soup.find_all(href=True):
                href = tag["href"]
                if href.startswith("http://"):
                    mixed_content = True
                    vulnerabilities.append(
                        {
                            "type": "Mixed Content",
                            "severity": "High",
                            "description": f"Insecure resource loaded over HTTP: {href}",
                            "location": f"<{tag.name} href>",
                        }
                    )

        # Check for insecure form actions (forms submitting over HTTP)
        if parsed_url.scheme == "https":
            for form in forms:
                action = form.get("action")
                if action:
                    form_action_url = urljoin(url, action)
                    form_parsed = urlparse(form_action_url)
                    if form_parsed.scheme != "https":
                        vulnerabilities.append(
                            {
                                "type": "Insecure Form Submission",
                                "severity": "High",
                                "description": f"Form submits to an insecure URL: {form_action_url}",
                                "location": "<form action>",
                            }
                        )

        # Check for inline JavaScript or CSS that could be exploited for XSS
        # Detecting inline event handlers and <script> tags without src
        inline_scripts = soup.find_all("script", src=False)
        if inline_scripts:
            for script in inline_scripts:
                if script.string and script.string.strip():
                    vulnerabilities.append(
                        {
                            "type": "Inline JavaScript",
                            "severity": "Medium",
                            "description": "Inline JavaScript detected, which can be exploited for XSS",
                            "location": str(script)[:200],
                        }
                    )

        inline_styles = soup.find_all("style")
        if inline_styles:
            for style in inline_styles:
                if style.string and style.string.strip():
                    vulnerabilities.append(
                        {
                            "type": "Inline CSS",
                            "severity": "Low",
                            "description": "Inline CSS detected, which can be exploited for certain attacks",
                            "location": str(style)[:200],
                        }
                    )

        # Check for deprecated or risky HTML elements/attributes
        deprecated_elements = ["marquee", "blink", "font"]
        for elem in deprecated_elements:
            found = soup.find_all(elem)
            for tag in found:
                vulnerabilities.append(
                    {
                        "type": "Deprecated HTML Element",
                        "severity": "Low",
                        "description": f"Deprecated HTML element <{elem}> used",
                        "location": str(tag)[:200],
                    }
                )

        # Check for outdated libraries with known vulnerabilities
        # This is a basic check; for comprehensive analysis, integrate with a vulnerability database
        script_tags = soup.find_all("script", src=True)
        library_patterns = {
            "jquery": r"jquery[-\.]([\d]+)\.([\d]+)\.([\d]+)",
            "bootstrap": r"bootstrap[-\.]([\d]+)\.([\d]+)\.([\d]+)",
            "angular": r"angular[-\.]([\d]+)\.([\d]+)\.([\d]+)",
            "react": r"react[-\.]([\d]+)\.([\d]+)\.([\d]+)",
            "vue": r"vue[-\.]([\d]+)\.([\d]+)\.([\d]+)",
        }
        # Example of known vulnerable versions (for demonstration purposes)
        vulnerable_versions = {
            "jquery": lambda v: v[0] < 3,
            "bootstrap": lambda v: v[0] < 5,
            "angular": lambda v: v[0] < 1,
            "react": lambda v: v[0] < 17,
            "vue": lambda v: v[0] < 3,
        }

        for script in script_tags:
            src = script["src"]
            for lib, pattern in library_patterns.items():
                match = re.search(pattern, src, re.IGNORECASE)
                if match:
                    version = tuple(map(int, match.groups()))
                    if lib in vulnerable_versions and vulnerable_versions[lib](version):
                        vulnerabilities.append(
                            {
                                "type": "Outdated Library",
                                "severity": "High",
                                "description": f"Library {lib} version {'.'.join(map(str, version))} is outdated and may have known vulnerabilities",
                                "location": src,
                            }
                        )

        ####

        # Enhanced Form Analysis

        forms = soup.find_all("form")
        for form in forms:
            # CSRF Protection Check
            if not form.find("input", {"name": ["csrf_token", "_token", "_csrf"]}):
                vulnerabilities.append(
                    {
                        "type": "CSRF",
                        "severity": "High",
                        "description": "Form found without CSRF protection",
                        "location": str(form)[:200],
                        "recommendation": "Implement CSRF tokens for all forms",
                    }
                )

            # Form Method Analysis
            method = form.get("method", "").lower()
            if method == "get" and form.find("input", {"type": ["password", "file"]}):
                vulnerabilities.append(
                    {
                        "type": "Insecure Form Method",
                        "severity": "High",
                        "description": "Sensitive form using GET method",
                        "location": str(form)[:200],
                        "recommendation": "Use POST method for forms handling sensitive data",
                    }
                )

            # Autocomplete Analysis
            password_fields = form.find_all("input", {"type": "password"})
            for field in password_fields:
                if not field.get("autocomplete") == "off":
                    vulnerabilities.append(
                        {
                            "type": "Password Field Security",
                            "severity": "Medium",
                            "description": "Password field without autocomplete=off",
                            "location": str(field)[:200],
                            "recommendation": "Add autocomplete='off' to password fields",
                        }
                    )

            # File Upload Analysis
            file_inputs = form.find_all("input", {"type": "file"})
            for file_input in file_inputs:
                if not file_input.get("accept"):
                    vulnerabilities.append(
                        {
                            "type": "Unrestricted File Upload",
                            "severity": "High",
                            "description": "File upload without file type restrictions",
                            "location": str(file_input)[:200],
                            "recommendation": "Implement file type restrictions using accept attribute",
                        }
                    )

        # Enhanced Input Field Analysis
        inputs = soup.find_all(["input", "textarea"])
        for input_field in inputs:
            input_type = input_field.get("type", "text")

            # Input Validation Analysis
            if input_type not in ["hidden", "submit", "button", "checkbox", "radio"]:
                validation_issues = []

                # Check for length restrictions
                if not input_field.get("maxlength"):
                    validation_issues.append("no maximum length restriction")

                # Check for pattern validation
                if not input_field.get("pattern"):
                    validation_issues.append("no pattern validation")

                # Check for required fields without validation
                if input_field.get("required") and not validation_issues:
                    validation_issues.append("required field without proper validation")

                if validation_issues:
                    vulnerabilities.append(
                        {
                            "type": "Input Validation",
                            "severity": "Medium",
                            "description": f"Input field with insufficient validation: {', '.join(validation_issues)}",
                            "location": str(input_field)[:200],
                            "recommendation": "Implement proper input validation constraints",
                        }
                    )

            # Sensitive Data Field Analysis
            sensitive_names = ["password", "token", "key", "secret", "ssn", "credit"]
            field_name = input_field.get("name", "").lower()

            if any(sens in field_name for sens in sensitive_names):
                if not input_field.get("autocomplete") == "off":
                    vulnerabilities.append(
                        {
                            "type": "Sensitive Data Exposure",
                            "severity": "High",
                            "description": f"Sensitive field '{field_name}' without autocomplete protection",
                            "location": str(input_field)[:200],
                            "recommendation": "Add autocomplete='off' to sensitive fields",
                        }
                    )

        # JavaScript Event Handler Analysis
        elements_with_events = soup.find_all(
            lambda tag: any(attr.startswith("on") for attr in tag.attrs)
        )
        for element in elements_with_events:
            event_handlers = [attr for attr in element.attrs if attr.startswith("on")]
            for handler in event_handlers:
                vulnerabilities.append(
                    {
                        "type": "Inline Event Handler",
                        "severity": "Medium",
                        "description": f"Inline JavaScript event handler ({handler}) detected",
                        "location": str(element)[:200],
                        "recommendation": "Move event handlers to external JavaScript files",
                    }
                )

        # Enhanced Content Security Analysis
        meta_tags = soup.find_all("meta")
        csp_meta = soup.find("meta", {"http-equiv": "Content-Security-Policy"})
        if not csp_meta and "content-security-policy" not in headers:
            vulnerabilities.append(
                {
                    "type": "Missing CSP",
                    "severity": "High",
                    "description": "No Content Security Policy found in meta tags or headers",
                    "location": "Document Head",
                    "recommendation": "Implement a strong Content Security Policy",
                }
            )

        # Iframe Security Analysis
        iframes = soup.find_all("iframe")
        for iframe in iframes:
            if not iframe.get("sandbox"):
                vulnerabilities.append(
                    {
                        "type": "Iframe Security",
                        "severity": "Medium",
                        "description": "Iframe without sandbox attribute",
                        "location": str(iframe)[:200],
                        "recommendation": "Add appropriate sandbox attributes to iframes",
                    }
                )

        # API Endpoint Detection
        api_patterns = [
            r"/api/v?\d*",
            r"/rest/v?\d*",
            r"/graphql",
            r"/swagger",
            r"/openapi",
        ]

        for pattern in api_patterns:
            api_links = soup.find_all(href=re.compile(pattern))
            for link in api_links:
                vulnerabilities.append(
                    {
                        "type": "Exposed API Endpoint",
                        "severity": "Medium",
                        "description": f"Potentially exposed API endpoint detected: {link.get('href')}",
                        "location": str(link)[:200],
                        "recommendation": "Ensure API endpoints are properly secured and not exposed in production",
                    }
                )

        # Session Management Analysis
        session_cookies = [
            cookie
            for cookie in headers.get("set-cookie", "").split(",")
            if any(name in cookie.lower() for name in ["session", "token", "auth"])
        ]

        for cookie in session_cookies:
            if "secure" not in cookie.lower() or "httponly" not in cookie.lower():
                vulnerabilities.append(
                    {
                        "type": "Insecure Session Cookie",
                        "severity": "High",
                        "description": "Session cookie without Secure and/or HttpOnly flags",
                        "location": "HTTP Headers",
                        "recommendation": "Set Secure and HttpOnly flags for all session cookies",
                    }
                )

        # Extract all paths
        paths = self.extract_paths(soup, url)

        return vulnerabilities, paths

    def extract_paths(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all paths from HTML content"""
        paths = set()

        # Extract href attributes
        for link in soup.find_all(["a", "link"]):
            href = link.get("href")
            if href:
                full_url = urljoin(base_url, href)
                paths.add(full_url)

        # Extract src attributes
        for element in soup.find_all(
            ["img", "script", "iframe", "embed", "video", "audio"]
        ):
            src = element.get("src")
            if src:
                full_url = urljoin(base_url, src)
                paths.add(full_url)

        # Extract form actions
        for form in soup.find_all("form"):
            action = form.get("action")
            if action:
                full_url = urljoin(base_url, action)
                paths.add(full_url)

        return list(paths)

    async def process_single_target(self, target: str) -> HTMLScanResult:
        """Process a single target:port combination"""
        start_time = datetime.now(timezone.utc)

        def format_url(target: str) -> str:
            """Format strings like `example.com` to `https://example.com` or `example.com:80` to `http://example.com:80` as urls"""
            # check if target has a port then add the protocol based on the port
            # use python 3 built in url functions to parse the url
            parsed = urlparse(target)

            # If no scheme specified, check for port to determine protocol
            if not parsed.scheme:
                # Split potential host:port
                parts = parsed.path.split(":")
                if len(parts) > 1:
                    # Has port specified
                    try:
                        port = int(parts[1])
                        if port == 443:
                            scheme = "https"
                        else:
                            scheme = "http"
                        host = parts[0]
                        return f"{scheme}://{host}:{port}"
                    except ValueError:
                        # Invalid port, use default https
                        return f"https://{target}"
                else:
                    # No port specified, use https by default
                    return f"https://{target}"

            # Already has scheme, return as-is
            return target

        target = format_url(target)
        try:
            # Fetch HTML content and headers
            html_content, headers = await self.fetch_html(target)

            # Analyze vulnerabilities and extract paths
            vulnerabilities, paths = self.analyze_vulnerabilities(
                html_content, headers, target
            )

            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()

            analysis_result = HTMLAnalysisResult(
                raw_html=html_content,
                potential_vulnerabilities=vulnerabilities,
                paths=paths,
                success=True,
                error=None,
                timestamp=end_time,
                metrics={
                    "processing_time_seconds": processing_time,
                    "html_size_bytes": len(html_content),
                    "vulnerability_count": len(vulnerabilities),
                    "paths_discovered": len(paths),
                },
            )

            return HTMLScanResult(target=target, analysis=analysis_result)

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            logger.error(f"Error processing {target} {str(e)}")

            return HTMLScanResult(
                target=target,
                analysis=HTMLAnalysisResult(
                    raw_html="",
                    potential_vulnerabilities=[],
                    paths=[],
                    success=False,
                    error=str(e),
                    timestamp=end_time,
                    metrics={
                        "processing_time_seconds": (
                            end_time - start_time
                        ).total_seconds(),
                        "error_type": type(e).__name__,
                    },
                ),
            )

    async def process_targets(self, targets: List[str]) -> List[HTMLScanResult]:
        """Process multiple targets concurrently"""
        tasks = []
        for target_info in targets:
            task = self.process_single_target(target_info)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def process_task(self, task: Dict) -> Dict:
        """Process a task with targets"""
        if not isinstance(task, dict) or "_id" not in task:
            self.task_logger.error("Invalid task format")
            return self._create_task_result(
                task.get("_id"), error="Invalid task format"
            )

        try:
            # Extract targets from task considering that targets need to be in the currently executing step
            current_step = next(
                (
                    step
                    for step in task["steps"]
                    if step["state"] == TaskState.IN_PROGRESS.value
                ),
                None,
            )
            if not current_step or not current_step.get("targets"):
                logger.error("No targets found in current step")

                return self._create_task_result(
                    task.get("_id"), error="No targets available"
                )

            targets = current_step["targets"]
            logger.info(f"Processing targets: {targets}")

            # Process all targets
            results = asyncio.run(self.process_targets(targets))
            # Create aggregated result
            aggregated_result = self._create_task_result(
                task.get("_id"),
                result="\n".join(
                    [
                        serialize_task_to_string(result.model_dump())
                        for result in results
                    ]
                ),
                metadata={
                    "timestamp": get_iso_timestamp(),
                    "total_targets": len(targets),
                    "successful_scans": sum(1 for r in results if r.analysis.success),
                    "failed_scans": sum(1 for r in results if not r.analysis.success),
                },
            )

            return aggregated_result

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(task.get("_id"), error=str(e))
