import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import aiohttp
import asyncio
from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel, Field
from urllib.parse import urljoin, urlparse
from boss.utils import get_iso_timestamp, serialize_task_to_string
from boss.models import TaskState
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class HTMLAnalysisResult(BaseModel):
    """Model for HTML analysis results"""
    raw_html: str
    potential_vulnerabilities: List[Dict]
    paths: List[str]
    success: bool
    error: Optional[str]
    timestamp: datetime
    metrics: Dict = Field(default_factory=dict)


class HTMLScanResult(BaseModel):
    """Model for overall scan results"""
    target: str
    analysis: HTMLAnalysisResult


class WrapperHTML(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_html_analyzer",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.agent_id = agent_id

    async def fetch_html(self, target: str) -> str:
        """Fetch HTML content from target"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{target}"
                async with session.get(url, timeout=10) as response:
                    return await response.text()
        except Exception as e:
            logger.error(f"Error fetching HTML from {target} {str(e)}")
            raise

    def analyze_vulnerabilities(self, html_content: str, url: str) -> List[Dict]:
        """Analyze HTML content for potential vulnerabilities"""
        vulnerabilities = []
        soup = BeautifulSoup(html_content, 'html.parser')

        # Check for forms without CSRF protection
        forms = soup.find_all('form')
        for form in forms:
            if not form.find('input', {'name': ['csrf_token', '_token', '_csrf']}):
                vulnerabilities.append({
                    'type': 'CSRF',
                    'severity': 'Medium',
                    'description': 'Form found without CSRF protection',
                    'location': str(form)[:200]
                })

        # Check for input fields with potential XSS vulnerabilities
        inputs = soup.find_all(['input', 'textarea'])
        for input_field in inputs:
            if input_field.get('type') not in ['hidden', 'submit', 'button']:
                if not any(attr in input_field.attrs for attr in ['pattern', 'maxlength']):
                    vulnerabilities.append({
                        'type': 'XSS',
                        'severity': 'High',
                        'description': 'Input field without proper validation',
                        'location': str(input_field)[:200]
                    })

        # Check for sensitive information in HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            sensitive_patterns = ['password', 'secret', 'key', 'token', 'api']
            if any(pattern in comment.lower() for pattern in sensitive_patterns):
                vulnerabilities.append({
                    'type': 'Information Disclosure',
                    'severity': 'Medium',
                    'description': 'Potentially sensitive information in HTML comments',
                    'location': str(comment)[:200]
                })

        # Extract all paths
        paths = self.extract_paths(soup, url)

        return vulnerabilities, paths

    def extract_paths(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all paths from HTML content"""
        paths = set()
        
        # Extract href attributes
        for link in soup.find_all(['a', 'link']):
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                paths.add(full_url)

        # Extract src attributes
        for element in soup.find_all(['img', 'script', 'iframe']):
            src = element.get('src')
            if src:
                full_url = urljoin(base_url, src)
                paths.add(full_url)

        # Extract form actions
        for form in soup.find_all('form'):
            action = form.get('action')
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
                parts = parsed.path.split(':')
                if len(parts) > 1:
                    # Has port specified
                    try:
                        port = int(parts[1])
                        if port == 443:
                            scheme = 'https'
                        else:
                            scheme = 'http'
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
            # Fetch HTML content
            html_content = await self.fetch_html(target)
            
            # Analyze vulnerabilities and extract paths
            vulnerabilities, paths = self.analyze_vulnerabilities(html_content, target)

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
                    'processing_time_seconds': processing_time,
                    'html_size_bytes': len(html_content),
                    'vulnerability_count': len(vulnerabilities),
                    'paths_discovered': len(paths)
                }
            )

            return HTMLScanResult(
                target=target,
                analysis=analysis_result
            )

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
                        'processing_time_seconds': (end_time - start_time).total_seconds(),
                        'error_type': type(e).__name__
                    }
                )
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
            return {"success": False, "note": "Invalid task format"}
     
        try:
            # Extract targets from task considering that targets need to be in the currently executing step
            current_step = next((step for step in task["steps"] if step["state"] == TaskState.IN_PROGRESS.value), None)
            if not current_step or not current_step.get("targets"):
                logger.error("No targets found in current step")
                return {"success": True, "note": "No targets found in any steps"}

            targets = current_step["targets"]
            logger.info(f"Processing targets: {targets}")
            
            # Process all targets
            results = asyncio.run(self.process_targets(targets))
            # Create aggregated result
            aggregated_result = {
                "task_id": str(task["_id"]),
                "agent_id": self.agent_id,
                "success": True,
                "result": serialize_task_to_string([result.model_dump(exclude_unset=True, exclude_none=True, exclude_defaults=True, serialize_as_any=True) for result in results]),
                "metadata": {
                    "timestamp": get_iso_timestamp(),
                    "total_targets": len(targets),
                    "successful_scans": sum(1 for r in results if r.analysis.success),
                    "failed_scans": sum(1 for r in results if not r.analysis.success)
                }
            }

            return aggregated_result

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return {"success": False, "error": str(e)}