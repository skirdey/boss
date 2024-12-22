import os
import re

os.environ["PYTHONIOENCODING"] = "utf-8"

import asyncio
import logging
import os
import shlex
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from boss.utils import serialize_task_to_string
from boss.wrappers.wrapper_agent import WrapperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.propagate = False


class WapitiCliCommand(BaseModel):
    """Model for Wapiti CLI command parameters"""

    # Basic options
    url: str = Field(description="Target URL for the Wapiti scan")
    swagger: Optional[str] = Field(description="Swagger URI for API scanning")
    data: Optional[str] = Field(description="Data to send with requests")

    # Scope and modules
    scope: Optional[str] = Field(
        description="Scope of the scan (url, page, folder, subdomain, domain, punk)"
    )
    modules: Optional[List[str]] = Field(
        description="List of specific Wapiti modules to enable"
    )
    list_modules: Optional[bool] = Field(description="List available modules and exit")

    # Logging and output
    log_level: Optional[str] = Field(
        alias="l",
        description="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    )
    log_output: Optional[str] = Field(
        alias="log", description="Path to save the log file"
    )
    output_format: Optional[str] = Field(
        alias="f", description="Output format for the report (e.g., json, html, xml)"
    )
    output_path: Optional[str] = Field(
        alias="o", description="Path to save the output report"
    )
    detailed_report_level: Optional[int] = Field(
        alias="dr", description="Level of detail for the report (0-3)"
    )
    no_bugreport: Optional[bool] = Field(
        description="Do not include bug report information in the output"
    )
    store_session: Optional[str] = Field(description="Path to store the session data")
    store_config: Optional[str] = Field(
        description="Path to store the configuration data"
    )

    # Proxy and network options
    proxy_url: Optional[str] = Field(
        alias="p", description="Proxy URL to route requests through"
    )
    tor: Optional[bool] = Field(description="Use Tor for anonymizing requests")
    mitm_port: Optional[int] = Field(description="Port for the Man-in-the-Middle proxy")
    wait: Optional[int] = Field(
        description="Time to wait between requests (in seconds)"
    )
    max_scan_time: Optional[int] = Field(
        description="Maximum time to spend on the scan (in seconds)"
    )
    max_attack_time: Optional[int] = Field(
        description="Maximum time to spend on attacks (in seconds)"
    )

    # Authentication options
    auth_user: Optional[str] = Field(description="Username for HTTP authentication")
    auth_password: Optional[str] = Field(description="Password for HTTP authentication")
    auth_method: Optional[str] = Field(
        description="Authentication method (basic, digest, ntlm)"
    )
    form_user: Optional[str] = Field(
        description="Username for form-based authentication"
    )
    form_password: Optional[str] = Field(
        description="Password for form-based authentication"
    )
    form_url: Optional[str] = Field(description="URL for form-based authentication")
    form_data: Optional[str] = Field(description="Data for form-based authentication")
    form_enctype: Optional[str] = Field(description="Encoding type for form data")
    form_script: Optional[str] = Field(
        description="Script for form-based authentication"
    )
    form_credentials: Optional[str] = Field(
        description="Credentials for form-based authentication"
    )

    # Cookie options
    cookie_file: Optional[str] = Field(alias="c", description="Path to the cookie file")
    cookie_value: Optional[str] = Field(
        alias="C", description="Value of the cookie to use"
    )
    drop_set_cookie: Optional[bool] = Field(
        description="Drop 'Set-Cookie' headers from responses"
    )

    # Crawling options
    skip_crawl: Optional[bool] = Field(description="Skip the crawling phase")
    resume_crawl: Optional[bool] = Field(
        description="Resume a previously interrupted crawl"
    )
    flush_attacks: Optional[bool] = Field(description="Flush attack data after scan")
    flush_session: Optional[bool] = Field(description="Flush session data after scan")

    # Parameter and link options
    max_depth: Optional[int] = Field(description="Maximum recursion depth")
    max_links_per_page: Optional[int] = Field(
        description="Maximum number of links per page"
    )
    max_files_per_dir: Optional[int] = Field(
        description="Maximum number of files per directory"
    )
    max_parameters: Optional[int] = Field(
        description="Maximum number of parameters to scan"
    )
    max_scan_time_seconds: Optional[int] = Field(
        description="Maximum scan time in seconds", alias="max-scan-time"
    )
    max_attack_time_seconds: Optional[int] = Field(
        description="Maximum attack time in seconds", alias="max-attack-time"
    )

    # Force and verification options
    force: Optional[bool] = Field(
        alias="S", description="Force the scan even if potential issues are detected"
    )
    verify_ssl: Optional[bool] = Field(
        alias="verify-ssl", description="Verify SSL certificates (0: no, 1: yes)"
    )

    # Endpoint options
    external_endpoint_url: Optional[str] = Field(
        description="URL for the external endpoint"
    )
    internal_endpoint_url: Optional[str] = Field(
        description="URL for the internal endpoint"
    )
    endpoint_url: Optional[str] = Field(description="URL for the endpoint")
    dns_endpoint_domain: Optional[str] = Field(
        description="Domain for the DNS endpoint"
    )

    # Header and agent options
    header: Optional[str] = Field(
        alias="H", description="Custom header to include in requests"
    )
    user_agent: Optional[str] = Field(
        alias="A", description="User-Agent string to use for requests"
    )

    # Color and verbosity options
    color: Optional[bool] = Field(description="Enable colored output")
    verbose: Optional[str] = Field(
        alias="v", description="Set verbosity level: 0, 1 or 2"
    )

    # CMS options
    cms_list: Optional[str] = Field(
        alias="cms", description="List of CMS to identify (comma-separated)"
    )
    wapp_url: Optional[str] = Field(
        alias="wapp-url", description="URL for Wappalyzer to identify technologies"
    )
    wapp_dir: Optional[str] = Field(
        alias="wapp-dir", description="Directory for Wappalyzer data"
    )

    # Miscellaneous options
    tasks: Optional[str] = Field(description="Tasks to execute (comma-separated)")
    external_endpoint_url: Optional[str] = Field(
        description="URL for the external endpoint"
    )
    internal_endpoint_url: Optional[str] = Field(
        description="URL for the internal endpoint"
    )
    dns_endpoint_domain: Optional[str] = Field(
        description="Domain for the DNS endpoint"
    )

    # Additional options can be added here as needed


class WrapperWapitiCliAgent(WrapperAgent):
    def __init__(
        self,
        agent_id="agent_wapiti_cli_scanner",
        kafka_bootstrap_servers="localhost:9092",
    ):
        super().__init__(agent_id, kafka_bootstrap_servers)
        self.setup_task_logger()

    def setup_task_logger(self):
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

    async def _call_openai_api(self, prompt: str) -> WapitiCliCommand:
        """Call OpenAI API with structured output parsing for Wapiti CLI parameters"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            system_prompt = """
            Extract Wapiti CLI parameters from the task description. Your goal is to construct a valid Wapiti command. Consider:
            1. Identify the target URL.
            2. Determine the appropriate scan scope (url, page, folder, subdomain, domain, punk).
            3. Identify specific modules to enable (e.g., sql, xss).
            4. Determine if crawling should be skipped.
            5. Extract authentication details (username, password, method) if provided.
            6. Determine maximum recursion depth and timeout if specified.
            7. DO NOT include the output format in the command.
            Map these to the corresponding Wapiti CLI options.
            ---
            Wapiti 3.2.0 (wapiti-scanner.github.io)
            usage: wapiti [-h] [-u URL] [--swagger URI] [--data data]
                        [--scope {url,page,folder,subdomain,domain,punk}] [-m MODULES_LIST]
                        [--list-modules] [-l LEVEL] [-p PROXY_URL] [--tor] [--mitm-port PORT]
                        [--headless {no,hidden,visible}] [--wait TIME] [-a CREDENTIALS]
                        [--auth-user USERNAME] [--auth-password PASSWORD]
                        [--auth-method {basic,digest,ntlm}] [--form-cred CREDENTIALS]
                        [--form-user USERNAME] [--form-password PASSWORD] [--form-url URL]
                        [--form-data DATA] [--form-enctype DATA] [--form-script FILENAME]
                        [-c COOKIE_FILE] [-sf SIDE_FILE] [-C COOKIE_VALUE] [--drop-set-cookie]
                        [--skip-crawl] [--resume-crawl] [--flush-attacks] [--flush-session]
                        [--store-session PATH] [--store-config PATH] [-s URL] [-x URL]
                        [-r PARAMETER] [--skip PARAMETER] [-d DEPTH] [--max-links-per-page MAX]
                        [--max-files-per-dir MAX] [--max-scan-time SECONDS]
                        [--max-attack-time SECONDS] [--max-parameters MAX] [-S FORCE]
                        [--tasks tasks] [--external-endpoint EXTERNAL_ENDPOINT_URL]
                        [--internal-endpoint INTERNAL_ENDPOINT_URL] [--endpoint ENDPOINT_URL]
                        [--dns-endpoint DNS_ENDPOINT_DOMAIN] [-t SECONDS] [-H HEADER]
                        [-A AGENT] [--verify-ssl {0,1}] [--color] [-v LEVEL]
                        [--log OUTPUT_PATH] [-f FORMAT] [-o OUTPUT_PATH]
                        [-dr DETAILED_REPORT_LEVEL] [--no-bugreport] [--update] [--version]
                        [--cms CMS_LIST] [--wapp-url WAPP_URL] [--wapp-dir WAPP_DIR]
            Shortest way (with default options) to launch a Wapiti scan :
            wapiti -u http://target/
            ---
            """

            completion = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=WapitiCliCommand,
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

    async def execute_wapiti_scan(self, command: WapitiCliCommand) -> Dict[str, Any]:
        """Execute Wapiti scan using the command line interface and capture output"""
        try:
            self.task_logger.info(f"Executing Wapiti scan on: {command.url}")
            start_time = datetime.now(timezone.utc)

            wapiti_command = ["wapiti", "-u", command.url]

            # Mapping model fields to CLI arguments
            field_mappings = {
                "swagger": ("--swagger", command.swagger),
                "data": ("--data", command.data),
                "scope": ("--scope", command.scope),
                "modules": (
                    "-m",
                    ",".join(command.modules) if command.modules else None,
                ),
                "list_modules": ("--list-modules", command.list_modules),
                "log_level": ("-l", command.log_level),
                "log_output": ("--log", command.log_output),
                "output_format": ("-f", "json"),
                "output_path": ("-o", command.output_path),
                "detailed_report_level": (
                    "-dr",
                    str(command.detailed_report_level)
                    if command.detailed_report_level
                    else None,
                ),
                "no_bugreport": ("--no-bugreport", command.no_bugreport),
                "store_session": ("--store-session", command.store_session),
                "store_config": ("--store-config", command.store_config),
                "proxy_url": ("-p", command.proxy_url),
                "tor": ("--tor", command.tor),
                "mitm_port": (
                    "--mitm-port",
                    str(command.mitm_port) if command.mitm_port else None,
                ),
                "wait": ("--wait", str(command.wait) if command.wait else None),
                "max_scan_time_seconds": (
                    "--max-scan-time",
                    str(command.max_scan_time_seconds)
                    if command.max_scan_time_seconds
                    else None,
                ),
                "max_attack_time_seconds": (
                    "--max-attack-time",
                    str(command.max_attack_time_seconds)
                    if command.max_attack_time_seconds
                    else None,
                ),
                "auth_user": ("--auth-user", command.auth_user),
                "auth_password": ("--auth-password", command.auth_password),
                "auth_method": ("--auth-method", command.auth_method),
                "form_user": ("--form-user", command.form_user),
                "form_password": ("--form-password", command.form_password),
                "form_url": ("--form-url", command.form_url),
                "form_data": ("--form-data", command.form_data),
                "form_enctype": ("--form-enctype", command.form_enctype),
                "form_script": ("--form-script", command.form_script),
                "form_credentials": ("--form-cred", command.form_credentials),
                "cookie_file": ("-c", command.cookie_file),
                "cookie_value": ("-C", command.cookie_value),
                "drop_set_cookie": ("--drop-set-cookie", command.drop_set_cookie),
                "skip_crawl": ("--skip-crawl", command.skip_crawl),
                "resume_crawl": ("--resume-crawl", command.resume_crawl),
                "flush_attacks": ("--flush-attacks", command.flush_attacks),
                "flush_session": ("--flush-session", command.flush_session),
                "max_depth": (
                    "-d",
                    str(command.max_depth) if command.max_depth else None,
                ),
                "max_links_per_page": (
                    "--max-links-per-page",
                    str(command.max_links_per_page)
                    if command.max_links_per_page
                    else None,
                ),
                "max_files_per_dir": (
                    "--max-files-per-dir",
                    str(command.max_files_per_dir)
                    if command.max_files_per_dir
                    else None,
                ),
                "max_parameters": (
                    "--max-parameters",
                    str(command.max_parameters) if command.max_parameters else None,
                ),
                "force": ("-S", command.force),
                "verify_ssl": (
                    "--verify-ssl",
                    "1"
                    if command.verify_ssl
                    else "0"
                    if command.verify_ssl is not None
                    else None,
                ),
                "external_endpoint_url": (
                    "--external-endpoint",
                    command.external_endpoint_url,
                ),
                "internal_endpoint_url": (
                    "--internal-endpoint",
                    command.internal_endpoint_url,
                ),
                "endpoint_url": ("--endpoint", command.endpoint_url),
                "dns_endpoint_domain": (
                    "--dns-endpoint",
                    command.dns_endpoint_domain,
                ),
                "header": ("-H", command.header),
                "user_agent": ("-A", command.user_agent),
                "color": ("--color", command.color),
                "verbose": ("-v", "0"),
                "cms_list": ("--cms", command.cms_list),
                "wapp_url": ("--wapp-url", command.wapp_url),
                "wapp_dir": ("--wapp-dir", command.wapp_dir),
                "tasks": ("--tasks", command.tasks),
            }

            # Dynamically add arguments based on field mappings
            for field, (cli_arg, value) in field_mappings.items():
                if isinstance(value, bool):
                    if value:
                        wapiti_command.append(cli_arg)
                elif value is not None:
                    wapiti_command.extend([cli_arg, value])

            self.task_logger.debug(
                f"Constructed Wapiti command: {' '.join(wapiti_command)}"
            )

            # Quote arguments with spaces
            quoted_command = [shlex.quote(arg) for arg in wapiti_command]
            command_string = " ".join(quoted_command)
            self.task_logger.info(f"Executing command: {command_string}")

            process = await asyncio.create_subprocess_shell(
                command_string,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            if process.returncode == 0:
                self.task_logger.info(
                    f"Wapiti scan completed successfully in {execution_time:.2f} seconds."
                )

                report_text = stdout.decode(encoding="utf-8", errors="replace")
                self.task_logger.info(f"Report text: {report_text}")

                # Extract report path using regex
                report_path_match = re.search(
                    r"A report has been generated in the file (.+\.json)", report_text
                )
                if report_path_match:
                    report_path = report_path_match.group(1)
                    if os.path.exists(report_path):
                        with open(report_path, "r", encoding="utf-8") as report_file:
                            report_content = report_file.read()
                    else:
                        report_content = "Report file not found."

                else:
                    report_content = "Report path not found in Wapiti output."

                return {
                    "success": True,
                    "execution_time": execution_time,
                    "report": report_content,
                }
            else:
                self.task_logger.error(
                    f"Wapiti scan failed with return code: {process.returncode}"
                )
                return {
                    "success": False,
                    "error": f"Wapiti scan failed with return code: {process.returncode}, stderr: {stderr.decode()}",
                    "execution_time": execution_time,
                }

        except FileNotFoundError:
            self.task_logger.error(
                "Wapiti executable not found. Ensure it's in your PATH."
            )
            return {
                "success": False,
                "error": "Wapiti executable not found. Ensure it's in your PATH.",
            }
        except Exception as e:
            self.task_logger.error(f"Error during Wapiti execution: {str(e)}")
            return {
                "success": False,
                "error": f"Error during Wapiti execution: {str(e)}",
            }

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task asynchronously"""
        self.task_logger.info("**************** WAPITI CLI AGENT ****************")

        if not isinstance(task, dict) or "task_id" not in task or "step_id" not in task:
            logger.error("Invalid task format")
            return self._create_task_result(
                task.get("task_id"), task.get("step_id"), error="Invalid task format"
            )

        try:
            task_id = task.get("task_id")
            step_id = task.get("step_id")
            self.task_logger.info(
                f"Processing task with ID: {task_id} and step ID: {step_id}"
            )

            step_description = task.get("description", "")
            task_prompt = f"Run a Wapiti scan with the following requirements:\n{step_description}\nPlease provide the report in JSON format if possible."

            targets = task.get("targets", [])
            if targets:
                task_prompt += f"\nTargets: {targets}\n"

            # Generate scan parameters using LLM
            parsed_command = await self._call_openai_api(task_prompt)

            # Execute the Wapiti scan using the CLI
            scan_result = await self.execute_wapiti_scan(parsed_command)

            # Prepare the result
            result = self._create_task_result(
                task_id=task_id,
                step_id=step_id,
                result=serialize_task_to_string(scan_result),
                metadata={
                    "url": parsed_command.url,
                    "scope": parsed_command.scope,
                    "modules": parsed_command.modules,
                    "auth_type": parsed_command.auth_method
                    if parsed_command.auth_user
                    else None,
                },
                step_description=step_description,
            )

            return result

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return self._create_task_result(
                task_id=task.get("task_id"),
                step_id=task.get("step_id"),
                error=f"Validation error: {ve}",
            )
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return self._create_task_result(
                task_id=task.get("task_id"),
                step_id=task.get("step_id"),
                error=str(e),
            )
