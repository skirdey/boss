#!/usr/bin/env python3

import asyncio
import json
import re
import socket
import ssl
import sys
from datetime import datetime
from typing import Dict, List, Optional

import aiodns
import aiohttp

if sys.platform.startswith("win"):
    # Force asyncio to use SelectorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ------------------------------------------------------------------------
# Example: If you cannot or prefer not to install aiodns, you can do DNS
# checks via the standard library in a thread pool:
#
# import functools
# import dns.resolver
#
# async def dns_lookup_in_threadpool(domain, record_type, loop):
#     def sync_lookup():
#         return dns.resolver.resolve(domain, record_type)
#     return await loop.run_in_executor(None, sync_lookup)
#
# Then call `await dns_lookup_in_threadpool(...)` instead of aiodns below.
# ------------------------------------------------------------------------


class APIScanner:
    def __init__(self, target: str, ports: List[int] = None, max_concurrency: int = 10):
        """Initialize the WESSy scanner with extended capabilities, now async."""
        self.target = self._clean_target(target)
        self.ports = ports or [
            21,
            22,
            23,
            25,
            53,
            80,
            443,
            445,
            3306,
            3389,
            5432,
            8080,
            8443,
        ]
        # Instead of a thread pool size, we talk about concurrency (semaphores) in asyncio
        self.max_concurrency = max_concurrency

        self.results = {
            "target": self.target,
            "scan_time": datetime.now().isoformat(),
            "vulnerabilities": [],
            "open_ports": [],
            "ssl_issues": [],
            "server_config": {},
            "cloud_storage": [],
            "injection_points": [],
            "directories": [],
            "subdomains": [],
            "dns_records": {},
            "cms_detection": {},
            "web_technologies": [],
            "security_headers": {},
            "file_disclosure": [],
            "cors_config": {},
            "csrf_vulnerabilities": [],
            "api_endpoints": [],
        }

        # Common directories and files to check
        self._load_wordlists()

        # Initialize regex patterns for technology detection
        self._init_detection_patterns()

        # Async DNS resolver (requires `aiodns`)
        self.resolver = aiodns.DNSResolver()

        # A semaphore to limit concurrency
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    def _clean_target(self, target: str) -> str:
        """Clean and validate the target URL."""
        if not target.startswith(("http://", "https://")):
            target = "http://" + target
        from urllib.parse import urlparse

        parsed = urlparse(target)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _get_base_domain(self) -> str:
        """Extract the base domain from the target."""
        from urllib.parse import urlparse

        parsed = urlparse(self.target)
        domain = parsed.netloc
        # Simple extraction, can be improved with tldextract
        parts = domain.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return domain

    def _load_wordlists(self):
        """Load wordlists for directory brute forcing and common files."""
        self.common_dirs = [
            "admin",
            "wp-admin",
            "administrator",
            "login",
            "backup",
            "wp-content",
            "uploads",
            "images",
            "img",
            "css",
            "js",
            "test",
            "temp",
            "dev",
            "private",
            "old",
            "doc",
            "docs",
            ".git",
            ".env",
            "config",
            "dashboard",
            "api",
            "v1",
            "v2",
            "php",
            "includes",
            "sql",
        ]

        self.common_files = [
            "robots.txt",
            "sitemap.xml",
            ".git/HEAD",
            ".env",
            "wp-config.php",
            "config.php",
            "info.php",
            "phpinfo.php",
            ".htaccess",
            "web.config",
            "backup.sql",
            "database.sql",
            ".DS_Store",
            "crossdomain.xml",
        ]

        self.subdomain_prefixes = [
            "www",
            "mail",
            "remote",
            "blog",
            "webmail",
            "server",
            "ns1",
            "ns2",
            "smtp",
            "secure",
            "vpn",
            "admin",
            "portal",
            "dev",
            "staging",
            "test",
            "api",
            "m",
            "mobile",
            "app",
        ]

    def _init_detection_patterns(self):
        """Initialize patterns for detecting web technologies and CMS."""
        self.tech_patterns = {
            "wordpress": {
                "pattern": [r"/wp-content/", r"/wp-includes/", r"wp-json"],
                "headers": ["x-powered-by: PHP"],
            },
            "drupal": {
                "pattern": [r"Drupal.settings", r"/sites/default/", r"/sites/all/"],
                "headers": ["x-drupal-cache", "x-generator: Drupal"],
            },
            "joomla": {
                "pattern": [r"/components/", r"/modules/", r"option=com_"],
                "headers": ["x-content-encoded-by: Joomla"],
            },
            "django": {
                "pattern": [r"csrfmiddlewaretoken", r"__admin__"],
                "headers": ["x-frame-options: SAMEORIGIN"],
            },
            "laravel": {
                "pattern": [r"laravel_session", r"XSRF-TOKEN"],
                "headers": ["x-powered-by: PHP"],
            },
        }

    async def enumerate_subdomains(self) -> List[str]:
        """Enumerate subdomains using DNS queries and common prefixes (aiodns)."""
        subdomains = set()
        base_domain = self._get_base_domain()

        async def try_subdomain(prefix: str):
            subdomain = f"{prefix}.{base_domain}"
            try:
                # Acquire the semaphore to limit concurrency
                async with self.semaphore:
                    answers = await self.resolver.query(subdomain, "A")
                    if answers:
                        subdomains.add(subdomain)
                        # Try to get CNAME records
                        try:
                            cname_answers = await self.resolver.query(
                                subdomain, "CNAME"
                            )
                            for rdata in cname_answers:
                                subdomains.add(str(rdata.host).rstrip("."))
                        except:
                            pass
            except:
                pass

        tasks = [
            asyncio.create_task(try_subdomain(prefix))
            for prefix in self.subdomain_prefixes
        ]
        await asyncio.gather(*tasks)
        return list(subdomains)

    async def check_dns_records(self) -> Dict:
        """Check various DNS records for the domain using aiodns."""
        dns_records = {}
        record_types = ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME", "PTR"]

        domain = self._get_base_domain()

        async def query_record(record_type: str):
            try:
                async with self.semaphore:
                    answers = await self.resolver.query(domain, record_type)
                return record_type, [str(rdata) for rdata in answers]
            except:
                return record_type, []

        tasks = [asyncio.create_task(query_record(rt)) for rt in record_types]
        results = await asyncio.gather(*tasks)

        for record_type, rdata_list in results:
            if rdata_list:
                dns_records[record_type] = rdata_list

        return dns_records

    async def _resolve_target(self) -> str:
        """Resolve the target hostname to an IP address."""
        from urllib.parse import urlparse

        parsed = urlparse(self.target)
        hostname = parsed.netloc
        try:
            loop = asyncio.get_running_loop()
            # Use getaddrinfo in an async-friendly manner
            infos = await loop.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
            if infos:
                # getaddrinfo returns a list of (family, type, proto, canonname, sockaddr)
                return infos[0][4][0]  # IP address
        except Exception:
            pass
        return ""

    async def scan_ports(self) -> List[int]:
        """Scan the specified ports to find open ones."""
        open_ports = []
        target_ip = await self._resolve_target()
        if not target_ip:
            return open_ports

        async def scan_port(port):
            try:
                # Acquire the semaphore
                async with self.semaphore:
                    conn = asyncio.open_connection(target_ip, port)
                    reader, writer = await asyncio.wait_for(conn, timeout=1)
                    # If we got here, it's open
                    open_ports.append(port)
                    writer.close()
                    await writer.wait_closed()
            except:
                pass

        tasks = [asyncio.create_task(scan_port(port)) for port in self.ports]
        await asyncio.gather(*tasks)
        return open_ports

    async def directory_bruteforce(self) -> List[Dict]:
        """Perform directory and file enumeration using aiohttp."""
        found_paths = []

        paths_to_check = self.common_dirs + self.common_files

        async def check_path(path: str, session: aiohttp.ClientSession):
            url = f"{self.target.rstrip('/')}/{path}"
            try:
                async with self.semaphore:
                    async with session.get(url, timeout=3) as response:
                        if response.status == 200:
                            content = await response.read()
                            found_paths.append(
                                {
                                    "path": path,
                                    "code": response.status,
                                    "size": len(content),
                                    "type": "file" if "." in path else "directory",
                                }
                            )
            except:
                pass

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(check_path(path, session))
                for path in paths_to_check
            ]
            await asyncio.gather(*tasks)

        return found_paths

    async def detect_web_technologies(self) -> List[Dict]:
        """Detect web technologies, frameworks, and libraries."""
        technologies = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.target, timeout=5) as response:
                    content = await response.text(errors="ignore")
                    raw_headers = response.headers
                    # Convert headers to a dict with lowercase keys
                    headers = {k.lower(): v for k, v in raw_headers.items()}

                    # Check for CMS and frameworks
                    for tech, patterns in self.tech_patterns.items():
                        # Check content patterns
                        for pattern in patterns["pattern"]:
                            if re.search(pattern, content, re.I):
                                technologies.append(
                                    {
                                        "name": tech,
                                        "confidence": "high",
                                        "found_by": "pattern",
                                    }
                                )
                                break

                        # Check headers
                        for header_pattern in patterns["headers"]:
                            if ": " in header_pattern:
                                header_name, header_val = header_pattern.split(": ", 1)
                                header_name = header_name.lower()
                                header_val = header_val.lower()
                                if (
                                    header_name in headers
                                    and header_val in headers[header_name].lower()
                                ):
                                    technologies.append(
                                        {
                                            "name": tech,
                                            "confidence": "high",
                                            "found_by": "header",
                                        }
                                    )

                    # Check for JavaScript frameworks
                    js_patterns = {
                        "jQuery": r"jquery[.-]",
                        "React": r"react[.-]",
                        "Vue.js": r"vue[.-]",
                        "Angular": r"angular[.-]",
                        "Bootstrap": r"bootstrap[.-]",
                    }
                    for framework, pattern in js_patterns.items():
                        if re.search(pattern, content, re.I):
                            technologies.append(
                                {
                                    "name": framework,
                                    "confidence": "medium",
                                    "found_by": "pattern",
                                }
                            )
        except:
            pass

        return technologies

    async def check_cors_config(self) -> Dict:
        """Check CORS configuration and potential misconfigurations."""
        cors_results = {
            "enabled": False,
            "allow_all_origins": False,
            "allow_credentials": False,
            "allowed_origins": [],
            "vulnerabilities": [],
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Send request with custom Origin header
                headers = {
                    "Origin": "https://evil.com",
                    "Access-Control-Request-Method": "GET,POST,PUT,DELETE",
                }
                async with session.get(
                    self.target, headers=headers, timeout=5
                ) as response:
                    response_headers = {
                        k.lower(): v for k, v in response.headers.items()
                    }
                    acao = "access-control-allow-origin"
                    acac = "access-control-allow-credentials"

                    if acao in response_headers:
                        cors_results["enabled"] = True
                        origin = response_headers[acao]
                        if origin == "*":
                            cors_results["allow_all_origins"] = True
                            cors_results["vulnerabilities"].append(
                                {
                                    "type": "permissive_cors",
                                    "details": "CORS allows all origins (*)",
                                }
                            )
                        else:
                            cors_results["allowed_origins"].append(origin)

                        if acac in response_headers:
                            cors_results["allow_credentials"] = True
                            if origin == "*":
                                cors_results["vulnerabilities"].append(
                                    {
                                        "type": "credentials_with_wildcard",
                                        "details": "Allowing credentials with wildcard origin",
                                    }
                                )
        except Exception as e:
            cors_results["error"] = str(e)

        return cors_results

    async def check_csrf_protection(self) -> List[Dict]:
        """Check for CSRF vulnerabilities."""
        csrf_issues = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.target, timeout=5) as response:
                    content = await response.text(errors="ignore")
                    raw_headers = response.headers
                    headers = {k.lower(): v for k, v in raw_headers.items()}

                    # Look for forms
                    forms = re.findall(
                        r"<form[^>]*>.*?</form>", content, re.DOTALL | re.I
                    )
                    for form in forms:
                        has_csrf_token = any(
                            x in form.lower()
                            for x in ["csrf", "xsrf", "_token", "authenticity_token"]
                        )
                        if not has_csrf_token:
                            csrf_issues.append(
                                {
                                    "type": "missing_csrf_token",
                                    "details": "Form found without CSRF token",
                                    "evidence": form[:100] + "...",
                                }
                            )

                    # Check security headers related to CSRF
                    if "x-frame-options" not in headers:
                        csrf_issues.append(
                            {
                                "type": "missing_x_frame_options",
                                "details": "X-Frame-Options header not set",
                            }
                        )

                    if "content-security-policy" not in headers:
                        csrf_issues.append(
                            {
                                "type": "missing_csp",
                                "details": "Content-Security-Policy header not set",
                            }
                        )

        except Exception as e:
            csrf_issues.append({"type": "error", "details": str(e)})

        return csrf_issues

    async def check_file_disclosure(self) -> List[Dict]:
        """Check for sensitive file disclosure."""
        sensitive_files = []

        # Additional sensitive files to check
        files_to_check = [
            ".git/config",
            ".svn/entries",
            ".env",
            "composer.json",
            "package.json",
            "wp-config.php.bak",
            "config.php.bak",
            ".htaccess.bak",
            "backup.sql",
            "database.sql",
            "dump.sql",
            "web.config.bak",
        ]

        async with aiohttp.ClientSession() as session:
            tasks = []

            async def check_single_file(file):
                url = f"{self.target.rstrip('/')}/{file}"
                try:
                    async with self.semaphore:
                        async with session.get(url, timeout=3) as response:
                            if response.status == 200:
                                content = await response.text(errors="ignore")
                                sensitive_files.append(
                                    {
                                        "file": file,
                                        "url": url,
                                        "size": len(content),
                                        "snippet": content[:100] if content else "",
                                    }
                                )
                except:
                    pass

            for f in files_to_check:
                tasks.append(asyncio.create_task(check_single_file(f)))
            await asyncio.gather(*tasks)

        return sensitive_files

    async def check_security_headers(self) -> Dict:
        """Check for security-related HTTP headers."""
        security_headers = {"present": {}, "missing": [], "issues": []}

        important_headers = {
            "strict-transport-security": "Protects against protocol downgrade attacks",
            "x-frame-options": "Prevents clickjacking attacks",
            "x-content-type-options": "Prevents MIME-sniffing attacks",
            "x-xss-protection": "Enables browser XSS filtering",
            "content-security-policy": "Defines content loading policies",
            "x-permitted-cross-domain-policies": "Controls data handling across domains",
            "referrer-policy": "Controls referrer information leakage",
            "expect-ct": "Certificate Transparency enforcement",
            "permissions-policy": "Controls browser features",
            "cache-control": "Defines caching policies",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.target, timeout=5) as response:
                    raw_headers = response.headers
                    headers = {k.lower(): v for k, v in raw_headers.items()}

                    for header, description in important_headers.items():
                        if header in headers:
                            security_headers["present"][header] = headers[header]
                        else:
                            security_headers["missing"].append(
                                {"header": header, "description": description}
                            )

                    # Analyze STS
                    sts_key = "strict-transport-security"
                    if sts_key in headers:
                        max_age_match = re.search(r"max-age=(\d+)", headers[sts_key])
                        if max_age_match:
                            max_age = int(max_age_match.group(1))
                            if max_age < 31536000:
                                security_headers["issues"].append(
                                    {
                                        "header": sts_key,
                                        "issue": f"max-age is too short: {max_age} seconds",
                                    }
                                )
                        else:
                            security_headers["issues"].append(
                                {
                                    "header": sts_key,
                                    "issue": "max-age directive missing",
                                }
                            )

                    # Analyze CSP
                    csp_key = "content-security-policy"
                    if csp_key in headers:
                        csp_value = headers[csp_key]
                        if "unsafe-inline" in csp_value or "unsafe-eval" in csp_value:
                            security_headers["issues"].append(
                                {
                                    "header": csp_key,
                                    "issue": "CSP contains unsafe directives like 'unsafe-inline' or 'unsafe-eval'",
                                }
                            )

                    # Analyze X-XSS-Protection
                    x_xss_key = "x-xss-protection"
                    if x_xss_key in headers:
                        if headers[x_xss_key] != "1; mode=block":
                            security_headers["issues"].append(
                                {
                                    "header": x_xss_key,
                                    "issue": f"Unexpected value: {headers[x_xss_key]}",
                                }
                            )

                    # Analyze X-Frame-Options
                    xfo_key = "x-frame-options"
                    if xfo_key in headers:
                        if headers[xfo_key].upper() not in ["DENY", "SAMEORIGIN"]:
                            security_headers["issues"].append(
                                {
                                    "header": xfo_key,
                                    "issue": f"Unexpected value: {headers[xfo_key]}",
                                }
                            )
        except Exception as e:
            security_headers["error"] = str(e)

        return security_headers

    async def check_ssl(self) -> List[Dict]:
        """Check SSL certificates and identify potential SSL issues."""
        ssl_issues = []
        from urllib.parse import urlparse

        parsed = urlparse(self.target)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        if parsed.scheme != "https":
            ssl_issues.append(
                {"type": "not_https", "details": "Target is not using HTTPS."}
            )
            return ssl_issues

        # Doing an SSL check asynchronously is trickier in Python; we can wrap
        # the blocking ssl socket connection in a run_in_executor, for example.
        async def ssl_check():
            context = ssl.create_default_context()
            try:
                with socket.create_connection((hostname, port), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        cert = ssock.getpeercert()
                        # Check certificate expiration
                        not_after = cert.get("notAfter")
                        expiration_date = datetime.strptime(
                            not_after, "%b %d %H:%M:%S %Y %Z"
                        )
                        if expiration_date < datetime.utcnow():
                            ssl_issues.append(
                                {
                                    "type": "expired_certificate",
                                    "details": f"Certificate expired on {not_after}.",
                                }
                            )

                        # Check cipher
                        cipher = ssock.cipher()
                        if cipher and cipher[0] in ["RC4-SHA", "DES-CBC3-SHA"]:
                            ssl_issues.append(
                                {
                                    "type": "weak_cipher",
                                    "details": f"Weak cipher detected: {cipher[0]}",
                                }
                            )
            except ssl.SSLError as e:
                ssl_issues.append({"type": "ssl_error", "details": str(e)})
            except Exception as e:
                ssl_issues.append({"type": "error", "details": str(e)})

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ssl_check)
        return ssl_issues

    async def check_server_config(self) -> Dict:
        """Retrieve and analyze server configuration from headers."""
        server_config = {}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.target, timeout=5) as response:
                    raw_headers = response.headers
                    # Example: Extract Server header
                    if "Server" in raw_headers:
                        server_config["Server"] = raw_headers["Server"]
                    # Example: Extract X-Powered-By header
                    if "X-Powered-By" in raw_headers:
                        server_config["X-Powered-By"] = raw_headers["X-Powered-By"]
        except Exception as e:
            server_config["error"] = str(e)

        return server_config

    async def check_cloud_storage(self) -> List[Dict]:
        """Identify exposed cloud storage endpoints."""
        cloud_services = {
            "aws_s3": ["s3.amazonaws.com", "s3.amazonaws.com.bucket"],
            "azure_blob": ["blob.core.windows.net"],
            "google_cloud_storage": ["storage.googleapis.com"],
        }

        exposed_endpoints = []
        for subdomain in self.results["subdomains"]:
            for service, patterns in cloud_services.items():
                for pattern in patterns:
                    if pattern in subdomain:
                        exposed_endpoints.append(
                            {"service": service, "subdomain": subdomain}
                        )
        return exposed_endpoints

    async def check_injection_points(self) -> List[Dict]:
        """Detect potential SQL and XSS injection points."""
        injection_issues = []
        test_payloads = {
            "sql": ["' OR '1'='1", '" OR "1"="1', "'; DROP TABLE users; --"],
            "xss": ["<script>alert('XSS')</script>", "'><script>alert(1)</script>"],
        }

        from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

        try:
            parsed_url = urlparse(self.target)
            base_query = parse_qs(parsed_url.query)

            async with aiohttp.ClientSession() as session:
                for param in ["q", "search", "query", "id"]:
                    for injection_type, payloads in test_payloads.items():
                        for payload in payloads:
                            query_params = dict(base_query)
                            query_params[param] = payload
                            encoded_query = urlencode(query_params, doseq=True)
                            vulnerable_url = urlunparse(
                                (
                                    parsed_url.scheme,
                                    parsed_url.netloc,
                                    parsed_url.path,
                                    parsed_url.params,
                                    encoded_query,
                                    parsed_url.fragment,
                                )
                            )
                            try:
                                async with self.semaphore:
                                    async with session.get(
                                        vulnerable_url, timeout=5
                                    ) as resp:
                                        content = await resp.text(errors="ignore")
                                        # If the payload is reflected in the content, naive check
                                        if payload in content:
                                            injection_issues.append(
                                                {
                                                    "type": f"{injection_type}_injection",
                                                    "parameter": param,
                                                    "payload": payload,
                                                    "url": vulnerable_url,
                                                }
                                            )
                            except:
                                pass
        except Exception as e:
            injection_issues.append({"type": "error", "details": str(e)})

        return injection_issues

    async def scan_api(self, ports: Optional[List[int]] = None) -> List[Dict]:
        """
        Scan for common API endpoints across specified ports asynchronously.
        """
        api_endpoints_found = []
        ports_to_scan = ports or self.ports

        api_paths = [
            "/openapi.json",
            "/openapi.yaml",
            "/docs",
            "/api/graphql",
            "/swagger",
            "/api/docs",
            "/v1/graphql",
            "/swagger-ui",
            "/api/explorer",
            "/swagger-ui.html",
            "/api/v1/docs",
            "/explorer",
            "/swagger.json",
            "/documentation",
            "/reference",
            "/api-docs",
            "/api/documentation",
            "/api/reference",
            "/api/swagger",
            "/graphql",
            "/api/v2",
            "/api/v3",
            "/api/keys",
            "/health",
            "/keys",
            "/status",
            "/ping",
            "/api/swagger-resources",
            "/v2/api-docs",
            "/v3/api-docs",
            "/.well-known",
            "/api/openapi",
            "/api/spec",
            "/api/schema",
            "/api",
            "/api/v1",
            "/rest",
            "/rest/v1",
            "/api/rest",
            "/api/rest/v1",
            "/api/health",
            "/api/status",
            "/api/metrics",
            "/metrics",
            "/health",
            "/healthchek",
            "/actuator",
            "/actuator/health",
            "/actuator/metrics",
            "/management",
            "/management/health",
            "/api/latest",
            "/api/beta",
            "/api/alpha",
            "/api/stable",
            "/api/experimental",
            "/api/internal",
            "/api/v4",
            "/api/v5",
            "/api/v6",
            "/api/v7",
            "/api/v8",
            "/api/v9",
            "/api/v10",
            "/api/version",
            "/api/versions",
            "/api/info",
            "/api/config",
            "/api/settings",
            "/auth",
            "/oauth",
            "/oauth2",
            "/login",
            "/api/auth",
            "/api/login",
            "/auth/login",
            "/api/oauth",
            "/api/token",
            "/api/refresh",
            "/api/logout",
            "/api/register",
            "/api/password",
            "/api/verify",
            "/api/2fa",
            "/api/sso",
            "/api/oidc",
            "/api/jwt",
            "/ws",
            "/websocket",
            "/socket",
            "/socket.io",
            "/api/ws",
            "/api/websocket",
            "/api/socket",
            "/api/socket.io",
            "/wss",
            "/api/wss",
            "/api/socket.io/1",
            "/api/socket.io/2",
            "/app",
            "/static",
            "/assets",
            "/dist",
            "/build",
            "/public",
            "/_next",
            "/webpack",
            "/index.html",
            "/favicon.ico",
            "/robots.txt",
            "/sitemap.xml",
            "/rails/info",
            "/laravel",
            "/django",
            "/spring",
            "/flask",
            "/express",
            "/nginx_status",
            "/server-status",
            "/rails/logs",
            "/rails/db",
            "/laravel/storage",
            "/laravel/public",
            "/django/admin",
            "/django/media",
            "/spring/actuator",
            "/flask/debug",
            "/express/logs",
            "/nginx/conf",
            "/apache/status",
            "/debug",
            "/dev",
            "/test",
            "/api/debug",
            "/api/dev",
            "/api/test",
            "/api/console",
            "/admin",
            "/api/admin",
            "/api/debug/logs",
            "/api/dev/logs",
            "/api/test/logs",
            "/api/console/logs",
            "/api/admin/logs",
            "/api/debug/info",
            "/api/dev/info",
            "/api/test/info",
            "/api/console/info",
            "/api/admin/info",
            "/api/gql",
            "/api/graphql",
            "/api/graphql/",
            "/gql",
            "/socket.io/v3",
            "/wss/v2",
        ]

        async with aiohttp.ClientSession() as session:

            async def scan_port_api(port):
                # Determine protocol based on port
                if port in [443, 8443]:
                    protocol = "https"
                else:
                    protocol = "http"

                base_domain = self._get_base_domain()
                for path in api_paths:
                    url = f"{protocol}://{base_domain}:{port}{path}"
                    try:
                        async with self.semaphore:
                            async with session.get(url, timeout=3) as response:
                                if response.status in [200, 301, 302]:
                                    api_endpoints_found.append(
                                        {
                                            "port": port,
                                            "path": path,
                                            "url": url,
                                            "status_code": response.status,
                                        }
                                    )
                    except Exception:
                        # You can optionally log errors here
                        pass

            tasks = [asyncio.create_task(scan_port_api(p)) for p in ports_to_scan]
            await asyncio.gather(*tasks)

        self.results["api_endpoints"] = api_endpoints_found
        return api_endpoints_found

    def _aggregate_vulnerabilities(self):
        """Aggregate various issues into the vulnerabilities list."""
        # SSL Issues
        for issue in self.results.get("ssl_issues", []):
            self.results["vulnerabilities"].append(issue)

        # CORS Vulnerabilities
        for vuln in self.results.get("cors_config", {}).get("vulnerabilities", []):
            self.results["vulnerabilities"].append(vuln)

        # CSRF Vulnerabilities
        for vuln in self.results.get("csrf_vulnerabilities", []):
            self.results["vulnerabilities"].append(vuln)

        # Injection Points
        for injection in self.results.get("injection_points", []):
            self.results["vulnerabilities"].append(injection)

        # Security Headers Issues
        for issue in self.results.get("security_headers", {}).get("issues", []):
            self.results["vulnerabilities"].append(issue)

        # File Disclosure
        if self.results.get("file_disclosure"):
            self.results["vulnerabilities"].append(
                {
                    "type": "file_disclosure",
                    "details": f"{len(self.results['file_disclosure'])} sensitive files disclosed.",
                }
            )

        # API Endpoints
        if self.results.get("api_endpoints"):
            self.results["vulnerabilities"].append(
                {
                    "type": "api_endpoints_discovery",
                    "details": f"{len(self.results['api_endpoints'])} API endpoints discovered.",
                }
            )

    async def scan(
        self, scan_api_enabled: bool = False, api_ports: Optional[List[int]] = None
    ) -> Dict:
        """Perform all scanning tasks and compile results asynchronously."""
        print(f"[*] Starting async scan on {self.target}")

        # Create tasks
        task_subdomains = asyncio.create_task(self.enumerate_subdomains())
        task_dns_records = asyncio.create_task(self.check_dns_records())
        task_ports = asyncio.create_task(self.scan_ports())
        task_dirs = asyncio.create_task(self.directory_bruteforce())
        task_tech = asyncio.create_task(self.detect_web_technologies())
        task_cors = asyncio.create_task(self.check_cors_config())
        task_csrf = asyncio.create_task(self.check_csrf_protection())
        task_files = asyncio.create_task(self.check_file_disclosure())
        task_headers = asyncio.create_task(self.check_security_headers())
        task_ssl = asyncio.create_task(self.check_ssl())
        task_server = asyncio.create_task(self.check_server_config())
        # Cloud storage depends on subdomain results, but we can still gather them at the end.
        task_injection = asyncio.create_task(self.check_injection_points())

        # Await them in parallel
        (
            subdomains,
            dns_records,
            open_ports,
            directories,
            techs,
            cors,
            csrf,
            files,
            security_headers,
            ssl_issues,
            server_config,
            injection_pts,
        ) = await asyncio.gather(
            task_subdomains,
            task_dns_records,
            task_ports,
            task_dirs,
            task_tech,
            task_cors,
            task_csrf,
            task_files,
            task_headers,
            task_ssl,
            task_server,
            task_injection,
        )

        # Update partial results
        self.results["subdomains"] = subdomains
        self.results["dns_records"] = dns_records
        self.results["open_ports"] = open_ports
        self.results["directories"] = directories
        self.results["web_technologies"] = techs
        self.results["cors_config"] = cors
        self.results["csrf_vulnerabilities"] = csrf
        self.results["file_disclosure"] = files
        self.results["security_headers"] = security_headers
        self.results["ssl_issues"] = ssl_issues
        self.results["server_config"] = server_config
        self.results["injection_points"] = injection_pts

        # Now we can check cloud storage after subdomains are populated
        cloud_storage = await self.check_cloud_storage()
        self.results["cloud_storage"] = cloud_storage

        # Optionally perform API scan
        if scan_api_enabled:
            await self.scan_api(api_ports)

        # Aggregate vulnerabilities
        self._aggregate_vulnerabilities()

        print(f"[+] Async scan completed on {self.target}")

        # Print results as JSON
        try:
            json_output = json.dumps(self.results, indent=4)
            print(json_output)
        except Exception as e:
            print(f"[-] Failed to convert results to JSON: {e}")

        return self.results


# --------------------------------------------------------
# Example usage:
#
# async def main():
#     scanner = APIScanner("example.com")
#     results = await scanner.scan(scan_api_enabled=True)
#     # Do something with results...
#
# if __name__ == "__main__":
#     asyncio.run(main())
# --------------------------------------------------------


# Standalone helper to quickly run an API scan
async def scan_api(target: str, ports: Optional[List[int]] = None) -> Dict:
    scanner = APIScanner(target)
    return await scanner.scan(scan_api_enabled=True, api_ports=ports)
