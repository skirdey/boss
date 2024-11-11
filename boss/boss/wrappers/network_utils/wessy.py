#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import queue
import re
import socket
import ssl
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Dict, List

import dns.resolver


class WESSyScanner:
    def __init__(self, target: str, ports: List[int] = None, threads: int = 10):
        """Initialize the WESSy scanner with extended capabilities."""
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
        self.threads = threads
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
        }

        # Common directories and files to check
        self._load_wordlists()

        # Initialize regex patterns for technology detection
        self._init_detection_patterns()

    def _clean_target(self, target: str) -> str:
        """Clean and validate the target URL."""
        if not target.startswith(("http://", "https://")):
            target = "http://" + target
        parsed = urllib.parse.urlparse(target)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _get_base_domain(self) -> str:
        """Extract the base domain from the target."""
        parsed = urllib.parse.urlparse(self.target)
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

    def enumerate_subdomains(self) -> List[str]:
        """Enumerate subdomains using DNS queries and common prefixes."""
        subdomains = set()
        base_domain = self._get_base_domain()

        def try_subdomain(prefix):
            try:
                subdomain = f"{prefix}.{base_domain}"
                answers = dns.resolver.resolve(subdomain, "A")
                if answers:
                    subdomains.add(subdomain)
                    # Try to get CNAME records
                    try:
                        cname_answers = dns.resolver.resolve(subdomain, "CNAME")
                        for rdata in cname_answers:
                            subdomains.add(str(rdata.target).rstrip("."))
                    except:
                        pass
            except:
                pass

        # Use ThreadPoolExecutor for parallel subdomain enumeration
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            executor.map(try_subdomain, self.subdomain_prefixes)

        return list(subdomains)

    def check_dns_records(self) -> Dict:
        """Check various DNS records for the domain."""
        dns_records = {}
        record_types = ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME", "PTR"]

        domain = self._get_base_domain()
        for record_type in record_types:
            try:
                answers = dns.resolver.resolve(domain, record_type)
                dns_records[record_type] = [str(rdata) for rdata in answers]
            except:
                continue

        return dns_records

    def scan_ports(self) -> List[int]:
        """Scan the specified ports to find open ones."""
        open_ports = []
        target_ip = self._resolve_target()

        def scan_port(port):
            try:
                with socket.create_connection((target_ip, port), timeout=1):
                    open_ports.append(port)
            except:
                pass

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            executor.map(scan_port, self.ports)

        return open_ports

    def _resolve_target(self) -> str:
        """Resolve the target hostname to an IP address."""
        try:
            parsed = urllib.parse.urlparse(self.target)
            return socket.gethostbyname(parsed.netloc)
        except Exception:
            return ""

    def directory_bruteforce(self) -> List[Dict]:
        """Perform directory and file enumeration."""
        found_paths = []
        url_queue = queue.Queue()

        # Add common directories and files to the queue
        for path in self.common_dirs + self.common_files:
            url_queue.put(path)

        def check_path():
            while True:
                try:
                    path = url_queue.get_nowait()
                except queue.Empty:
                    break

                url = f"{self.target.rstrip('/')}/{path}"
                try:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "WESSyScanner"}
                    )
                    response = urllib.request.urlopen(request, timeout=3)
                    if response.getcode() == 200:
                        content = response.read()
                        found_paths.append(
                            {
                                "path": path,
                                "code": response.getcode(),
                                "size": len(content),
                                "type": "file" if "." in path else "directory",
                            }
                        )
                except:
                    continue

        # Use multiple threads for directory bruteforce
        threads = []
        for _ in range(self.threads):
            t = threading.Thread(target=check_path)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return found_paths

    def detect_web_technologies(self) -> List[Dict]:
        """Detect web technologies, frameworks, and libraries."""
        technologies = []

        try:
            request = urllib.request.Request(
                self.target, headers={"User-Agent": "WESSyScanner"}
            )
            response = urllib.request.urlopen(request, timeout=5)
            content = response.read().decode("utf-8", errors="ignore")
            headers = dict(response.headers)

            # Check for CMS and frameworks
            for tech, patterns in self.tech_patterns.items():
                # Check content patterns
                for pattern in patterns["pattern"]:
                    if re.search(pattern, content, re.I):
                        technologies.append(
                            {"name": tech, "confidence": "high", "found_by": "pattern"}
                        )
                        break

                # Check headers
                for header_pattern in patterns["headers"]:
                    if ": " in header_pattern:
                        header_name, header_value = header_pattern.split(": ", 1)
                        header_value_lower = header_value.lower()
                        if header_name.lower() in headers:
                            if (
                                header_value_lower
                                in headers[header_name.lower()].lower()
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

        except Exception:
            pass

        return technologies

    def check_cors_config(self) -> Dict:
        """Check CORS configuration and potential misconfigurations."""
        cors_results = {
            "enabled": False,
            "allow_all_origins": False,
            "allow_credentials": False,
            "allowed_origins": [],
            "vulnerabilities": [],
        }

        try:
            # Send request with Origin header
            headers = {
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "GET,POST,PUT,DELETE",
            }
            request = urllib.request.Request(self.target, headers=headers)
            response = urllib.request.urlopen(request, timeout=5)
            response_headers = dict(response.headers)

            if "access-control-allow-origin" in response_headers:
                cors_results["enabled"] = True
                origin = response_headers["access-control-allow-origin"]

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

                if "access-control-allow-credentials" in response_headers:
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

    def check_csrf_protection(self) -> List[Dict]:
        """Check for CSRF vulnerabilities."""
        csrf_issues = []

        try:
            # Check for forms
            request = urllib.request.Request(
                self.target, headers={"User-Agent": "WESSyScanner"}
            )
            response = urllib.request.urlopen(request, timeout=5)
            content = response.read().decode("utf-8", errors="ignore")

            # Look for forms without CSRF tokens
            forms = re.findall(r"<form[^>]*>.*?</form>", content, re.DOTALL | re.I)
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
                            "evidence": form[:100] + "...",  # First 100 chars of form
                        }
                    )

            # Check security headers related to CSRF
            headers = dict(response.headers)
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

    def check_file_disclosure(self) -> List[Dict]:
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

        for file in files_to_check:
            url = f"{self.target.rstrip('/')}/{file}"
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "WESSyScanner"}
                )
                response = urllib.request.urlopen(request, timeout=3)
                if response.getcode() == 200:
                    content = response.read().decode("utf-8", errors="ignore")
                    sensitive_files.append(
                        {
                            "file": file,
                            "url": url,
                            "size": len(content),
                            "snippet": content[:100] if len(content) > 0 else "",
                        }
                    )
            except:
                continue

        return sensitive_files

    def check_security_headers(self) -> Dict:
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
            request = urllib.request.Request(
                self.target, headers={"User-Agent": "WESSyScanner"}
            )
            response = urllib.request.urlopen(request, timeout=5)
            headers = dict(response.headers)

            # Normalize header keys to lowercase
            headers = {k.lower(): v for k, v in headers.items()}

            # Check present headers
            for header, description in important_headers.items():
                if header in headers:
                    security_headers["present"][header] = headers[header]
                else:
                    security_headers["missing"].append(
                        {"header": header, "description": description}
                    )

            # Analyze header values
            if "strict-transport-security" in headers:
                max_age_match = re.search(
                    r"max-age=(\d+)", headers["strict-transport-security"]
                )
                if max_age_match:
                    max_age = int(max_age_match.group(1))
                    if max_age < 31536000:  # Less than 1 year
                        security_headers["issues"].append(
                            {
                                "header": "strict-transport-security",
                                "issue": f"max-age is too short: {max_age} seconds",
                            }
                        )
                else:
                    security_headers["issues"].append(
                        {
                            "header": "strict-transport-security",
                            "issue": "max-age directive missing",
                        }
                    )

            if "content-security-policy" in headers:
                csp = headers["content-security-policy"]
                if "unsafe-inline" in csp or "unsafe-eval" in csp:
                    security_headers["issues"].append(
                        {
                            "header": "content-security-policy",
                            "issue": "CSP contains unsafe directives like 'unsafe-inline' or 'unsafe-eval'",
                        }
                    )

            if "x-xss-protection" in headers:
                if headers["x-xss-protection"] != "1; mode=block":
                    security_headers["issues"].append(
                        {
                            "header": "x-xss-protection",
                            "issue": f"Unexpected value: {headers['x-xss-protection']}",
                        }
                    )

            if "x-frame-options" in headers:
                if headers["x-frame-options"].upper() not in ["DENY", "SAMEORIGIN"]:
                    security_headers["issues"].append(
                        {
                            "header": "x-frame-options",
                            "issue": f"Unexpected value: {headers['x-frame-options']}",
                        }
                    )

        except Exception as e:
            security_headers["error"] = str(e)

        return security_headers

    def check_ssl(self) -> List[Dict]:
        """Check SSL certificates and identify potential SSL issues."""
        ssl_issues = []

        try:
            parsed = urllib.parse.urlparse(self.target)
            hostname = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            if parsed.scheme != "https":
                ssl_issues.append(
                    {"type": "not_https", "details": "Target is not using HTTPS."}
                )
                return ssl_issues

            context = ssl.create_default_context()
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

                    # Check for weak ciphers (this is a simplified check)
                    cipher = ssock.cipher()
                    if cipher[0] in ["RC4-SHA", "DES-CBC3-SHA"]:
                        ssl_issues.append(
                            {
                                "type": "weak_cipher",
                                "details": f"Weak cipher detected: {cipher[0]}",
                            }
                        )

                    # Check for certificate chain issues
                    # Note: Detailed chain validation requires more complex handling
                    # Here we assume that if connection is successful, the chain is valid

        except ssl.SSLError as e:
            ssl_issues.append({"type": "ssl_error", "details": str(e)})
        except Exception as e:
            ssl_issues.append({"type": "error", "details": str(e)})

        return ssl_issues

    def check_server_config(self) -> Dict:
        """Retrieve and analyze server configuration from headers."""
        server_config = {}

        try:
            request = urllib.request.Request(
                self.target, headers={"User-Agent": "WESSyScanner"}
            )
            response = urllib.request.urlopen(request, timeout=5)
            headers = dict(response.headers)

            # Example: Extract Server header
            if "Server" in headers:
                server_config["Server"] = headers["Server"]

            # Example: Extract X-Powered-By header
            if "X-Powered-By" in headers:
                server_config["X-Powered-By"] = headers["X-Powered-By"]

            # Add more server configuration checks as needed

        except Exception as e:
            server_config["error"] = str(e)

        return server_config

    def check_cloud_storage(self) -> List[Dict]:
        """Identify exposed cloud storage endpoints."""
        cloud_services = {
            "aws_s3": ["s3.amazonaws.com", "s3.amazonaws.com.bucket"],
            "azure_blob": ["blob.core.windows.net"],
            "google_cloud_storage": ["storage.googleapis.com"],
        }

        exposed_endpoints = []

        # Enumerate subdomains and check against known cloud storage patterns
        for subdomain in self.results["subdomains"]:
            for service, patterns in cloud_services.items():
                for pattern in patterns:
                    if pattern in subdomain:
                        exposed_endpoints.append(
                            {"service": service, "subdomain": subdomain}
                        )

        return exposed_endpoints

    def check_injection_points(self) -> List[Dict]:
        """Detect potential SQL and XSS injection points."""
        injection_issues = []
        test_payloads = {
            "sql": ["' OR '1'='1", '" OR "1"="1', "'; DROP TABLE users; --"],
            "xss": ["<script>alert('XSS')</script>", "'><script>alert(1)</script>"],
        }

        try:
            for param in ["q", "search", "query", "id"]:
                for injection_type, payloads in test_payloads.items():
                    for payload in payloads:
                        parsed_url = urllib.parse.urlparse(self.target)
                        query_params = urllib.parse.parse_qs(parsed_url.query)
                        query_params[param] = payload
                        encoded_query = urllib.parse.urlencode(query_params, doseq=True)
                        vulnerable_url = urllib.parse.urlunparse(
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
                            request = urllib.request.Request(
                                vulnerable_url, headers={"User-Agent": "WESSyScanner"}
                            )
                            response = urllib.request.urlopen(request, timeout=5)
                            content = response.read().decode("utf-8", errors="ignore")

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
                            continue

        except Exception as e:
            injection_issues.append({"type": "error", "details": str(e)})

        return injection_issues

    def scan(self) -> Dict:
        """Perform all scanning tasks, print results as JSON, and return the JSON object."""
        print(f"[*] Starting scan on {self.target}")

        # Start concurrent tasks
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            future_dns = executor.submit(self.enumerate_subdomains)
            future_dns_records = executor.submit(self.check_dns_records)
            future_ports = executor.submit(self.scan_ports)
            future_dirs = executor.submit(self.directory_bruteforce)
            future_tech = executor.submit(self.detect_web_technologies)
            future_cors = executor.submit(self.check_cors_config)
            future_csrf = executor.submit(self.check_csrf_protection)
            future_files = executor.submit(self.check_file_disclosure)
            future_headers = executor.submit(self.check_security_headers)
            future_ssl = executor.submit(self.check_ssl)
            future_server = executor.submit(self.check_server_config)
            future_cloud = executor.submit(self.check_cloud_storage)
            future_injection = executor.submit(self.check_injection_points)

            # Collect results
            self.results["subdomains"] = future_dns.result()
            self.results["dns_records"] = future_dns_records.result()
            self.results["open_ports"] = future_ports.result()
            self.results["directories"] = future_dirs.result()
            self.results["web_technologies"] = future_tech.result()
            self.results["cors_config"] = future_cors.result()
            self.results["csrf_vulnerabilities"] = future_csrf.result()
            self.results["file_disclosure"] = future_files.result()
            self.results["security_headers"] = future_headers.result()
            self.results["ssl_issues"] = future_ssl.result()
            self.results["server_config"] = future_server.result()
            self.results["cloud_storage"] = future_cloud.result()
            self.results["injection_points"] = future_injection.result()

        # Aggregate vulnerabilities
        self._aggregate_vulnerabilities()

        print(f"[+] Scan completed on {self.target}")

        # Print results as JSON
        try:
            json_output = json.dumps(self.results, indent=4)
            print(json_output)
        except Exception as e:
            print(f"[-] Failed to convert results to JSON: {e}")

        return self.results

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

    # Additional methods can be added here as needed


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WESSyScanner - Comprehensive Web Security Scanner"
    )
    parser.add_argument("target", help="Target URL to scan (e.g., https://example.com)")
    parser.add_argument(
        "-p",
        "--ports",
        help="Comma-separated list of ports to scan (default: common ports)",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--threads",
        help="Number of concurrent threads (default: 10)",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main() -> Dict:
    args = parse_arguments()

    # Parse ports
    if args.ports:
        try:
            ports = [int(port.strip()) for port in args.ports.split(",")]
        except ValueError:
            print(
                "[-] Invalid ports specified. Please provide a comma-separated list of integers."
            )
            sys.exit(1)
    else:
        ports = None

    scanner = WESSyScanner(target=args.target, ports=ports, threads=args.threads)
    results = scanner.scan()
    return results
