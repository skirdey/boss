import concurrent.futures
import random
import socket
import time

from wrappers.network_utils import parse_target


def _validate_input_params(timeout, max_threads, min_delay, max_delay):
    """Validate input parameters for the scan_ports function."""
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    if max_threads <= 0:
        raise ValueError("Number of threads must be positive")
    if min_delay <= 0 or max_delay <= 0:
        raise ValueError("Delay values must be positive")
    if max_delay <= min_delay:
        raise ValueError("Maximum delay must be greater than minimum delay")
    return True


def scan_ports(target, timeout=0.5, max_threads=100, min_delay=0.01, max_delay=0.1):
    """
    Scans all TCP ports on the specified target (URL, hostname, or IP)
    and returns a list of open ports along with their common service names.
    Incorporates jitter and delays to avoid potential rate limiting.

    :param target: The target as a URL, hostname, or IP address.
    :param timeout: Timeout in seconds for each port scan attempt.
    :param max_threads: Maximum number of concurrent threads.
    :param min_delay: Minimum delay (in seconds) before each scan attempt.
    :param max_delay: Maximum delay (in seconds) before each scan attempt.
    :return: A formatted string listing all open ports and their services for each resolved IP.
    """
    _validate_input_params(timeout, max_threads, min_delay, max_delay)
    hostname, ip_addresses = parse_target(target)

    if not ip_addresses:
        print("No IP addresses to scan.")
        return ""

    def scan_single_ip(ip):
        open_ports = []

        def scan_port(port):
            # Introduce random delay to avoid rate limiting
            delay = random.uniform(min_delay, max_delay)
            time.sleep(delay)

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    result = s.connect_ex((ip, port))
                    if result == 0:
                        try:
                            service = socket.getservbyport(port, "tcp")
                        except OSError:
                            service = "unknown"
                        return (port, service)
            except Exception:
                pass
            return None

        print(f"\nStarting scan on IP: {ip}")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_port = {
                executor.submit(scan_port, port): port for port in range(1, 65536)
            }
            for future in concurrent.futures.as_completed(future_to_port):
                port = future_to_port[future]
                try:
                    result = future.result()
                    if result:
                        open_ports.append(result)
                        print(f"Port {result[0]} open ({result[1]})")
                except Exception as exc:
                    print(f"Port {port} generated an exception: {exc}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"Scan on {ip} completed in {duration:.2f} seconds.")

        open_ports.sort(key=lambda x: x[0])

        return open_ports

    scan_results = {}
    for ip in ip_addresses:
        scan_results[ip] = scan_single_ip(ip)

    # Format the results similar to Nmap
    report = ""
    for ip, ports in scan_results.items():
        report += f"\nScan Report for {ip} ({hostname}):\n"
        report += f"{'PORT':<10}{'STATE':<10}{'SERVICE'}\n"
        report += "-" * 30 + "\n"
        if ports:
            for port, service in ports:
                report += f"{port:<10}{'open':<10}{service}\n"
        else:
            report += "No open ports found.\n"

    return report
