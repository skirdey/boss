import asyncio
import socket

from boss.wrappers.network_utils.parse_target import parse_target


async def scan_port(ip, port, timeout):
    try:
        # Create a coroutine-friendly socket
        conn = asyncio.open_connection(ip, port)
        reader, writer = await asyncio.wait_for(conn, timeout=timeout)
        # If connection is successful, retrieve the service name
        try:
            service = socket.getservbyport(port, "tcp")
        except OSError:
            service = "unknown"
        writer.close()
        await writer.wait_closed()
        return port, service
    except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
        # Log the exception and return None
        print(f"Port {port} closed or unreachable: {e}")
        return None
    except Exception as e:
        # Catch any other exceptions
        print(f"Unexpected error on port {port}: {e}")
        return None


async def scan_single_ip(ip, ports, timeout, semaphore):
    open_ports = []

    async def semaphore_scan(port):
        async with semaphore:
            result = await scan_port(ip, port, timeout)
            if result:
                open_ports.append(result)
                print(f"Port {result[0]} open ({result[1]})")

    tasks = [semaphore_scan(port) for port in ports]
    await asyncio.gather(*tasks)
    return sorted(open_ports, key=lambda x: x[0])


# Modify the scan_ports function to accept a list of ports
def scan_ports(target, ports_to_scan, timeout=0.5, max_concurrent=100):
    """
    Asynchronously scans TCP ports on the specified target and returns a report.

    :param target: The target as a URL, hostname, or IP address.
    :param ports_to_scan: List of ports to scan.
    :param timeout: Timeout in seconds for each port scan attempt.
    :param max_concurrent: Maximum number of concurrent connections.
    :return: A formatted string listing all open ports and their services.
    """
    try:
        hostname, ip_addresses = parse_target(target)
    except Exception as e:
        print(f"Error parsing target: {e}")
        return ""

    if not ip_addresses:
        print("No IP addresses to scan.")
        return ""

    async def main():
        semaphore = asyncio.Semaphore(max_concurrent)
        scan_results = {}
        for ip in ip_addresses:
            print(f"\nStarting scan on IP: {ip}")
            try:
                open_ports = await scan_single_ip(ip, ports_to_scan, timeout, semaphore)
                scan_results[ip] = open_ports
                print(f"Scan on {ip} completed. {len(open_ports)} open ports found.")
            except Exception as e:
                print(f"Error scanning IP {ip}: {e}")
                continue
        return scan_results

    try:
        scan_results = asyncio.run(main())
    except Exception as e:
        print(f"Error during asynchronous scanning: {e}")
        return ""

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
