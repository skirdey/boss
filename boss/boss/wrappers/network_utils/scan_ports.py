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
    except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
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


def scan_ports(target, timeout=0.5, max_concurrent=100, port_range=(1, 4000)):
    """
    Asynchronously scans TCP ports on the specified target and returns a report.

    :param target: The target as a URL, hostname, or IP address.
    :param timeout: Timeout in seconds for each port scan attempt.
    :param max_concurrent: Maximum number of concurrent connections.
    :param port_range: Tuple indicating the range of ports to scan.
    :return: A formatted string listing all open ports and their services.
    """
    hostname, ip_addresses = parse_target(target)

    if not ip_addresses:
        print("No IP addresses to scan.")
        return ""

    async def main():
        semaphore = asyncio.Semaphore(max_concurrent)
        scan_results = {}
        for ip in ip_addresses:
            print(f"\nStarting scan on IP: {ip}")
            open_ports = await scan_single_ip(
                ip, range(*port_range), timeout, semaphore
            )
            scan_results[ip] = open_ports
            print(f"Scan on {ip} completed. {len(open_ports)} open ports found.")
        return scan_results

    scan_results = asyncio.run(main())

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
