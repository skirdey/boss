import ipaddress
import socket
from urllib.parse import urlparse


def parse_target(target):
    """
    Parses the input target, resolving it to a hostname and a list of IP addresses.
    Supports input as URLs (with http/https), hostnames, or direct IP addresses.

    :param target: The target as a URL, hostname, or IP address.
    :return: Tuple containing:
             - hostname (str): The resolved hostname or the input IP if no hostname.
             - ip_addresses (list): List of resolved IP addresses (both IPv4 and IPv6).
    """
    hostname = ""
    ip_addresses = []

    # Attempt to parse the target as a URL
    parsed = urlparse(target)
    if parsed.scheme and parsed.hostname:
        hostname = parsed.hostname
    else:
        # If no scheme is present, treat the entire target as hostname or IP
        hostname = target

    # Check if the hostname is a valid IP address
    try:
        ip_obj = ipaddress.ip_address(hostname)
        # It's a valid IP address
        ip_addresses.append(hostname)
        return hostname, ip_addresses
    except ValueError:
        # Not a direct IP, proceed to resolve as hostname
        pass

    # Attempt to resolve the hostname to IP addresses
    try:
        # Get address info for both IPv4 and IPv6
        addrinfo = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        # Extract IP addresses, ensuring uniqueness
        ip_addresses = list(set([info[4][0] for info in addrinfo]))
    except socket.gaierror as e:
        print(f"DNS resolution failed for {hostname}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during DNS resolution for {hostname}: {e}")

    return hostname, ip_addresses
