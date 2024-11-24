import socket

from wrappers.network_utils import parse_target


def reverse_dns_lookup(target):
    """
    Performs a reverse DNS lookup on the specified target.

    :param target: The target as a URL, hostname, or IP address.
    :return: Dictionary mapping IP addresses to their PTR records.
    """
    hostname, ip_addresses = parse_target(target)

    if not ip_addresses:
        print("No IP addresses to perform reverse DNS lookup.")
        return {}

    reverse_dns_results = {}

    for ip in ip_addresses:
        try:
            ptr_record, aliases, _ = socket.gethostbyaddr(ip)
            reverse_dns_results[ip] = ptr_record
        except socket.herror:
            reverse_dns_results[ip] = "No PTR record found."
        except Exception as e:
            reverse_dns_results[ip] = f"Error: {e}"

    return reverse_dns_results
