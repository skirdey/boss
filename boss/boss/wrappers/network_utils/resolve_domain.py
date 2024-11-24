import socket


def resolve_domain(domain):
    """
    Resolves a domain to its IPv4 and IPv6 addresses.

    :param domain: The domain name to resolve.
    :return: Dictionary with 'A' and 'AAAA' records.
    """
    records = {"A": [], "AAAA": []}

    try:
        # IPv4 addresses
        ipv4_addresses = socket.getaddrinfo(domain, None, socket.AF_INET)
        records["A"] = list(set([addr[4][0] for addr in ipv4_addresses]))
    except socket.gaierror as e:
        print(f"Error resolving A records for {domain}: {e}")

    try:
        # IPv6 addresses
        ipv6_addresses = socket.getaddrinfo(domain, None, socket.AF_INET6)
        records["AAAA"] = list(set([addr[4][0] for addr in ipv6_addresses]))
    except socket.gaierror as e:
        print(f"Error resolving AAAA records for {domain}: {e}")

    return records
