import socket
import struct

from wrappers.network_utils.parse_target import parse_target


def get_mx_records(target):
    """
    Retrieves MX records for the specified target.

    :param target: The target as a URL, hostname, or IP address.
    :return: Dictionary mapping IP addresses to their MX records.
    """
    hostname, ip_addresses = parse_target(target)

    if not hostname:
        print("No hostname available for MX record lookup.")
        return {}

    # MX record retrieval typically depends on the domain, not IP
    # So we'll use the hostname directly
    domain = hostname

    # DNS server to query
    dns_server = "8.8.8.8"
    port = 53
    timeout = 5

    # Construct DNS query for MX records
    transaction_id = b"\xdd\xdd"  # Random transaction ID
    flags = b"\x01\x00"  # Standard query
    qdcount = b"\x00\x01"  # One question
    ancount = b"\x00\x00"
    nscount = b"\x00\x00"
    arcount = b"\x00\x00"
    query = transaction_id + flags + qdcount + ancount + nscount + arcount

    # Encode domain name
    for part in domain.split("."):
        query += bytes([len(part)]) + part.encode()
    query += b"\x00"  # End of domain name

    # Query type MX (15) and class IN (1)
    query += struct.pack(">HH", 15, 1)

    # Send DNS query
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.sendto(query, (dns_server, port))
        data, _ = sock.recvfrom(512)
        sock.close()
    except socket.timeout:
        print("DNS query timed out.")
        return {}
    except Exception as e:
        print(f"Error during DNS query: {e}")
        return {}

    # Parse the response
    try:
        # Header is first 12 bytes
        header = data[:12]
        qdcount = struct.unpack(">H", header[4:6])[0]
        ancount = struct.unpack(">H", header[6:8])[0]

        # Skip the question section
        pointer = 12
        for _ in range(qdcount):
            while data[pointer] != 0:
                pointer += 1
            pointer += 1 + 4  # Null byte and QTYPE/QCLASS

        mx_records = []
        for _ in range(ancount):
            # Skip name (could be a pointer)
            if data[pointer] & 0xC0 == 0xC0:
                pointer += 2
            else:
                while data[pointer] != 0:
                    pointer += 1
                pointer += 1

            # Read type, class, ttl, data length
            rtype, rclass, ttl, rdlength = struct.unpack(
                ">HHIH", data[pointer : pointer + 10]
            )
            pointer += 10

            if rtype == 15:  # MX record
                preference = struct.unpack(">H", data[pointer : pointer + 2])[0]
                pointer += 2
                # Read exchange server name
                exchange = []
                while data[pointer] != 0:
                    if data[pointer] & 0xC0 == 0xC0:
                        pointer += 2
                        break
                    length = data[pointer]
                    pointer += 1
                    exchange.append(data[pointer : pointer + length].decode())
                    pointer += length
                mail_server = ".".join(exchange)
                mx_records.append((preference, mail_server))
            else:
                pointer += rdlength

        # Sort MX records by preference
        mx_records.sort()
        return {hostname: mx_records}

    except Exception as e:
        print(f"Error parsing DNS response: {e}")
        return {}
