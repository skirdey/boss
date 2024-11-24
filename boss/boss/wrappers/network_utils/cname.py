import socket
import struct


def decode_dns_name(data, offset):
    """
    Decode a DNS name from the response, handling compression.
    Returns the decoded name and the new offset.
    """
    parts = []
    while True:
        length = data[offset]
        if length == 0:
            offset += 1
            break
        elif length & 0xC0 == 0xC0:
            # Handle name compression
            pointer = struct.unpack(">H", data[offset : offset + 2])[0] & 0x3FFF
            parts.extend(decode_dns_name(data, pointer)[0].split("."))
            offset += 2
            break
        else:
            offset += 1
            parts.append(data[offset : offset + length].decode())
            offset += length

    return ".".join(parts), offset


def get_cname_records(domain):
    """
    Retrieves CNAME records for a given domain by performing a DNS query.

    :param domain: The domain to query.
    :return: List of CNAME records as strings.
    """
    if not domain:
        return []

    # DNS server to query
    dns_server = "8.8.8.8"
    port = 53
    timeout = 5

    # Construct DNS query
    transaction_id = b"\xcc\xcc"  # Random transaction ID
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

    # Query type CNAME (5) and class IN (1)
    query += struct.pack(">HH", 5, 1)

    # Send DNS query
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.sendto(query, (dns_server, port))
        data, _ = sock.recvfrom(512)
        sock.close()
    except socket.timeout:
        print("DNS query timed out.")
        return []
    except Exception as e:
        print(f"Error during DNS query: {e}")
        return []

    # Parse the response
    try:
        # Header is first 12 bytes
        header = data[:12]
        qdcount = struct.unpack(">H", header[4:6])[0]
        ancount = struct.unpack(">H", header[6:8])[0]

        # Skip the question section
        pointer = 12
        for _ in range(qdcount):
            # Skip the domain name
            while data[pointer] != 0:
                if data[pointer] & 0xC0 == 0xC0:  # Handle compression
                    pointer += 2
                    break
                pointer += data[pointer] + 1
            if data[pointer] == 0:  # If we didn't break due to compression
                pointer += 1
            pointer += 4  # Skip QTYPE and QCLASS

        cname_records = []
        for _ in range(ancount):
            # Skip name field (using compression pointer)
            if data[pointer] & 0xC0 == 0xC0:
                pointer += 2
            else:
                while data[pointer] != 0:
                    pointer += data[pointer] + 1
                pointer += 1

            # Read type, class, ttl, data length
            rtype, rclass, ttl, rdlength = struct.unpack(
                ">HHIH", data[pointer : pointer + 10]
            )
            pointer += 10

            if rtype == 5:  # CNAME record
                cname, new_pointer = decode_dns_name(data, pointer)
                cname_records.append(cname)
                pointer = pointer + rdlength
            else:
                pointer += rdlength

        return cname_records

    except Exception as e:
        print(f"Error parsing DNS response: {e}")
        return []
