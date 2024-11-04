import socket
import struct


def get_ns_records(domain):
    """
    Retrieves NS records for a given domain by performing a DNS query.

    :param domain: The domain to query.
    :return: List of name servers as strings.
    """
    # DNS server to query
    dns_server = "8.8.8.8"
    port = 53
    timeout = 5

    # Construct DNS query
    transaction_id = b"\xaa\xaa"  # Random transaction ID
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

    # Query type NS (2) and class IN (1)
    query += struct.pack(">HH", 2, 1)

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
            while data[pointer] != 0:
                pointer += 1
            pointer += 1 + 4  # Null byte and QTYPE/QCLASS

        ns_records = []
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

            if rtype == 2:  # NS record
                # Read NS server name
                ns = []
                while data[pointer] != 0:
                    if data[pointer] & 0xC0 == 0xC0:
                        pointer += 2
                        break
                    length = data[pointer]
                    pointer += 1
                    ns.append(data[pointer : pointer + length].decode())
                    pointer += length
                name_server = ".".join(ns)
                ns_records.append(name_server)
            else:
                pointer += rdlength

        return ns_records

    except Exception as e:
        print(f"Error parsing DNS response: {e}")
        return []


# Example usage:
if __name__ == "__main__":
    domain = "developers.inflection.ai"
    ns = get_ns_records(domain)
    print(f"NS records for {domain}:")
    for ns_server in ns:
        print(f"Name Server: {ns_server}")
