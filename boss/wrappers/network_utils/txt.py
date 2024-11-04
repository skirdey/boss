import socket
import struct


def get_txt_records(domain):
    """
    Retrieves TXT records for a given domain by performing a DNS query.

    :param domain: The domain to query.
    :return: List of TXT records as strings.
    """
    # DNS server to query
    dns_server = "8.8.8.8"
    port = 53
    timeout = 5

    # Construct DNS query
    transaction_id = b"\xbb\xbb"  # Random transaction ID
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

    # Query type TXT (16) and class IN (1)
    query += struct.pack(">HH", 16, 1)

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

        txt_records = []
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

            if rtype == 16:  # TXT record
                txt_length = data[pointer]
                pointer += 1
                txt = data[pointer : pointer + txt_length].decode()
                txt_records.append(txt)
                pointer += txt_length
            else:
                pointer += rdlength

        return txt_records

    except Exception as e:
        print(f"Error parsing DNS response: {e}")
        return []
