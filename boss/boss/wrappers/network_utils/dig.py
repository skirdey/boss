import random
import socket
import struct
import time
from datetime import datetime, timezone
from typing import Dict, List, Union


class DNSQuery:
    """
    Simple DNS Query builder
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.query_type = 1  # A Record

    def build_query(self) -> bytes:
        """Build a DNS query packet"""
        # Random query ID
        packet = struct.pack(">H", random.randint(0, 65535))

        # Flags: Standard recursive query
        packet += struct.pack(">H", 0x0100)

        # Questions count (1), and all other counts (0)
        packet += struct.pack(">HHHH", 1, 0, 0, 0)

        # Convert domain to DNS format
        for part in self.domain.split("."):
            packet += struct.pack("B", len(part))
            packet += part.encode()

        packet += struct.pack("B", 0)  # End of domain name

        # Type A (1) and Class IN (1)
        packet += struct.pack(">HH", self.query_type, 1)

        return packet


class PythonDig:
    """
    A pure Python implementation of basic 'dig' functionality using only built-in libraries
    """

    def __init__(self):
        # Default DNS servers (Google's public DNS)
        self.dns_servers = ["8.8.8.8", "8.8.4.4"]
        self.timeout = 15
        self.port = 53

    def _parse_response(self, response: bytes) -> List[str]:
        """
        Parse DNS response packet
        Returns list of IP addresses found
        """
        try:
            # Skip header and question section
            # This is a simplified parser that looks for IP addresses
            results = []
            pos = 12

            # Skip the original question
            while True:
                length = response[pos]
                if length == 0:
                    break
                pos += length + 1
            pos += 5  # Skip type and class

            # Answer section
            answers_count = (response[6] << 8) + response[7]

            for _ in range(answers_count):
                # Skip name pointer
                while pos < len(response):
                    length = response[pos]
                    if length == 0 or length >= 192:  # Pointer
                        pos += 2
                        break
                    pos += length + 1

                if pos + 10 > len(response):
                    break

                # Read type
                r_type = (response[pos] << 8) + response[pos + 1]
                pos += 8  # Skip to rdata length

                r_len = (response[pos] << 8) + response[pos + 1]
                pos += 2

                # Type A record
                if r_type == 1 and r_len == 4:
                    ip = ".".join(str(x) for x in response[pos : pos + 4])
                    results.append(ip)

                pos += r_len

            return results if results else ["No A records found"]

        except Exception as e:
            return [f"Error parsing response: {str(e)}"]

    def _query_dns(self, domain: str) -> List[str]:
        """
        Send DNS query and get response
        """
        query = DNSQuery(domain)
        packet = query.build_query()

        for dns_server in self.dns_servers:
            try:
                # Create UDP socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(self.timeout)

                # Send query
                sock.sendto(packet, (dns_server, self.port))

                # Receive response
                response, _ = sock.recvfrom(1024)

                return self._parse_response(response)

            except socket.timeout:
                continue
            except Exception as e:
                return [f"Error querying DNS: {str(e)}"]
            finally:
                sock.close()

        return ["Query timed out"]

    def _get_response_time(self, domain: str) -> float:
        """
        Measure response time for a DNS query
        """
        try:
            start_time = time.time()
            socket.gethostbyname(domain)
            end_time = time.time()
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception:
            return -1

    def dig(self, domain: str) -> Dict[str, Union[str, List[str]]]:
        """
        Perform a DNS lookup similar to 'dig' command

        :param domain: Domain name to query
        :return: Dictionary containing DNS query results
        """
        results = {
            "domain": domain,
            "query_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "dns_server": self.dns_servers[0],
            "response_time_ms": self._get_response_time(domain),
            "records": self._query_dns(domain),
        }

        return results

    def format_results(self, results: Dict[str, Union[str, List[str]]]) -> str:
        """
        Format the results in a dig-like output
        """
        output = []
        output.append(f"; <<>> PyDig <<>> {results['domain']}")
        output.append(f";; Query time: {results['response_time_ms']:.2f} msec")
        output.append(f";; SERVER: {results['dns_server']}")
        output.append(f";; WHEN: {results['query_time']}")
        output.append(";; ANSWER SECTION:")

        for record in results["records"]:
            if not record.startswith(("No", "Error")):
                output.append(f"{results['domain']}.\t\tIN\tA\t{record}")
            else:
                output.append(f";; {record}")

        return "\n".join(output)
