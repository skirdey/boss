import socket
import ssl
from datetime import datetime


def get_ssl_certificate(host, port=443):
    """
    Retrieves the SSL certificate from the specified host.

    :param host: The target hostname.
    :param port: The port to connect to (default is 443).
    :return: Dictionary containing certificate details or empty dict on error.
    """
    context = ssl.create_default_context()

    try:
        with socket.create_connection((host, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()

        # Process certificate only if we have all required fields
        if not all(
            key in cert for key in ["subject", "issuer", "notBefore", "notAfter"]
        ):
            return {}

        certificate = {}
        try:
            certificate["subject"] = dict(x[0] for x in cert["subject"])
            certificate["issuer"] = dict(x[0] for x in cert["issuer"])
            certificate["version"] = cert.get("version")
            certificate["serialNumber"] = cert.get("serialNumber")
            certificate["notBefore"] = datetime.strptime(
                cert["notBefore"], "%b %d %H:%M:%S %Y %Z"
            )
            certificate["notAfter"] = datetime.strptime(
                cert["notAfter"], "%b %d %H:%M:%S %Y %Z"
            )
            certificate["subjectAltName"] = cert.get("subjectAltName")
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing certificate data: {e}")
            return {}

        return certificate
    except Exception as e:
        print(f"Error retrieving SSL certificate: {e}")
        return {}
