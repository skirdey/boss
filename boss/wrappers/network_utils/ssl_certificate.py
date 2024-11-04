import socket
import ssl
from datetime import datetime


def get_ssl_certificate(host, port=443):
    """
    Retrieves the SSL certificate from the specified host.

    :param host: The target hostname.
    :param port: The port to connect to (default is 443).
    :return: Dictionary containing certificate details.
    """
    context = ssl.create_default_context()

    try:
        with socket.create_connection((host, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
    except Exception as e:
        print(f"Error retrieving SSL certificate: {e}")
        return {}

    certificate = {}
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

    return certificate


# Example usage:
if __name__ == "__main__":
    host = "developers.inflection.ai"
    cert_info = get_ssl_certificate(host)
    print(f"SSL Certificate for {host}:")
    for key, value in cert_info.items():
        print(f"{key}: {value}")
