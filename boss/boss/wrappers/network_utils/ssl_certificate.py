import logging
import socket
import ssl
from datetime import datetime
from typing import Any, Dict, Union

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_ssl_certificate(host: str, port: int = 443) -> Union[Dict[str, Any], str]:
    """
    Retrieves the SSL certificate from the specified host using only native Python libraries.

    :param host: The target hostname.
    :param port: The port to connect to (default is 443).
    :return: Dictionary containing certificate details or error message
    """
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED  # Changed from CERT_NONE
        context.verify_flags = ssl.VERIFY_X509_TRUSTED_FIRST

        with socket.create_connection((host, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()

                if not cert:
                    return "No certificate found"

                certificate = {}

                # Extract subject information
                subject_dict = {}
                for field in cert.get("subject", []):
                    if field and len(field) > 0:
                        key, value = field[0]
                        subject_dict[key] = value
                certificate["subject"] = subject_dict

                # Extract issuer information
                issuer_dict = {}
                for field in cert.get("issuer", []):
                    if field and len(field) > 0:
                        key, value = field[0]
                        issuer_dict[key] = value
                certificate["issuer"] = issuer_dict

                # Extract other certificate details
                certificate["version"] = cert.get("version")
                certificate["serialNumber"] = cert.get("serialNumber")

                # Parse dates
                try:
                    not_before = datetime.strptime(
                        cert["notBefore"], "%b %d %H:%M:%S %Y %Z"
                    )
                    not_after = datetime.strptime(
                        cert["notAfter"], "%b %d %H:%M:%S %Y %Z"
                    )
                    certificate["notBefore"] = not_before.isoformat()
                    certificate["notAfter"] = not_after.isoformat()
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing certificate dates: {e}")
                    certificate["notBefore"] = cert.get("notBefore", "Unknown")
                    certificate["notAfter"] = cert.get("notAfter", "Unknown")

                # Extract Subject Alternative Names (SANs)
                certificate["subjectAltName"] = cert.get("subjectAltName", [])

                # Get the certificate fingerprints
                certificate["cipher"] = ssock.cipher()

                # Get the SSL/TLS version
                certificate["tls_version"] = ssock.version()

                logger.info(f"Successfully retrieved certificate for {host}")
                return certificate

    except (socket.gaierror, socket.timeout) as e:
        error_msg = f"Connection error: {str(e)}"
        logger.error(f"{error_msg} for {host}:{port}")
        return error_msg
    except ssl.SSLError as e:
        error_msg = f"SSL error: {str(e)}"
        logger.error(f"{error_msg} for {host}:{port}")
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"{error_msg} for {host}:{port}")
        return error_msg


def format_certificate_info(cert_info: Union[Dict[str, Any], str]) -> str:
    """
    Formats the certificate information into a readable string.

    :param cert_info: The certificate information dictionary or error message
    :return: Formatted string of certificate information
    """
    if isinstance(cert_info, str):
        return f"Error: {cert_info}"

    output = []
    output.append("Certificate Information:")
    output.append("-" * 50)

    if "subject" in cert_info:
        output.append("Subject:")
        for key, value in cert_info["subject"].items():
            output.append(f"  {key}: {value}")

    if "issuer" in cert_info:
        output.append("\nIssuer:")
        for key, value in cert_info["issuer"].items():
            output.append(f"  {key}: {value}")

    if "notBefore" in cert_info:
        output.append(f"\nValid From: {cert_info['notBefore']}")
    if "notAfter" in cert_info:
        output.append(f"Valid Until: {cert_info['notAfter']}")

    if "subjectAltName" in cert_info:
        output.append("\nSubject Alternative Names:")
        for san_type, san_value in cert_info["subjectAltName"]:
            output.append(f"  {san_type}: {san_value}")

    if "tls_version" in cert_info:
        output.append(f"\nTLS Version: {cert_info['tls_version']}")

    if "cipher" in cert_info:
        output.append(f"Cipher Suite: {cert_info['cipher']}")

    return "\n".join(output)
