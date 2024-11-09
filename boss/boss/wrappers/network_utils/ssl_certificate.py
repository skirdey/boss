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
    :return: Dictionary containing certificate details or "resource doesn't employ SSL certificate" message
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    try:
        with socket.create_connection((host, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()

                if not cert:
                    logger.warning(f"No certificate found for {host}:{port}")
                    return "resource doesn't employ SSL certificate"

                if not all(
                    key in cert
                    for key in ["subject", "issuer", "notBefore", "notAfter"]
                ):
                    logger.warning("Certificate missing required fields")
                    return "resource doesn't employ SSL certificate"

        certificate = {}
        try:
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
                certificate["notBefore"] = datetime.strptime(
                    cert["notBefore"], "%b %d %H:%M:%S %Y %Z"
                )
                certificate["notAfter"] = datetime.strptime(
                    cert["notAfter"], "%b %d %H:%M:%S %Y %Z"
                )
            except ValueError as e:
                logger.error(f"Error parsing certificate dates: {e}")
                return "resource doesn't employ SSL certificate"

            # Extract Subject Alternative Names (SANs)
            certificate["subjectAltName"] = cert.get("subjectAltName", [])

            # Verify that we have at least some basic information
            if not certificate["subject"] or not certificate["issuer"]:
                logger.warning("Certificate missing subject or issuer information")
                return "resource doesn't employ SSL certificate"

            logger.info("Successfully retrieved certificate details")
            return certificate

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Error processing certificate data: {e}")
            return "resource doesn't employ SSL certificate"

    except (socket.gaierror, socket.timeout) as e:
        logger.error(f"Connection error for {host}:{port} - {e}")
        return "resource doesn't employ SSL certificate"
    except ssl.SSLError as e:
        logger.error(f"SSL error for {host}:{port} - {e}")
        return "resource doesn't employ SSL certificate"
    except Exception as e:
        logger.error(
            f"Unexpected error retrieving SSL certificate for {host}:{port} - {e}"
        )
        return "resource doesn't employ SSL certificate"
