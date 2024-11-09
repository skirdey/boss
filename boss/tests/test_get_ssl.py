import socket
import ssl
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from boss.wrappers.network_utils.ssl_certificate import get_ssl_certificate


@pytest.fixture
def mock_ssl_certificate():
    """Fixture providing a mock SSL certificate dictionary."""
    return {
        "subject": ((("commonName", "www.example.com"),),),
        "issuer": ((("commonName", "ExampleCA"),),),
        "version": 3,
        "serialNumber": "1234567890",
        "notBefore": "Jan 1 00:00:00 2023 GMT",
        "notAfter": "Dec 31 23:59:59 2023 GMT",
        "subjectAltName": (("DNS", "www.example.com"), ("DNS", "example.com")),
    }


@pytest.fixture
def expected_certificate():
    """Fixture providing the expected processed certificate dictionary."""
    return {
        "subject": {"commonName": "www.example.com"},
        "issuer": {"commonName": "ExampleCA"},
        "version": 3,
        "serialNumber": "1234567890",
        "notBefore": datetime(2023, 1, 1, 0, 0),
        "notAfter": datetime(2023, 12, 31, 23, 59, 59),
        "subjectAltName": (("DNS", "www.example.com"), ("DNS", "example.com")),
    }


def test_successful_certificate_retrieval(mock_ssl_certificate, expected_certificate):
    """Test successful SSL certificate retrieval and processing."""
    with patch("socket.create_connection") as mock_socket:
        with patch("ssl.SSLContext.wrap_socket") as mock_wrap_socket:
            # Configure the mock SSL socket
            mock_ssl_socket = MagicMock()
            mock_ssl_socket.getpeercert.return_value = mock_ssl_certificate
            mock_wrap_socket.return_value.__enter__.return_value = mock_ssl_socket

            # Get the certificate
            result = get_ssl_certificate("example.com")

            # Verify the result matches expected certificate
            assert result == expected_certificate
            assert isinstance(result["notBefore"], datetime)
            assert isinstance(result["notAfter"], datetime)

            # Verify correct method calls
            mock_socket.assert_called_once_with(("example.com", 443), timeout=10)
            mock_wrap_socket.assert_called_once()


def test_custom_port(mock_ssl_certificate):
    """Test certificate retrieval with a custom port."""
    with patch("socket.create_connection") as mock_socket:
        with patch("ssl.SSLContext.wrap_socket") as mock_wrap_socket:
            # Configure the mock SSL socket with the certificate
            mock_ssl_socket = MagicMock()
            mock_ssl_socket.getpeercert.return_value = mock_ssl_certificate
            mock_wrap_socket.return_value.__enter__.return_value = mock_ssl_socket

            get_ssl_certificate("example.com", port=8443)
            mock_socket.assert_called_once_with(("example.com", 8443), timeout=10)


def test_connection_timeout():
    """Test handling of connection timeout."""
    with patch(
        "socket.create_connection", side_effect=socket.timeout("Connection timed out")
    ):
        result = get_ssl_certificate("example.com")
        assert result == "resource doesn't employ SSL certificate"


def test_ssl_error():
    """Test handling of SSL-related errors."""
    with patch("socket.create_connection") as mock_socket:
        with patch("ssl.SSLContext.wrap_socket", side_effect=ssl.SSLError("SSL error")):
            result = get_ssl_certificate("example.com")
            assert result == "resource doesn't employ SSL certificate"


def test_invalid_certificate_format():
    """Test handling of invalid certificate format."""
    invalid_cert = {
        "subject": ((("commonName", "www.example.com"),),),
        "issuer": ((("commonName", "ExampleCA"),),),
        "version": 3,
        "serialNumber": "1234567890",
        "notBefore": "Invalid Date Format",  # This will trigger the exception
        "notAfter": "Dec 31 23:59:59 2023 GMT",
        "subjectAltName": (("DNS", "www.example.com"), ("DNS", "example.com")),
    }

    with patch("socket.create_connection") as mock_socket:
        with patch("ssl.SSLContext.wrap_socket") as mock_wrap_socket:
            mock_ssl_socket = MagicMock()
            mock_ssl_socket.getpeercert.return_value = invalid_cert
            mock_wrap_socket.return_value.__enter__.return_value = mock_ssl_socket

            result = get_ssl_certificate("example.com")
            assert result == "resource doesn't employ SSL certificate"


def test_missing_certificate_fields():
    """Test handling of missing certificate fields."""
    incomplete_cert = {
        "subject": ((("commonName", "www.example.com"),),),
        "issuer": ((("commonName", "ExampleCA"),),),
        # Intentionally missing other fields
    }

    with patch("socket.create_connection") as mock_socket:
        with patch("ssl.SSLContext.wrap_socket") as mock_wrap_socket:
            mock_ssl_socket = MagicMock()
            mock_ssl_socket.getpeercert.return_value = incomplete_cert
            mock_wrap_socket.return_value.__enter__.return_value = mock_ssl_socket

            result = get_ssl_certificate("example.com")
            assert (
                result == "resource doesn't employ SSL certificate"
            )  # Should return empty dict on error
