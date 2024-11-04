import socket
from unittest.mock import MagicMock, patch

import pytest
from wrappers.network_utils.whois import whois_lookup


def test_successful_whois_lookup():
    """Test a successful WHOIS lookup for a .com domain"""
    mock_response = (
        b"Domain Name: EXAMPLE.COM\nRegistry Domain ID: 2336799_DOMAIN_COM-VRSN"
    )

    # Create a mock socket object
    mock_socket = MagicMock()
    mock_socket.recv.side_effect = [mock_response, b""]

    with patch("socket.socket") as mock_socket_creator:
        mock_socket_creator.return_value = mock_socket

        result = whois_lookup("example.com")

        # Verify the correct server was used
        mock_socket_creator.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_socket.connect.assert_called_once_with(("whois.verisign-grs.com", 43))
        mock_socket.sendall.assert_called_once_with(b"example.com\r\n")

        assert "EXAMPLE.COM" in result
        assert "Registry Domain ID" in result


def test_io_domain_lookup():
    """Test WHOIS lookup for a .io domain"""
    with patch("socket.socket") as mock_socket:
        mock_instance = MagicMock()
        mock_socket.return_value = mock_instance
        mock_instance.recv.side_effect = [b"Domain Name: EXAMPLE.IO", b""]

        result = whois_lookup("example.io")

        mock_instance.connect.assert_called_once_with(("whois.nic.io", 43))
        assert "EXAMPLE.IO" in result


def test_unknown_tld_lookup():
    """Test WHOIS lookup for an unknown TLD"""
    with patch("socket.socket") as mock_socket:
        mock_instance = MagicMock()
        mock_socket.return_value = mock_instance
        mock_instance.recv.side_effect = [b"Referral Server: WHOIS.EXAMPLE", b""]

        result = whois_lookup("example.xyz")

        mock_instance.connect.assert_called_once_with(("whois.iana.org", 43))
        assert "Referral Server" in result


def test_connection_timeout():
    """Test handling of connection timeout"""
    with patch("socket.socket") as mock_socket:
        mock_instance = MagicMock()
        mock_socket.return_value = mock_instance
        mock_instance.connect.side_effect = socket.timeout("Connection timed out")

        result = whois_lookup("example.com")

        assert "Error performing WHOIS lookup" in result
        assert "Connection timed out" in result


def test_connection_refused():
    """Test handling of connection refused error"""
    with patch("socket.socket") as mock_socket:
        mock_instance = MagicMock()
        mock_socket.return_value = mock_instance
        mock_instance.connect.side_effect = ConnectionRefusedError("Connection refused")

        result = whois_lookup("example.com")

        assert "Error performing WHOIS lookup" in result
        assert "Connection refused" in result


# def test_invalid_domain_format():
#     """Test lookup with invalid domain format"""
#     with patch("socket.socket") as mock_socket:
#         result = whois_lookup("invalid..domain..com")
#         assert "Error performing WHOIS lookup" in result


def test_socket_cleanup():
    """Test that socket is properly closed after use"""
    mock_socket = MagicMock()
    mock_socket.recv.side_effect = [b"Domain Name: EXAMPLE.COM", b""]

    with patch("socket.socket") as mock_socket_creator:
        mock_socket_creator.return_value = mock_socket

        whois_lookup("example.com")

        mock_socket.close.assert_called_once()


@pytest.mark.parametrize(
    "domain,expected_server",
    [
        ("example.com", "whois.verisign-grs.com"),
        ("example.net", "whois.verisign-grs.com"),
        ("example.org", "whois.verisign-grs.com"),
        ("example.io", "whois.nic.io"),
        ("example.xyz", "whois.iana.org"),
    ],
)
def test_server_selection(domain, expected_server):
    """Test correct WHOIS server selection for different TLDs"""
    with patch("socket.socket") as mock_socket:
        mock_instance = MagicMock()
        mock_socket.return_value = mock_instance
        mock_instance.recv.side_effect = [b"Test response", b""]

        whois_lookup(domain)

        mock_instance.connect.assert_called_once_with((expected_server, 43))
