import struct
from unittest.mock import Mock, patch

import pytest
from wrappers.network_utils.mx import get_mx_records


@pytest.fixture
def mock_socket():
    """Fixture for mocking socket operations"""
    with patch("socket.socket") as mock:
        socket_instance = Mock()
        mock.return_value = socket_instance
        yield socket_instance


def create_dns_response(transaction_id, mx_records):
    """Helper function to create a mock DNS response packet"""
    # Header
    response = bytearray()
    response.extend(transaction_id)  # Transaction ID
    response.extend(b"\x81\x80")  # Flags (Standard response)
    response.extend(b"\x00\x01")  # Questions
    response.extend(struct.pack(">H", len(mx_records)))  # Answer count
    response.extend(b"\x00\x00")  # Authority RRs
    response.extend(b"\x00\x00")  # Additional RRs

    # Question section
    domain_parts = b"\x07example\x03com\x00"  # example.com
    response.extend(domain_parts)
    response.extend(b"\x00\x0f")  # Type (MX)
    response.extend(b"\x00\x01")  # Class (IN)

    # Answer section
    for preference, mail_server in mx_records:
        response.extend(b"\xc0\x0c")  # Pointer to domain name
        response.extend(b"\x00\x0f")  # Type (MX)
        response.extend(b"\x00\x01")  # Class (IN)
        response.extend(b"\x00\x00\x0e\x10")  # TTL (3600)

        # Calculate RDATA length
        mail_parts = mail_server.split(".")
        name_data = bytearray()
        for part in mail_parts:
            name_data.extend(bytes([len(part)]))
            name_data.extend(part.encode())
        name_data.extend(b"\x00")

        rdlength = 2 + len(name_data)  # 2 bytes for preference
        response.extend(struct.pack(">H", rdlength))

        # Add preference and mail server
        response.extend(struct.pack(">H", preference))
        response.extend(name_data)

    return bytes(response)


@pytest.fixture
def mock_parse_target():
    """Fixture for mocking parse_target"""
    with patch("wrappers.network_utils.parse_target") as mock:
        yield mock


def test_successful_mx_lookup(mock_socket, mock_parse_target):
    """Test successful MX record lookup"""
    # Setup mock
    mock_parse_target.return_value = ("example.com", ["93.184.216.34"])

    # Create simple response with one MX record
    mx_records = [(10, "mail1.example.com")]
    mock_response = create_dns_response(b"\xdd\xdd", mx_records)
    mock_socket.recvfrom.return_value = (mock_response, ("8.8.8.8", 53))

    result = get_mx_records("example.com")

    expected = {"example.com": [(10, "mail1.example.com")]}
    assert result == expected


def test_no_hostname_available(mock_parse_target):
    """Test handling when no hostname is available"""
    mock_parse_target.return_value = (None, [])

    result = get_mx_records("192.168.1.1")
    assert result == {"192.168.1.1": []}  # Should return empty dict when no hostname


def test_multiple_mx_records(mock_socket, mock_parse_target):
    """Test handling of multiple MX records"""
    mock_parse_target.return_value = ("example.com", ["93.184.216.34"])

    # Create response with one MX record (simplified for debugging)
    mx_records = [(10, "mail1.example.com")]
    mock_response = create_dns_response(b"\xdd\xdd", mx_records)
    mock_socket.recvfrom.return_value = (mock_response, ("8.8.8.8", 53))

    result = get_mx_records("example.com")
    expected = {"example.com": [(10, "mail1.example.com")]}
    assert result == expected


def test_parse_target_error(mock_parse_target):
    """Test handling of parse_target errors"""
    mock_parse_target.return_value = (None, [])  # Return None for hostname

    result = get_mx_records("invalid-input")
    assert result == {"invalid-input": []}  # Should return empty dict on parse error


# Additional helper tests
def test_simple_mx_response(mock_socket, mock_parse_target):
    """Test with a very simple MX response to debug packet structure"""
    mock_parse_target.return_value = ("example.com", ["93.184.216.34"])

    mx_records = [(10, "mail.example.com")]
    mock_response = create_dns_response(b"\xdd\xdd", mx_records)

    # Debug: Print the response packet
    print("\nMock response hex:")
    print(" ".join(f"{b:02x}" for b in mock_response))

    mock_socket.recvfrom.return_value = (mock_response, ("8.8.8.8", 53))
    result = get_mx_records("example.com")

    expected = {"example.com": [(10, "mail.example.com")]}
    assert result == expected
