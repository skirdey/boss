import socket
import struct
from unittest.mock import Mock, patch

import pytest

from boss.wrappers.network_utils.cname import get_cname_records


@pytest.fixture
def mock_socket():
    with patch("socket.socket") as mock:
        # Create a mock socket instance
        socket_instance = Mock()
        mock.return_value = socket_instance
        yield socket_instance


def create_dns_response(transaction_id, answers):
    """Helper function to create a mock DNS response packet"""
    # Header
    response = transaction_id  # Transaction ID
    response += b"\x81\x80"  # Flags (Standard response)
    response += b"\x00\x01"  # Questions
    response += struct.pack(">H", len(answers))  # Answer count
    response += b"\x00\x00"  # Authority RRs
    response += b"\x00\x00"  # Additional RRs

    # Question section (example.com)
    domain_parts = [b"\x07example", b"\x03com", b"\x00"]
    question = b"".join(domain_parts)
    response += question  # Domain name
    response += b"\x00\x05"  # Type (CNAME)
    response += b"\x00\x01"  # Class (IN)

    # Answer section
    offset = len(response)  # Keep track of the offset for compression pointers

    for cname in answers:
        response += b"\xc0\x0c"  # Pointer to domain name in question section
        response += b"\x00\x05"  # Type (CNAME)
        response += b"\x00\x01"  # Class (IN)
        response += b"\x00\x00\x0e\x10"  # TTL (3600)

        # Construct CNAME with DNS name compression
        cname_parts = cname.split(".")
        rdata = b""
        for part in cname_parts:
            rdata += bytes([len(part)]) + part.encode()
        rdata += b"\x00"

        response += struct.pack(">H", len(rdata))  # RDLENGTH
        response += rdata

    return response


def test_successful_cname_lookup(mock_socket):
    """Test successful CNAME record lookup"""
    cname = "target.example.com"
    mock_response = create_dns_response(b"\xcc\xcc", [cname])
    mock_socket.recvfrom.return_value = (mock_response, ("8.8.8.8", 53))

    result = get_cname_records("example.com")

    assert len(result) == 1
    assert result[0] == cname
    mock_socket.sendto.assert_called_once()


def test_multiple_cname_records(mock_socket):
    """Test handling multiple CNAME records"""
    cnames = ["target1.example.com", "target2.example.com"]
    mock_response = create_dns_response(b"\xcc\xcc", cnames)

    # Debug: Print the hex representation of the mock response
    print("Mock response hex:")
    print(" ".join(f"{b:02x}" for b in mock_response))

    mock_socket.recvfrom.return_value = (mock_response, ("8.8.8.8", 53))
    result = get_cname_records("example.com")

    # Debug: Print the actual result
    print("Result:", result)

    assert len(result) == 2
    assert result == cnames
    mock_socket.sendto.assert_called_once()


def test_timeout_handling(mock_socket):
    """Test handling of socket timeout"""
    mock_socket.recvfrom.side_effect = socket.timeout

    result = get_cname_records("example.com")

    assert result == []
    mock_socket.sendto.assert_called_once()


def test_socket_error_handling(mock_socket):
    """Test handling of general socket errors"""
    mock_socket.sendto.side_effect = socket.error("Mock socket error")

    result = get_cname_records("example.com")

    assert result == []


def test_malformed_response_handling(mock_socket):
    """Test handling of malformed DNS response"""
    mock_socket.recvfrom.return_value = (b"Invalid DNS response", ("8.8.8.8", 53))

    result = get_cname_records("example.com")

    assert result == []
    mock_socket.sendto.assert_called_once()


def test_empty_domain_name():
    """Test handling of empty domain name"""
    result = get_cname_records("")

    assert result == []


def test_no_cname_records(mock_socket):
    """Test handling of response with no CNAME records"""
    mock_response = create_dns_response(b"\xcc\xcc", [])
    mock_socket.recvfrom.return_value = (mock_response, ("8.8.8.8", 53))

    result = get_cname_records("example.com")

    assert result == []
    mock_socket.sendto.assert_called_once()


def test_invalid_dns_server(mock_socket):
    """Test handling of invalid DNS server"""
    mock_socket.sendto.side_effect = socket.gaierror("Mock DNS server error")

    result = get_cname_records("example.com")

    assert result == []
