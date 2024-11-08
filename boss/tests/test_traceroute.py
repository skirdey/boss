import socket
from unittest.mock import MagicMock, patch

import pytest

from boss.wrappers.network_utils.tracert import traceroute


@pytest.fixture
def mock_sockets():
    with patch("socket.socket") as mock_socket:
        recv_socket = MagicMock()
        send_socket = MagicMock()

        # Configure socket.socket to return our mock sockets
        mock_socket.side_effect = [
            recv_socket,
            send_socket,
        ] * 30  # Support multiple hops

        # Configure default mock behaviors
        recv_socket.recvfrom.return_value = (b"", ("192.168.1.1", 0))

        yield {
            "socket": mock_socket,
            "recv_socket": recv_socket,
            "send_socket": send_socket,
        }


@pytest.fixture
def mock_time():
    with patch("time.time") as mock:
        mock.side_effect = [0, 0.1]  # start_time and end_time for RTT calculation
        yield mock


@pytest.fixture
def mock_dns():
    with patch("socket.gethostbyname") as mock_gethostbyname, patch(
        "socket.gethostbyaddr"
    ) as mock_gethostbyaddr:
        mock_gethostbyname.return_value = "192.168.1.1"
        mock_gethostbyaddr.return_value = ("host.example.com", [], ["192.168.1.1"])
        yield {"gethostbyname": mock_gethostbyname, "gethostbyaddr": mock_gethostbyaddr}


def test_successful_traceroute(mock_sockets, mock_dns, mock_time):
    mock_time.side_effect = [0, 0.1]  # 100ms RTT
    mock_sockets["recv_socket"].recvfrom.return_value = (b"", ("192.168.1.1", 0))

    result = traceroute("example.com", max_hops=1)

    assert len(result) == 1
    assert result[0][0] == 1  # hop number
    assert result[0][1] == "192.168.1.1"  # IP address
    assert result[0][2] == "100.00 ms"  # RTT string


def test_timeout_response(mock_sockets, mock_dns):
    mock_sockets["recv_socket"].recvfrom.side_effect = socket.timeout

    result = traceroute("example.com", max_hops=1)

    assert len(result) == 1
    assert result[0] == (1, "*", "*")


def test_multiple_hops(mock_sockets, mock_dns, mock_time):
    # Set up time responses for each hop (start and end time for each hop)
    mock_time.side_effect = [
        0,
        0.1,  # First hop
        0.2,
        0.3,  # Second hop
        0.4,
        0.5,
    ]  # Third hop

    # Make sure the destination is not reached until the last hop
    mock_dns["gethostbyname"].return_value = "192.168.1.3"

    # Set up socket responses for each hop
    mock_sockets["recv_socket"].recvfrom.side_effect = [
        (b"", ("192.168.1.1", 0)),
        (b"", ("192.168.1.2", 0)),
        (b"", ("192.168.1.3", 0)),
    ]

    result = traceroute("example.com", max_hops=3)

    assert len(result) == 3
    assert result[0] == (1, "192.168.1.1", "100.00 ms")
    assert result[1] == (2, "192.168.1.2", "100.00 ms")
    assert result[2] == (3, "192.168.1.3", "100.00 ms")


def test_destination_reached_early(mock_sockets, mock_dns, mock_time):
    mock_dns["gethostbyname"].return_value = "192.168.1.2"
    mock_time.side_effect = [0, 0.1, 0.2, 0.3]

    mock_sockets["recv_socket"].recvfrom.side_effect = [
        (b"", ("192.168.1.1", 0)),
        (b"", ("192.168.1.2", 0)),
    ]

    result = traceroute("example.com", max_hops=5)

    assert len(result) == 2
    assert result[-1][1] == "192.168.1.2"


def test_hostname_resolution_failure(mock_sockets, mock_dns, mock_time):
    mock_dns["gethostbyaddr"].side_effect = socket.error
    mock_time.side_effect = [0, 0.1]

    result = traceroute("example.com", max_hops=1)

    assert len(result) == 1
    assert result[0][1] == "192.168.1.1"


@patch("socket.gethostbyname")
def test_invalid_destination(mock_gethostbyname):
    mock_gethostbyname.side_effect = socket.gaierror()

    with pytest.raises(socket.gaierror):
        traceroute("invalid.example.com")


def test_custom_timeout(mock_sockets, mock_dns):
    custom_timeout = 1.5
    result = traceroute("example.com", max_hops=1, timeout=custom_timeout)

    mock_sockets["recv_socket"].settimeout.assert_called_with(custom_timeout)


def test_custom_port(mock_sockets, mock_dns):
    custom_port = 44444
    result = traceroute("example.com", max_hops=1, port=custom_port)

    mock_sockets["recv_socket"].bind.assert_called_with(("", custom_port))
    mock_sockets["send_socket"].sendto.assert_called_with(
        b"", ("example.com", custom_port)
    )
