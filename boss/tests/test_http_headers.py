import http.client
import socket
from unittest.mock import Mock, patch

import pytest

from boss.wrappers.network_utils.http_headers import fetch_http_headers


@pytest.fixture
def mock_http_response():
    """Fixture for mocking HTTP response"""
    mock_response = Mock()
    mock_response.getheaders.return_value = [
        ("Server", "nginx"),
        ("Content-Type", "text/html"),
        ("Connection", "keep-alive"),
    ]
    return mock_response


@pytest.fixture
def mock_https_conn(mock_http_response):
    """Fixture for mocking HTTPS connection"""
    with patch("http.client.HTTPSConnection") as mock_conn:
        conn_instance = Mock()
        conn_instance.getresponse.return_value = mock_http_response
        mock_conn.return_value = conn_instance
        yield conn_instance


@pytest.fixture
def mock_http_conn(mock_http_response):
    """Fixture for mocking HTTP connection"""
    with patch("http.client.HTTPConnection") as mock_conn:
        conn_instance = Mock()
        conn_instance.getresponse.return_value = mock_http_response
        mock_conn.return_value = conn_instance
        yield conn_instance


def test_fetch_https_headers(mock_https_conn, mock_http_response):
    """Test fetching headers from HTTPS URL"""
    url = "https://example.com"
    headers = fetch_http_headers(url)

    # Verify the connection was made with correct parameters
    mock_https_conn.request.assert_called_once_with("HEAD", "/")
    mock_https_conn.getresponse.assert_called_once()
    mock_https_conn.close.assert_called_once()

    # Verify headers are returned correctly
    assert headers == {
        "Server": "nginx",
        "Content-Type": "text/html",
        "Connection": "keep-alive",
    }


def test_fetch_http_headers(mock_http_conn, mock_http_response):
    """Test fetching headers from HTTP URL"""
    url = "http://example.com"
    headers = fetch_http_headers(url)

    # Verify the connection was made with correct parameters
    mock_http_conn.request.assert_called_once_with("HEAD", "/")
    mock_http_conn.getresponse.assert_called_once()
    mock_http_conn.close.assert_called_once()

    # Verify headers are returned correctly
    assert headers == {
        "Server": "nginx",
        "Content-Type": "text/html",
        "Connection": "keep-alive",
    }


def test_fetch_headers_with_path():
    """Test fetching headers from URL with path"""
    with patch("http.client.HTTPSConnection") as mock_conn:
        conn_instance = Mock()
        mock_response = Mock()
        mock_response.getheaders.return_value = [("Server", "nginx")]
        conn_instance.getresponse.return_value = mock_response
        mock_conn.return_value = conn_instance

        url = "https://example.com/path/to/resource"
        headers = fetch_http_headers(url)

        # Verify the correct path was requested
        conn_instance.request.assert_called_once_with("HEAD", "/path/to/resource")
        assert headers == {"Server": "nginx"}


def test_unsupported_scheme():
    """Test handling of unsupported URL scheme"""
    url = "ftp://example.com"
    headers = fetch_http_headers(url)
    assert headers == {}


def test_connection_error():
    """Test handling of connection error"""
    with patch("http.client.HTTPSConnection") as mock_conn:
        conn_instance = Mock()
        conn_instance.request.side_effect = ConnectionError("Connection failed")
        mock_conn.return_value = conn_instance

        url = "https://example.com"
        headers = fetch_http_headers(url)

        assert headers == {}


def test_empty_response():
    """Test handling of empty response headers"""
    with patch("http.client.HTTPSConnection") as mock_conn:
        conn_instance = Mock()
        mock_response = Mock()
        mock_response.getheaders.return_value = []
        conn_instance.getresponse.return_value = mock_response
        mock_conn.return_value = conn_instance

        url = "https://example.com"
        headers = fetch_http_headers(url)

        assert headers == {}


def test_malformed_url():
    """Test handling of malformed URL"""
    url = "not_a_valid_url"
    headers = fetch_http_headers(url)
    assert headers == {}


def test_timeout_handling():
    """Test handling of connection timeout"""
    with patch("http.client.HTTPSConnection") as mock_conn:
        conn_instance = Mock()
        conn_instance.request.side_effect = socket.timeout("Connection timed out")
        mock_conn.return_value = conn_instance

        url = "https://example.com"
        headers = fetch_http_headers(url)

        assert headers == {}


def test_response_error():
    """Test handling of response error"""
    with patch("http.client.HTTPSConnection") as mock_conn:
        conn_instance = Mock()
        conn_instance.getresponse.side_effect = http.client.HTTPException()
        mock_conn.return_value = conn_instance

        url = "https://example.com"
        headers = fetch_http_headers(url)

        assert headers == {}
