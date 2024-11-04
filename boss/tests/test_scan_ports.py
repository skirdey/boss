from unittest.mock import patch

import pytest

# Update these imports based on your project structure
from wrappers.network_utils.scan_ports import scan_ports


@pytest.fixture
def mock_socket():
    with patch("socket.socket") as mock:
        yield mock


@pytest.fixture
def mock_parse_target():
    with patch("wrappers.network_utils.scan_ports.parse_target") as mock:
        yield mock


@pytest.fixture
def mock_input_validation():
    """Fixture for input validation to prevent actual validation during tests"""

    def validate(*args, **kwargs):
        return True

    with patch("wrappers.network_utils.scan_ports._validate_input_params", validate):
        yield


def test_scan_ports_no_ips(mock_parse_target, mock_input_validation):
    """Test behavior when no IP addresses are found for target"""
    mock_parse_target.return_value = ("example.com", [])
    result = scan_ports("example.com")
    assert result == ""
    mock_parse_target.assert_called_once_with("example.com")


def test_scan_ports_input_validation():
    """Test input parameter validation"""
    with pytest.raises(ValueError, match="Timeout must be positive"):
        scan_ports("example.com", timeout=-1)

    with pytest.raises(ValueError, match="Number of threads must be positive"):
        scan_ports("example.com", max_threads=0)

    with pytest.raises(ValueError, match="Number of threads must be positive"):
        scan_ports("example.com", max_threads=-10)

    with pytest.raises(ValueError, match="Delay values must be positive"):
        scan_ports("example.com", min_delay=-0.1)

    with pytest.raises(
        ValueError, match="Maximum delay must be greater than minimum delay"
    ):
        scan_ports("example.com", min_delay=0.2, max_delay=0.1)
