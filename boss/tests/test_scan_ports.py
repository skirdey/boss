from unittest.mock import patch

import pytest

# Update these imports based on your project structure


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
