import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from boss.wrappers.network_utils.sql_injection import (
    ScanResult,
    ScanTarget,
    SecurityScanner,
    TestParameter,
    anthropic,
)


@pytest.fixture
def mock_anthropic_client():
    client = MagicMock(spec=anthropic.Anthropic)
    messages = MagicMock()
    messages.create = AsyncMock()
    client.messages = messages
    return client


@pytest.fixture
def mock_httpx_client():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock()
    client.post = AsyncMock()
    return client


@pytest.fixture
def scan_target():
    return ScanTarget(
        url="https://test-target.com",
        paths=["/login", "/api/users"],
        description="Test target",
    )


@pytest.fixture
async def scanner(mock_anthropic_client, scan_target):
    async with SecurityScanner(
        client=mock_anthropic_client, scan_target=scan_target, concurrency=2, timeout=5
    ) as scanner:
        yield scanner


def create_mock_response(status_code=200, text="", elapsed=timedelta(seconds=0.1)):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = text
    response.elapsed = elapsed
    response.headers = {"content-type": "application/json"}
    return response


@pytest.mark.asyncio
async def test_scanner_initialization(mock_anthropic_client, scan_target):
    scanner = SecurityScanner(client=mock_anthropic_client, scan_target=scan_target)
    assert scanner.client == mock_anthropic_client
    assert scanner.target == scan_target
    assert scanner.concurrency == 3  # default value
    assert scanner.timeout == 30  # default value


@pytest.mark.asyncio
async def test_scanner_invalid_concurrency():
    with pytest.raises(ValueError, match="Concurrency must be positive"):
        SecurityScanner(
            client=MagicMock(),
            scan_target=ScanTarget(url="https://test.com", paths=["/"]),
            concurrency=0,
        )


@pytest.mark.skip
@pytest.mark.asyncio
async def test_parameter_generation(mock_anthropic_client, scan_target):
    mock_response = MagicMock()
    mock_response.content[
        0
    ].text = """{"name": "id", "value": "1' OR '1'='1", "injection_type": "boolean_based", "expected_pattern": "error"}"""
    mock_anthropic_client.messages.create.return_value = mock_response

    async with SecurityScanner(
        client=mock_anthropic_client, scan_target=scan_target, concurrency=2, timeout=5
    ) as scanner:
        params = await scanner._generate_test_parameters("/test")

        print(f"*********** {params} *********")
        assert len(params) == 1
        assert isinstance(params, TestParameter)
        assert params.name == "id"
        assert params.injection_type == "boolean_based"


@pytest.mark.skip
@pytest.mark.asyncio
async def test_scan_path(mock_anthropic_client, scan_target):
    # Mock parameter generation
    mock_response = MagicMock()
    mock_response.content = {
        "parameters": [
            {
                "name": "id",
                "value": "1' OR '1'='1",
                "injection_type": "boolean_based",
                "expected_pattern": "error",
            }
        ]
    }
    mock_anthropic_client.messages.create.return_value = mock_response

    # Mock HTTP responses
    mock_get_response = create_mock_response(status_code=500, text="SQL error in query")
    mock_post_response = create_mock_response(status_code=200, text="normal response")

    async with SecurityScanner(
        client=mock_anthropic_client, scan_target=scan_target, concurrency=2, timeout=5
    ) as scanner:
        with patch.object(
            scanner.http_client, "get", return_value=mock_get_response
        ), patch.object(scanner.http_client, "post", return_value=mock_post_response):
            result = await scanner._scan_path("/test")

            assert isinstance(result, ScanResult)
            assert result.target == "https://test-target.com"
            assert result.path == "/test"
            assert len(result.findings) == 1
            assert result.findings["parameter"] == "id"
            assert result.success is True


@pytest.mark.skip
@pytest.mark.asyncio
async def test_scan_with_timeout(mock_anthropic_client, scan_target):
    async def mock_slow_scan(*args, **kwargs):
        await asyncio.sleep(2)
        return None

    async with SecurityScanner(
        client=mock_anthropic_client, scan_target=scan_target, concurrency=2, timeout=1
    ) as scanner:
        with patch.object(scanner, "_scan_path", side_effect=mock_slow_scan):
            results, errors = await scanner.scan()
            assert len(errors) > 0
            assert "timed out" in errors.lower()


@pytest.mark.asyncio
async def test_concurrent_scanning(mock_anthropic_client, scan_target):
    # Mock parameter generation
    mock_response = MagicMock()
    mock_response.content = {
        "parameters": [
            {
                "name": "id",
                "value": "1' OR '1'='1",
                "injection_type": "boolean_based",
                "expected_pattern": "error",
            }
        ]
    }
    mock_anthropic_client.messages.create.return_value = mock_response

    # Mock HTTP responses
    mock_response = create_mock_response(status_code=200, text="normal response")

    async with SecurityScanner(
        client=mock_anthropic_client, scan_target=scan_target, concurrency=2, timeout=5
    ) as scanner:
        with patch.object(
            scanner.http_client, "get", return_value=mock_response
        ), patch.object(scanner.http_client, "post", return_value=mock_response):
            results, errors = await scanner.scan()

            assert len(results) == 2  # Number of paths in scan_target
            assert not errors
            assert all(isinstance(r, ScanResult) for r in results)


@pytest.mark.skip
@pytest.mark.asyncio
async def test_scan_with_http_error(mock_anthropic_client, scan_target):
    # Mock parameter generation
    mock_response = MagicMock()
    mock_response.content = {
        "parameters": [
            {
                "name": "id",
                "value": "1' OR '1'='1",
                "injection_type": "boolean_based",
                "expected_pattern": "error",
            }
        ]
    }
    mock_anthropic_client.messages.create.return_value = mock_response

    # Mock HTTP client to raise an exception
    async def mock_error(*args, **kwargs):
        raise httpx.RequestError("Connection failed")

    async with SecurityScanner(
        client=mock_anthropic_client, scan_target=scan_target, concurrency=2, timeout=5
    ) as scanner:
        with patch.object(
            scanner.http_client, "get", side_effect=mock_error
        ), patch.object(scanner.http_client, "post", side_effect=mock_error):
            results, errors = await scanner.scan()
            assert len(errors) > 0
            assert any("failed" in error.lower() for error in errors)


@pytest.mark.asyncio
async def test_analyze_response(mock_anthropic_client, scan_target):
    async with SecurityScanner(
        client=mock_anthropic_client, scan_target=scan_target, concurrency=2, timeout=5
    ) as scanner:
        # Test positive case
        response_with_error = create_mock_response(
            status_code=500, text="SQL syntax error"
        )
        assert scanner._analyze_response(response_with_error, "SQL") is True

        # Test negative case
        normal_response = create_mock_response(status_code=200, text="Success")
        assert scanner._analyze_response(normal_response, "error") is False


@pytest.mark.asyncio
async def test_extract_response_details(mock_anthropic_client, scan_target):
    mock_response = create_mock_response(
        status_code=200, text="Test response body", elapsed=timedelta(seconds=0.5)
    )

    async with SecurityScanner(
        client=mock_anthropic_client, scan_target=scan_target, concurrency=2, timeout=5
    ) as scanner:
        details = scanner._extract_response_details(mock_response)
        assert details["status_code"] == 200
        assert "Test response body" in details["body_preview"]
        assert details["response_time"] == 0.5
        assert "content-type" in details["headers"]
