import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import Response

from boss.wrappers.network_utils.sql_injection import (
    ScanResult,
    ScanTarget,
    SecurityScanner,
    TestParameter,
)


@pytest.fixture
def mock_anthropic_client():
    client = MagicMock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def scan_target():
    return ScanTarget(url="https://example.com", paths=["/test", "/admin"])


@pytest.fixture
def test_parameters():
    return [
        TestParameter(
            name="id",
            value="1 OR 1=1",
            injection_type="error_based",
            expected_pattern="error",
        ),
        TestParameter(
            name="name",
            value="admin' --",
            injection_type="boolean_based",
            expected_pattern="admin",
        ),
    ]


@pytest.fixture
def mock_scan_result():
    return ScanResult(
        target="https://example.com",
        path="/test",
        timestamp=datetime.now().isoformat(),
        findings=[],
        parameters_tested=[],
        errors=[],
        success=True,
    )


@pytest.mark.asyncio
async def test_security_scanner_init(scan_target, mock_anthropic_client):
    scanner = SecurityScanner(mock_anthropic_client, scan_target)
    assert scanner.target == scan_target
    assert scanner.concurrency == 3
    assert scanner.timeout == 30


@pytest.mark.asyncio
async def test_security_scanner_invalid_concurrency(mock_anthropic_client, scan_target):
    with pytest.raises(ValueError):
        SecurityScanner(mock_anthropic_client, scan_target, concurrency=0)


@pytest.mark.asyncio
async def test_security_scanner_invalid_timeout(mock_anthropic_client, scan_target):
    with pytest.raises(ValueError):
        SecurityScanner(mock_anthropic_client, scan_target, timeout=0)


@pytest.mark.asyncio
async def test_security_scanner_scan(
    mock_anthropic_client, scan_target, test_parameters, mock_scan_result
):
    mock_anthropic_client.messages.create.return_value = MagicMock(
        content=[
            MagicMock(
                text=json.dumps(
                    {"parameters": asdict(param) for param in test_parameters}
                )
            )
        ]
    )
    mock_http_client = mock_anthropic_client.http_client = MagicMock()
    mock_http_client.get.return_value = Response(200, text="No error")
    mock_http_client.post.return_value = Response(200, text="No error")

    async with SecurityScanner(mock_anthropic_client, scan_target) as scanner:
        scanner.http_client = mock_http_client
        results, errors = await scanner.scan()

    assert len(results) == 2
    assert not errors
    for result in results:
        assert isinstance(result, ScanResult)
        assert result.success


@pytest.mark.asyncio
async def test_security_scanner_scan_timeout(
    mock_anthropic_client, scan_target, test_parameters, mock_scan_result
):
    # Mock the response from the LLM
    mock_anthropic_client.messages.create.return_value = MagicMock(
        content=[
            MagicMock(
                text=json.dumps(
                    {"parameters": [asdict(param) for param in test_parameters]}
                )
            )
        ]
    )
    # Create the mock HTTP client
    mock_http_client = MagicMock()
    mock_http_client.get.side_effect = asyncio.TimeoutError()
    mock_http_client.post.side_effect = asyncio.TimeoutError()

    async with SecurityScanner(
        mock_anthropic_client, scan_target, timeout=1
    ) as scanner:
        scanner.http_client = mock_http_client  # Assign the mock HTTP client
        results, errors = await scanner.scan()

    assert len(results) == 2
    assert not errors
    for result in results:
        assert isinstance(result, ScanResult)
        assert not result.success
        assert result.errors == [f"Scan timed out after {scanner.timeout} seconds"]


@pytest.mark.asyncio
async def test_security_scanner_scan_error(
    mock_anthropic_client, scan_target, test_parameters, mock_scan_result
):
    # Mock the response from the LLM
    mock_anthropic_client.messages.create.return_value = MagicMock(
        content=[
            MagicMock(
                text=json.dumps(
                    {"parameters": [asdict(param) for param in test_parameters]}
                )
            )
        ]
    )
    # Create the mock HTTP client
    mock_http_client = MagicMock()
    mock_http_client.get.return_value = Response(500, text="Internal Server Error")
    mock_http_client.post.return_value = Response(500, text="Internal Server Error")

    async with SecurityScanner(mock_anthropic_client, scan_target) as scanner:
        scanner.http_client = mock_http_client  # Assign the mock HTTP client
        results, errors = await scanner.scan()

    assert len(results) == 2
    assert not errors
    for result in results:
        assert isinstance(result, ScanResult)
        assert result.success
        assert len(result.findings) > 0
        for finding in result.findings:
            assert finding["details"].get("get_vulnerable") or finding["details"].get(
                "post_vulnerable"
            )


@pytest.mark.asyncio
async def test_security_scanner_scan_with_no_paths(mock_anthropic_client):
    scan_target = ScanTarget(url="https://example.com", paths=[])

    with pytest.raises(ValueError):
        scan_target = ScanTarget(url="https://example.com", paths=[])
        scanner = SecurityScanner(mock_anthropic_client, scan_target)


@pytest.mark.asyncio
async def test_security_scanner_scan_with_invalid_url(mock_anthropic_client):
    scan_target = ScanTarget(url="", paths=["/test"])

    with pytest.raises(ValueError):
        scan_target = ScanTarget(url="", paths=["/test"])
        scanner = SecurityScanner(mock_anthropic_client, scan_target)


@pytest.mark.asyncio
async def test_security_scanner_scan_with_invalid_parameter_response(
    mock_anthropic_client, scan_target
):
    mock_anthropic_client.messages.create.return_value = MagicMock(
        content=[
            MagicMock(
                text=json.dumps(
                    {"parameters": [asdict(param) for param in test_parameters]}
                )
            )
        ]
    )

    mock_http_client = mock_anthropic_client.http_client = MagicMock()
    mock_http_client.get.return_value = Response(200, text="No error")
    mock_http_client.post.return_value = Response(200, text="No error")

    async with SecurityScanner(mock_anthropic_client, scan_target) as scanner:
        scanner.http_client = mock_http_client  # Assign the mock HTTP client
        results, errors = await scanner.scan()

    assert len(results) == 2
    assert not errors
    for result in results:
        assert isinstance(result, ScanResult)
        assert result.success
        assert not result.findings


@pytest.mark.asyncio
async def test_security_scanner_scan_with_invalid_parameter_format(
    mock_anthropic_client, scan_target
):
    mock_anthropic_client.messages.create.return_value = MagicMock(
        content=[
            MagicMock(
                text=json.dumps({"parameters": [{"name": "id", "value": "1 OR 1=1"}]})
            )
        ]
    )
    mock_http_client = mock_anthropic_client.http_client = MagicMock()
    mock_http_client.get.return_value = Response(200, text="No error")
    mock_http_client.post.return_value = Response(200, text="No error")

    async with SecurityScanner(mock_anthropic_client, scan_target) as scanner:
        scanner.http_client = mock_http_client
        results, errors = await scanner.scan()

    assert len(results) == 2
    assert not errors
    for result in results:
        assert isinstance(result, ScanResult)
        assert result.success
        assert not result.findings
