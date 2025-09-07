import pytest
import asyncio
from typing import Any, Dict, Optional, Type
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aiohttp import ClientError, ClientResponse
import pytest_asyncio

from src.common.http_client import AsyncHttpClient


class MockResponse:
    """Mock implementation of aiohttp.ClientResponse for testing"""

    def __init__(
        self,
        status: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        raise_on_json: bool = False,
    ):
        self.status = status
        self._json_data = json_data or {}
        self._raise_on_json = raise_on_json
        self.closed = False

    async def json(self) -> Dict[str, Any]:
        if self._raise_on_json:
            raise aiohttp.ContentTypeError(None, None)
        return self._json_data

    async def close(self) -> None:
        self.closed = True

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@pytest.fixture
def mock_session() -> AsyncMock:
    """Fixture for a mocked aiohttp.ClientSession"""
    session = AsyncMock(spec=aiohttp.ClientSession)
    return session


@pytest_asyncio.fixture
async def http_client() -> AsyncHttpClient:
    """Fixture for an AsyncHttpClient with mocked session"""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session_instance = AsyncMock()
        mock_session_class.return_value = mock_session_instance
        client = AsyncHttpClient(retries=3, delay=0)  # Use delay=0 to speed up tests
        await client.__aenter__()
        yield client
        await client.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_client_init() -> None:
    """Test client initialization with default and custom parameters"""
    client1 = AsyncHttpClient()
    assert client1.retries == 3
    assert client1.delay == 2

    client2 = AsyncHttpClient(retries=5, delay=1)
    assert client2.retries == 5
    assert client2.delay == 1


@pytest.mark.asyncio
async def test_context_manager() -> None:
    """Test proper session management with context manager"""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session_instance = AsyncMock()
        mock_session_class.return_value = mock_session_instance

        async with AsyncHttpClient() as client:
            assert client._session is not None
            mock_session_class.assert_called_once()

        # Session should be closed and reset after exiting context
        mock_session_instance.close.assert_called_once()
        assert client._session is None


@pytest.mark.asyncio
async def test_fetch_data_success(http_client: AsyncHttpClient) -> None:
    """Test successful data fetching"""
    mock_response = MockResponse(status=200, json_data={"test": "data"})
    http_client._session.get.return_value = mock_response

    response = await http_client.fetch_data("https://api.example.com/data")

    assert response is mock_response
    http_client._session.get.assert_called_once_with(
        "https://api.example.com/data", params=None
    )


@pytest.mark.asyncio
async def test_fetch_data_failure(http_client: AsyncHttpClient) -> None:
    """Test retry behavior on failure"""
    mock_error_response = MockResponse(status=500)
    http_client._session.get.return_value = mock_error_response

    with patch("builtins.print") as mock_print:
        response = await http_client.fetch_data("https://api.example.com/data")

    assert response is None
    assert http_client._session.get.call_count == 3  # All retries used
    assert mock_error_response.closed  # Response should be closed
    assert "failed with status 500" in mock_print.call_args_list[0][0][0]


@pytest.mark.asyncio
async def test_fetch_data_client_error(http_client: AsyncHttpClient) -> None:
    """Test handling of client errors"""
    http_client._session.get.side_effect = aiohttp.ClientError("Connection error")

    with patch("builtins.print") as mock_print:
        response = await http_client.fetch_data("https://api.example.com/data")

    assert response is None
    assert http_client._session.get.call_count == 3  # All retries used
    assert "failed with error" in mock_print.call_args_list[0][0][0]


@pytest.mark.asyncio
async def test_fetch_data_retry_then_success(http_client: AsyncHttpClient) -> None:
    """Test retry mechanism that eventually succeeds"""
    error_response = MockResponse(status=500)
    success_response = MockResponse(status=200, json_data={"test": "data"})

    # First call fails, second call succeeds
    http_client._session.get.side_effect = [error_response, success_response]

    with patch("builtins.print") as mock_print:
        response = await http_client.fetch_data("https://api.example.com/data")

    assert response is success_response
    assert http_client._session.get.call_count == 2
    assert error_response.closed  # First response should be closed
    assert not success_response.closed  # Success response shouldn't be closed


@pytest.mark.asyncio
async def test_ensure_session() -> None:
    """Test session creation when none exists"""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session_instance = AsyncMock()
        mock_session_class.return_value = mock_session_instance

        client = AsyncHttpClient()
        assert client._session is None

        session = await client._ensure_session()
        assert session is mock_session_instance
        assert client._session is mock_session_instance

        # Second call should use existing session
        mock_session_class.reset_mock()
        session2 = await client._ensure_session()
        assert session2 is mock_session_instance
        mock_session_class.assert_not_called()


@pytest.mark.asyncio
async def test_handle_response_with_response() -> None:
    """Test handle_response context manager with a response"""
    client = AsyncHttpClient()
    mock_response = MockResponse(status=200)

    async with client.handle_response(mock_response) as response:
        assert response is mock_response
        assert not mock_response.closed

    assert mock_response.closed  # Should be closed after exiting context


@pytest.mark.asyncio
async def test_handle_response_with_none() -> None:
    """Test handle_response context manager with None"""
    client = AsyncHttpClient()

    async with client.handle_response(None) as response:
        assert response is None


@pytest.mark.asyncio
async def test_fetch_json_success(http_client: AsyncHttpClient) -> None:
    """Test successful JSON fetching"""
    mock_response = MockResponse(status=200, json_data={"test": "data"})
    http_client._session.get.return_value = mock_response

    data = await http_client.fetch_json("https://api.example.com/data")

    assert data == {"test": "data"}
    http_client._session.get.assert_called_once_with(
        "https://api.example.com/data", params=None
    )
    assert mock_response.closed  # Response should be closed after use


@pytest.mark.asyncio
async def test_fetch_json_failure(http_client: AsyncHttpClient) -> None:
    """Test JSON fetching with failed request"""
    http_client._session.get.return_value = MockResponse(status=500)

    data = await http_client.fetch_json("https://api.example.com/data")

    assert data is None
    assert http_client._session.get.call_count == 3  # All retries used


@pytest.mark.asyncio
async def test_fetch_json_with_params(http_client: AsyncHttpClient) -> None:
    """Test JSON fetching with query parameters"""
    mock_response = MockResponse(status=200, json_data={"test": "data"})
    http_client._session.get.return_value = mock_response

    params = {"page": 1, "limit": 10}
    data = await http_client.fetch_json("https://api.example.com/data", params=params)

    assert data == {"test": "data"}
    http_client._session.get.assert_called_once_with(
        "https://api.example.com/data", params=params
    )


@pytest.mark.asyncio
async def test_fetch_json_invalid_json(http_client: AsyncHttpClient) -> None:
    """Test handling of invalid JSON response"""
    mock_response = MockResponse(status=200, raise_on_json=True)
    http_client._session.get.return_value = mock_response

    with patch("builtins.print"):
        with pytest.raises(aiohttp.ContentTypeError):
            await http_client.fetch_json("https://api.example.com/data")

    assert mock_response.closed  # Response should be closed even on error


@pytest.mark.asyncio
async def test_concurrent_requests(http_client: AsyncHttpClient) -> None:
    """Test handling multiple concurrent requests"""
    response1 = MockResponse(status=200, json_data={"id": 1})
    response2 = MockResponse(status=200, json_data={"id": 2})
    response3 = MockResponse(status=200, json_data={"id": 3})

    http_client._session.get.side_effect = [response1, response2, response3]

    results = await asyncio.gather(
        http_client.fetch_json("https://api.example.com/1"),
        http_client.fetch_json("https://api.example.com/2"),
        http_client.fetch_json("https://api.example.com/3"),
    )

    assert results == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert http_client._session.get.call_count == 3
    assert response1.closed
    assert response2.closed
    assert response3.closed
