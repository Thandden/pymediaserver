import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio

from src.common.db import AsyncDatabaseSession
from src.common.dto import ImageDownloaderParams
from src.common.http_client import AsyncHttpClient
from src.common.logger import Logger
from src.workers.image_downloader import ImageDownloader


class MockResponse:
    """Mock implementation of aiohttp.ClientResponse for testing"""

    def __init__(
        self,
        status: int = 200,
        data: bytes = b"test_image_data",
        raise_on_read: bool = False,
    ):
        self.status = status
        self._data = data
        self._raise_on_read = raise_on_read
        self.closed = False

    async def read(self) -> bytes:
        if self._raise_on_read:
            raise aiohttp.ClientError("Error reading response")
        return self._data

    async def release(self) -> None:
        self.closed = True

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@pytest_asyncio.fixture  # type: ignore
async def mock_db_session() -> AsyncMock:
    """Fixture for a mocked database session."""
    return AsyncMock(spec=AsyncDatabaseSession)


@pytest_asyncio.fixture  # type: ignore
async def mock_logger() -> MagicMock:
    """Fixture for a mocked logger."""
    return MagicMock(spec=Logger)


@pytest_asyncio.fixture  # type: ignore
async def mock_http_client() -> AsyncMock:
    """Fixture for a mocked HTTP client."""
    mock_client = AsyncMock(spec=AsyncHttpClient)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest_asyncio.fixture  # type: ignore
async def image_downloader(
    mock_db_session: AsyncMock, mock_logger: MagicMock
) -> AsyncGenerator[ImageDownloader, None]:
    """Fixture for an ImageDownloader with mocked dependencies."""
    with patch(
        "src.workers.image_downloader.AsyncHttpClient"
    ) as mock_http_client_class:
        mock_http_client = AsyncMock()
        mock_http_client_class.return_value = mock_http_client
        mock_http_client.__aenter__.return_value = mock_http_client

        downloader = ImageDownloader(db_session=mock_db_session, logger=mock_logger)
        downloader.http_client = mock_http_client

        yield downloader


@pytest.mark.asyncio
async def test_execute_with_valid_params(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
    tmp_path: Path,
) -> None:
    """Test execute method with valid parameters."""
    # Setup
    test_entity_id = uuid.uuid4()
    test_image_url = "/path/to/image.jpg"
    params = ImageDownloaderParams(image_url=test_image_url, entity_id=test_entity_id)

    # Mock the config.IMAGE_DIRECTORY
    with patch("src.workers.image_downloader.config") as mock_config:
        mock_config.IMAGE_DIRECTORY = str(tmp_path)

        # Mock the _download_image method to return success
        with patch.object(
            image_downloader, "_download_image", return_value=True
        ) as mock_download:
            # Execute
            result = await image_downloader.execute(params)

            # Assert
            assert result is None

            # Verify _download_image was called with correct parameters
            expected_full_url = (
                f"{ImageDownloader.TMDB_IMAGE_BASE_URL}/{test_image_url.lstrip('/')}"
            )
            expected_path = Path(tmp_path) / str(test_entity_id) / "image.jpg"
            mock_download.assert_called_once_with(expected_full_url, expected_path)

            # Verify logger was called
            mock_logger.info.assert_any_call(
                f"Downloading image from {expected_full_url} to {expected_path}"
            )
            mock_logger.info.assert_any_call(
                f"Successfully downloaded image to {expected_path}"
            )


@pytest.mark.asyncio
async def test_execute_with_absolute_url(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
    tmp_path: Path,
) -> None:
    """Test execute method with an absolute URL."""
    # Setup
    test_entity_id = uuid.uuid4()
    test_image_url = "https://example.com/images/poster.jpg"
    params = ImageDownloaderParams(image_url=test_image_url, entity_id=test_entity_id)

    # Mock the config.IMAGE_DIRECTORY
    with patch("src.workers.image_downloader.config") as mock_config:
        mock_config.IMAGE_DIRECTORY = str(tmp_path)

        # Mock the _download_image method to return success
        with patch.object(
            image_downloader, "_download_image", return_value=True
        ) as mock_download:
            # Execute
            result = await image_downloader.execute(params)

            # Assert
            assert result is None

            # Verify _download_image was called with the absolute URL
            expected_path = Path(tmp_path) / str(test_entity_id) / "poster.jpg"
            mock_download.assert_called_once_with(test_image_url, expected_path)


@pytest.mark.asyncio
async def test_execute_with_none_params(
    image_downloader: ImageDownloader,
) -> None:
    """Test execute method with None parameters."""
    # Execute and assert
    with pytest.raises(ValueError, match="Parameters cannot be None"):
        await image_downloader.execute(None)


@pytest.mark.asyncio
async def test_execute_with_wrong_params_type(
    image_downloader: ImageDownloader,
) -> None:
    """Test execute method with wrong parameter type."""

    # Setup - create a different parameter type
    class WrongParams:
        pass

    # Execute and assert
    with pytest.raises(
        ValueError, match="Parameters must be of type ImageDownloaderParams"
    ):
        await image_downloader.execute(cast(Any, WrongParams()))


@pytest.mark.asyncio
async def test_execute_with_empty_image_url(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
) -> None:
    """Test execute method with empty image URL."""
    # Setup
    test_entity_id = uuid.uuid4()
    params = ImageDownloaderParams(image_url="", entity_id=test_entity_id)

    # Execute
    result = await image_downloader.execute(params)

    # Assert
    assert result is None
    mock_logger.error.assert_called_once_with("Image URL cannot be empty")


@pytest.mark.asyncio
async def test_execute_download_failure(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
    tmp_path: Path,
) -> None:
    """Test execute method when download fails."""
    # Setup
    test_entity_id = uuid.uuid4()
    test_image_url = "/path/to/image.jpg"
    params = ImageDownloaderParams(image_url=test_image_url, entity_id=test_entity_id)

    # Mock the config.IMAGE_DIRECTORY
    with patch("src.workers.image_downloader.config") as mock_config:
        mock_config.IMAGE_DIRECTORY = str(tmp_path)

        # Mock the _download_image method to return failure
        with patch.object(
            image_downloader, "_download_image", return_value=False
        ) as mock_download:
            # Execute
            result = await image_downloader.execute(params)

            # Assert
            assert result is None

            # Verify _download_image was called
            expected_full_url = (
                f"{ImageDownloader.TMDB_IMAGE_BASE_URL}/{test_image_url.lstrip('/')}"
            )
            mock_download.assert_called_once()

            # Verify logger error was called
            mock_logger.error.assert_called_once_with(
                f"Failed to download image from {expected_full_url}"
            )


@pytest.mark.asyncio
async def test_download_image_success(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
    tmp_path: Path,
) -> None:
    """Test _download_image method with successful download."""
    # Setup
    test_url = "https://example.com/image.jpg"
    test_path = tmp_path / "test_image.jpg"
    test_data = b"test_image_data"

    # Create mock response
    mock_response = MockResponse(status=200, data=test_data)

    # Configure the HTTP client to return the mock response
    image_downloader.http_client.fetch_data.return_value = mock_response  # type: ignore

    # Mock aiofiles.open
    mock_file = AsyncMock()
    mock_file_context = AsyncMock()
    mock_file_context.__aenter__.return_value = mock_file

    with patch("aiofiles.open", return_value=mock_file_context):
        # Execute
        # We're testing a protected method, but it's necessary for thorough testing
        result = await image_downloader._download_image(test_url, test_path)  # type: ignore

        # Assert
        assert result is True

        # Verify HTTP client was called
        image_downloader.http_client.fetch_data.assert_called_once_with(test_url)  # type: ignore

        # Verify file was written
        mock_file.write.assert_called_once_with(test_data)

        # Verify response was released
        assert mock_response.closed is True


@pytest.mark.asyncio
async def test_download_image_http_failure(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
) -> None:
    """Test _download_image method when HTTP request fails."""
    # Setup
    test_url = "https://example.com/image.jpg"
    test_path = Path("/tmp/test_image.jpg")

    # Configure the HTTP client to return None (request failed)
    image_downloader.http_client.fetch_data.return_value = None  # type: ignore

    # Execute
    # We're testing a protected method, but it's necessary for thorough testing
    result = await image_downloader._download_image(test_url, test_path)  # type: ignore

    # Assert
    assert result is False

    # Verify HTTP client was called
    image_downloader.http_client.fetch_data.assert_called_once_with(test_url)  # type: ignore

    # Verify logger error was called
    mock_logger.error.assert_called_once_with(
        f"Failed to download image from {test_url}"
    )


@pytest.mark.asyncio
async def test_download_image_empty_data(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
) -> None:
    """Test _download_image method when response contains empty data."""
    # Setup
    test_url = "https://example.com/image.jpg"
    test_path = Path("/tmp/test_image.jpg")

    # Create mock response with empty data
    mock_response = MockResponse(status=200, data=b"")

    # Configure the HTTP client to return the mock response
    image_downloader.http_client.fetch_data.return_value = mock_response  # type: ignore

    # Execute
    # We're testing a protected method, but it's necessary for thorough testing
    result = await image_downloader._download_image(test_url, test_path)  # type: ignore

    # Assert
    assert result is False

    # Verify HTTP client was called
    image_downloader.http_client.fetch_data.assert_called_once_with(test_url)  # type: ignore

    # Verify logger error was called
    mock_logger.error.assert_called_once_with(
        f"Empty image data received from {test_url}"
    )

    # Verify response was released
    assert mock_response.closed is True


@pytest.mark.asyncio
async def test_download_image_read_exception(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
) -> None:
    """Test _download_image method when reading response raises an exception."""
    # Setup
    test_url = "https://example.com/image.jpg"
    test_path = Path("/tmp/test_image.jpg")

    # Create mock response that raises on read
    mock_response = MockResponse(status=200, raise_on_read=True)

    # Configure the HTTP client to return the mock response
    image_downloader.http_client.fetch_data.return_value = mock_response  # type: ignore

    # Execute
    # We're testing a protected method, but it's necessary for thorough testing
    result = await image_downloader._download_image(test_url, test_path)  # type: ignore

    # Assert
    assert result is False

    # Verify HTTP client was called
    image_downloader.http_client.fetch_data.assert_called_once_with(test_url)  # type: ignore

    # Verify logger error was called
    mock_logger.error.assert_called_once()

    # Verify response was released
    assert mock_response.closed is True


@pytest.mark.asyncio
async def test_download_image_write_exception(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
) -> None:
    """Test _download_image method when writing to file raises an exception."""
    # Setup
    test_url = "https://example.com/image.jpg"
    test_path = Path("/tmp/test_image.jpg")
    test_data = b"test_image_data"

    # Create mock response
    mock_response = MockResponse(status=200, data=test_data)

    # Configure the HTTP client to return the mock response
    image_downloader.http_client.fetch_data.return_value = mock_response  # type: ignore

    # Mock aiofiles.open to raise an exception
    mock_file_context = AsyncMock()
    mock_file_context.__aenter__.side_effect = IOError("Failed to open file")

    with patch("aiofiles.open", return_value=mock_file_context):
        # Execute
        # We're testing a protected method, but it's necessary for thorough testing
        result = await image_downloader._download_image(test_url, test_path)  # type: ignore

        # Assert
        assert result is False

        # Verify HTTP client was called
        image_downloader.http_client.fetch_data.assert_called_once_with(test_url)  # type: ignore

        # Verify logger error was called
        mock_logger.error.assert_called_once()

        # Verify response was released
        assert mock_response.closed is True


@pytest.mark.asyncio
async def test_makedirs_called(
    image_downloader: ImageDownloader,
    mock_logger: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that os.makedirs is called to create the directory."""
    # Setup
    test_entity_id = uuid.uuid4()
    test_image_url = "/path/to/image.jpg"
    params = ImageDownloaderParams(image_url=test_image_url, entity_id=test_entity_id)

    # Mock the config.IMAGE_DIRECTORY
    with patch("src.workers.image_downloader.config") as mock_config:
        mock_config.IMAGE_DIRECTORY = str(tmp_path)

        # Mock os.makedirs
        with patch("os.makedirs") as mock_makedirs:
            # Mock the _download_image method to return success
            with patch.object(image_downloader, "_download_image", return_value=True):
                # Execute
                await image_downloader.execute(params)

                # Assert
                expected_dir = Path(tmp_path) / str(test_entity_id)
                mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
