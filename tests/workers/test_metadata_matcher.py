import pytest
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4

from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    JobType,
    MatchedData,
    MetadataMatcherParams,
    MovieMatcherParams,
    TvMatcherParams,
    EntityType,
    MetadataStatus,
)
from src.common.logger import Logger
from src.common.system_types import MediaType
from src.workers.metadata_matcher import MetadataMatcher


@pytest.fixture
def mock_db_session() -> AsyncDatabaseSession:
    """Create a mock database session."""
    mock_session = MagicMock(spec=AsyncDatabaseSession)
    session_context = AsyncMock()
    mock_session.get_session.return_value.__aiter__.return_value = [session_context]
    return mock_session


@pytest.fixture
def mock_logger() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


@pytest.fixture
def metadata_matcher(
    mock_db_session: AsyncDatabaseSession, mock_logger: Logger
) -> MetadataMatcher:
    """Create a MetadataMatcher instance with mocked dependencies."""
    return MetadataMatcher(db_session=mock_db_session, logger=mock_logger)


@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create a mock HTTP client."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.mark.asyncio
async def test_execute_invalid_parameters(metadata_matcher: MetadataMatcher) -> None:
    """Test that execute raises ValueError when parameters are not of the correct type."""
    with pytest.raises(
        ValueError, match="Parameters must be of type MetadataMatcherParams"
    ):
        await metadata_matcher.execute(cast(Any, {"invalid": "params"}))


@pytest.mark.asyncio
async def test_execute_movie_match(metadata_matcher: MetadataMatcher) -> None:
    """Test that execute handles movie matches correctly."""
    # Update mock results to include all required fields
    mock_results = [
        {
            "id": 12345,
            "title": "Test Movie",
            "release_date": "2020-01-01",
            "poster_path": "/path.jpg",
        }
    ]
    # Use monkeypatch to avoid accessing protected method directly
    setattr(metadata_matcher, "_search_movie", AsyncMock(return_value=mock_results))
    setattr(metadata_matcher, "_insert_media_search_result", AsyncMock())

    test_uuid = uuid4()
    params = MetadataMatcherParams(
        matched_data=MatchedData(
            title="Test Movie", year=2020, media_type=MediaType.MOVIE
        ),
        file_id=test_uuid,
    )

    await metadata_matcher.execute(params)

    # Verify the search_movie method was called with the correct parameters
    search_movie = getattr(metadata_matcher, "_search_movie")
    search_movie.assert_called_once_with(params.matched_data)

    # Verify _insert_media_search_result was called
    insert_media = getattr(metadata_matcher, "_insert_media_search_result")
    assert insert_media.called


@pytest.mark.asyncio
async def test_execute_tv_match(metadata_matcher: MetadataMatcher) -> None:
    """Test that execute handles TV show matches correctly."""
    # Update mock results to include all required fields for TV shows
    mock_results = [
        {
            "id": 67890,
            "name": "Test TV Show",
            "title": "Test TV Show",
            "release_date": "2020-01-01",
            "poster_path": "/path.jpg",
        }
    ]
    setattr(metadata_matcher, "_search_tv", AsyncMock(return_value=mock_results))
    setattr(metadata_matcher, "_insert_media_search_result", AsyncMock())

    test_uuid = uuid4()
    params = MetadataMatcherParams(
        matched_data=MatchedData(
            title="Test TV Show",
            year=2020,
            media_type=MediaType.TV,
            season_number=1,
            episode_number=1,
        ),
        file_id=test_uuid,
    )

    await metadata_matcher.execute(params)

    # Verify the search_tv method was called with the correct parameters
    search_tv = getattr(metadata_matcher, "_search_tv")
    search_tv.assert_called_once_with(params.matched_data)

    # Verify _insert_media_search_result was called
    insert_media = getattr(metadata_matcher, "_insert_media_search_result")
    assert insert_media.called


@pytest.mark.asyncio
async def test_execute_unsupported_media_type(
    metadata_matcher: MetadataMatcher,
) -> None:
    """Test that execute handles unsupported media types correctly."""
    test_uuid = uuid4()
    params = MetadataMatcherParams(
        matched_data=MatchedData(
            title="Test Media",
            year=2020,
            media_type=MediaType.MUSIC,  # Using valid enum value
        ),
        file_id=test_uuid,
    )

    await metadata_matcher.execute(params)

    # Verify logger.error was called with the expected message
    assert metadata_matcher.logger is not None
    metadata_matcher.logger.error.assert_called_once_with(
        f"Unsupported media type: {MediaType.MUSIC}"
    )


@pytest.mark.asyncio
async def test_execute_no_matches(metadata_matcher: MetadataMatcher) -> None:
    """Test that execute handles no matches correctly."""
    setattr(metadata_matcher, "_search_movie", AsyncMock(return_value=[]))

    test_uuid = uuid4()
    params = MetadataMatcherParams(
        matched_data=MatchedData(
            title="Non-existent Movie", year=2020, media_type=MediaType.MOVIE
        ),
        file_id=test_uuid,
    )

    await metadata_matcher.execute(params)

    # Verify logger.info was called with the expected message
    assert metadata_matcher.logger is not None
    metadata_matcher.logger.info.assert_called_once_with(
        "No matches found for Non-existent Movie"
    )


@pytest.mark.asyncio
async def test_search_movie(
    metadata_matcher: MetadataMatcher, mock_http_client: AsyncMock
) -> None:
    """Test the _search_movie method."""
    # We're testing a protected method directly, but it's a test so it's acceptable
    metadata_matcher.http_client = mock_http_client
    mock_response = {"results": [{"id": 12345, "title": "Test Movie"}]}
    mock_http_client.fetch_json.return_value = mock_response

    matched_data = MatchedData(
        title="Test Movie", year=2020, media_type=MediaType.MOVIE
    )

    # Mock the call with proper params
    mock_http_client.fetch_json.call_args = None
    mock_http_client.fetch_json.return_value = mock_response

    result = await metadata_matcher._search_movie(matched_data)  # type: ignore

    assert result == mock_response["results"]
    mock_http_client.fetch_json.assert_called_once()


@pytest.mark.asyncio
async def test_search_movie_no_year(
    metadata_matcher: MetadataMatcher, mock_http_client: AsyncMock
) -> None:
    """Test the _search_movie method without a year."""
    metadata_matcher.http_client = mock_http_client
    mock_response = {"results": [{"id": 12345, "title": "Test Movie"}]}
    mock_http_client.fetch_json.return_value = mock_response

    matched_data = MatchedData(
        title="Test Movie", year=None, media_type=MediaType.MOVIE
    )

    result = await metadata_matcher._search_movie(matched_data)  # type: ignore

    assert result == mock_response["results"]
    mock_http_client.fetch_json.assert_called_once()


@pytest.mark.asyncio
async def test_search_tv(
    metadata_matcher: MetadataMatcher, mock_http_client: AsyncMock
) -> None:
    """Test the _search_tv method."""
    metadata_matcher.http_client = mock_http_client
    mock_response = {"results": [{"id": 67890, "name": "Test TV Show"}]}
    mock_http_client.fetch_json.return_value = mock_response

    matched_data = MatchedData(title="Test TV Show", year=2020, media_type=MediaType.TV)

    result = await metadata_matcher._search_tv(matched_data)

    assert result == mock_response["results"]
    mock_http_client.fetch_json.assert_called_once()


@pytest.mark.asyncio
async def test_insert_entity(metadata_matcher: MetadataMatcher) -> None:
    """Test the _insert_entity method."""
    test_uuid = uuid4()
    file_uuid = uuid4()

    # Create mock entity with valid Entity model fields
    entity = MagicMock()
    entity.id = test_uuid
    entity.model_dump.return_value = {
        "id": test_uuid,
        "file_id": file_uuid,
        "entity_type": EntityType.MOVIE,  # Using EntityType instead of MediaType
        "metadata_status": MetadataStatus.PENDING,  # Required field with default
        "matched_data": {"tmdb_id": 12345},  # Store external ID in matched_data
    }

    result = await metadata_matcher._insert_entity(entity)
    assert result == test_uuid

    session = (
        metadata_matcher.db_session.get_session.return_value.__aiter__.return_value[0]
    )
    session.add.assert_called_once()
    session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_insert_entity_no_id(metadata_matcher: MetadataMatcher) -> None:
    """Test the _insert_entity method when no ID is generated."""
    file_uuid = uuid4()
    entity = MagicMock()
    entity.id = None
    entity.model_dump.return_value = {
        "file_id": file_uuid,
        "entity_type": EntityType.MOVIE,  # Using EntityType instead of MediaType
        "metadata_status": MetadataStatus.PENDING,  # Required field with default
        "matched_data": {"tmdb_id": 12345},  # Store external ID in matched_data
    }

    with pytest.raises(ValueError, match="Failed to generate entity ID"):
        await metadata_matcher._insert_entity(entity)
