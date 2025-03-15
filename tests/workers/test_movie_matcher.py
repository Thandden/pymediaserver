from datetime import date
from typing import Any, AsyncGenerator, Dict, cast
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.db import AsyncDatabaseSession
from src.common.dto import EntityDTO, MovieDTO, MovieMatcherParams
from src.common.http_client import AsyncHttpClient
from src.common.logger import Logger
from src.common.models import Entity, Movie
from src.common.system_types import EntityType, MetadataStatus
from src.workers.movie_matcher import MovieMatcher


@pytest.fixture
def mock_db_session() -> tuple[AsyncMock, AsyncMock]:
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncDatabaseSession)

    # Create mock session that will be returned by the context manager
    db_session = AsyncMock(spec=AsyncSession)

    # Mock the async context manager
    async def async_session_context() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    # Set up the session factory to return our context manager
    session.get_session.return_value = async_session_context()

    return session, db_session


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


@pytest.fixture
def movie_matcher(
    mock_db_session: tuple[AsyncMock, AsyncMock], mock_logger: MagicMock
) -> MovieMatcher:
    """Create a MovieMatcher instance with mocked dependencies."""
    session, _ = mock_db_session
    return MovieMatcher(session, mock_logger)


@pytest.fixture
def mock_tmdb_data() -> Dict[str, Any]:
    """Mock TMDB API response data."""
    return {
        "id": 123,
        "title": "Test Movie",
        "overview": "A test movie description",
        "poster_path": "/path/to/poster.jpg",
        "backdrop_path": "/path/to/backdrop.jpg",
        "release_date": "2023-01-15",
        "credits": {"cast": [{"name": "Actor 1", "character": "Character 1"}]},
        "keywords": {"keywords": [{"id": 1, "name": "action"}]},
        "videos": {"results": [{"key": "trailer_id", "site": "YouTube"}]},
        "images": {"backdrops": [], "posters": []},
    }


@pytest.fixture
def mock_params() -> MovieMatcherParams:
    """Create mock parameters for the movie matcher."""
    return MovieMatcherParams(tmdb_id=123, file_id=uuid.uuid4())


@pytest.fixture
def movie_matcher_with_test_access(
    mock_db_session: tuple[AsyncMock, AsyncMock], mock_logger: MagicMock
) -> MovieMatcher:
    """Create a MovieMatcher instance with test-access to protected methods."""
    session, _ = mock_db_session
    matcher = MovieMatcher(session, mock_logger)

    # Expose protected methods for testing
    # This is a common pattern for testing protected methods
    matcher.fetch_movie_details = matcher._fetch_movie_details  # type: ignore
    matcher.insert_movie = matcher._insert_movie  # type: ignore
    matcher.insert_entity = matcher._insert_entity  # type: ignore
    matcher.create_movie_dto = matcher._create_movie_dto  # type: ignore
    matcher.find_pending_entity = matcher._find_pending_entity  # type: ignore

    return matcher


@pytest.mark.asyncio
async def test_execute_invalid_parameters(movie_matcher: MovieMatcher) -> None:
    """Test execute method with invalid parameters."""
    # Call with invalid parameters
    with pytest.raises(
        ValueError, match="Parameters must be of type MovieMatcherParams"
    ):
        await movie_matcher.execute({})  # type: ignore


@pytest.mark.asyncio
async def test_execute_fetch_details_failure(
    movie_matcher: MovieMatcher, mock_params: MovieMatcherParams
) -> None:
    """Test execute method when fetch_movie_details fails."""
    # Mock the dependencies
    movie_matcher._fetch_movie_details = AsyncMock(return_value={})

    # Call the method
    result = await movie_matcher.execute(mock_params)

    # Assertions
    assert result == []
    movie_matcher._fetch_movie_details.assert_called_once_with(mock_params.tmdb_id)
    mock_logger = cast(MagicMock, movie_matcher.logger)
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_execute_insert_movie_failure(
    movie_matcher: MovieMatcher,
    mock_params: MovieMatcherParams,
    mock_tmdb_data: Dict[str, Any],
) -> None:
    """Test execute method when insert_movie fails."""
    # Mock the dependencies
    movie_matcher._fetch_movie_details = AsyncMock(return_value=mock_tmdb_data)
    movie_matcher._insert_movie = AsyncMock(return_value=None)

    # Call the method
    result = await movie_matcher.execute(mock_params)

    # Assertions
    assert result == []
    movie_matcher._fetch_movie_details.assert_called_once_with(mock_params.tmdb_id)
    movie_matcher._insert_movie.assert_called_once()
    mock_logger = cast(MagicMock, movie_matcher.logger)
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_movie_details_success(
    movie_matcher: MovieMatcher, mock_tmdb_data: Dict[str, Any]
) -> None:
    """Test _fetch_movie_details method success case."""
    # Mock HTTP client
    http_client_mock = AsyncMock(spec=AsyncHttpClient)
    http_client_mock.__aenter__.return_value = http_client_mock
    http_client_mock.fetch_json = AsyncMock(return_value=mock_tmdb_data)

    with patch.object(movie_matcher, "http_client", http_client_mock):
        result = await movie_matcher._fetch_movie_details(123)

    # Assertions
    assert result == mock_tmdb_data
    http_client_mock.fetch_json.assert_called_once()
    params_call = http_client_mock.fetch_json.call_args[0][1]
    assert "api_key" in params_call
    assert "append_to_response" in params_call


@pytest.mark.asyncio
async def test_fetch_movie_details_failure(movie_matcher: MovieMatcher) -> None:
    """Test _fetch_movie_details method when API call fails."""
    # Mock HTTP client to raise exception
    http_client_mock = AsyncMock(spec=AsyncHttpClient)
    http_client_mock.__aenter__.return_value = http_client_mock
    http_client_mock.fetch_json = AsyncMock(side_effect=Exception("API error"))

    with patch.object(movie_matcher, "http_client", http_client_mock):
        result = await movie_matcher._fetch_movie_details(123)

    # Assertions
    assert result == {}
    http_client_mock.fetch_json.assert_called_once()
    mock_logger = cast(MagicMock, movie_matcher.logger)
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_movie_details_empty_response(movie_matcher: MovieMatcher) -> None:
    """Test _fetch_movie_details method when API returns empty data."""
    # Mock HTTP client to return None
    http_client_mock = AsyncMock(spec=AsyncHttpClient)
    http_client_mock.__aenter__.return_value = http_client_mock
    http_client_mock.fetch_json = AsyncMock(return_value=None)

    with patch.object(movie_matcher, "http_client", http_client_mock):
        result = await movie_matcher._fetch_movie_details(123)

    # Assertions
    assert result == {}


def test_create_movie_dto_complete_data(
    movie_matcher: MovieMatcher, mock_tmdb_data: Dict[str, Any]
) -> None:
    """Test _create_movie_dto method with complete data."""
    result = movie_matcher._create_movie_dto(mock_tmdb_data)

    # Assertions
    assert isinstance(result, MovieDTO)
    assert result.tmdb_id == mock_tmdb_data["id"]
    assert result.title == mock_tmdb_data["title"]
    assert result.overview == mock_tmdb_data["overview"]
    assert result.poster_path == mock_tmdb_data["poster_path"]
    assert result.backdrop_path == mock_tmdb_data["backdrop_path"]
    assert result.release_date == date(2023, 1, 15)
    assert result.year == 2023


def test_create_movie_dto_missing_release_date(
    movie_matcher: MovieMatcher, mock_tmdb_data: Dict[str, Any]
) -> None:
    """Test _create_movie_dto method with missing release date."""
    # Remove release date from mock data
    data_without_date = mock_tmdb_data.copy()
    data_without_date.pop("release_date")

    result = movie_matcher._create_movie_dto(data_without_date)

    # Assertions
    assert isinstance(result, MovieDTO)
    assert result.release_date is None
    assert result.year is None


def test_create_movie_dto_invalid_release_date(
    movie_matcher: MovieMatcher, mock_tmdb_data: Dict[str, Any]
) -> None:
    """Test _create_movie_dto method with invalid release date."""
    # Set invalid release date in mock data
    data_with_invalid_date = mock_tmdb_data.copy()
    data_with_invalid_date["release_date"] = "not-a-date"

    result = movie_matcher._create_movie_dto(data_with_invalid_date)

    # Assertions
    assert isinstance(result, MovieDTO)
    assert result.release_date is None
    assert result.year is None


def test_create_movie_dto_minimal_data(movie_matcher: MovieMatcher) -> None:
    """Test _create_movie_dto method with minimal required data."""
    minimal_data = {
        "id": 123,
        "title": "Test Movie",
        "poster_path": "/path/to/poster.jpg",
        "backdrop_path": "/path/to/backdrop.jpg",
    }

    result = movie_matcher._create_movie_dto(minimal_data)

    # Assertions
    assert isinstance(result, MovieDTO)
    assert result.tmdb_id == minimal_data["id"]
    assert result.title == minimal_data["title"]
    assert result.overview is None
    assert result.release_date is None
    assert result.year is None


@pytest.mark.asyncio
async def test_find_pending_entity_not_found(
    movie_matcher: MovieMatcher, mock_db_session: tuple[AsyncMock, AsyncMock]
) -> None:
    """Test _find_pending_entity method when no entity is found."""
    session_factory, db_session = mock_db_session

    # Mock the execution result
    mappings_result = AsyncMock()
    mappings_result.first.return_value = None
    execution_result = AsyncMock()
    execution_result.mappings.return_value = mappings_result
    db_session.execute = AsyncMock(return_value=execution_result)

    # Call the method
    result = await movie_matcher._find_pending_entity(123)

    # Assertions
    assert result is None
    assert db_session.execute.call_count == 1


@pytest.mark.asyncio
async def test_find_pending_entity_failure(
    movie_matcher: MovieMatcher, mock_db_session: tuple[AsyncMock, AsyncMock]
) -> None:
    """Test _find_pending_entity method when database operation fails."""
    session_factory, db_session = mock_db_session
    db_session.execute = AsyncMock(side_effect=Exception("Database error"))

    # Call the method
    result = await movie_matcher._find_pending_entity(123)

    # Assertions
    assert result is None
    assert db_session.execute.call_count == 1
