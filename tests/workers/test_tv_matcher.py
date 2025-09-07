# pyright: reportProtectedMemberAccess=false
import asyncio
from datetime import date
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from logging import Logger

from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    ChildJobRequest,
    EntityDTO,
    ImageDownloaderParams,
    JobType,
    NoChildJob,
    TVEpisodeDTO,
    TVSeasonDTO,
    TVShowDTO,
    TvMatcherParams,
)
from src.common.system_types import EntityType, MetadataStatus
from src.common.models import Entity, TVEpisode, TVSeason, TVShow
from src.workers.tv_matcher import TVMatcher


class MockResponse:
    """Mock implementation of HTTP response for testing"""

    def __init__(
        self,
        status: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
    ):
        self.status = status
        self._json_data = json_data or {}
        self.closed = False

    async def json(self) -> Dict[str, Any]:
        return self._json_data

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def mock_logger() -> Logger:
    """Fixture for a mocked logger"""
    return Mock(spec=Logger)


@pytest.fixture
def mock_db_session() -> AsyncDatabaseSession:
    """Fixture for a mocked database session"""
    mock_session = AsyncMock(spec=AsyncDatabaseSession)

    # Create a mock for the session context manager
    mock_session_context = AsyncMock()
    session_obj = AsyncMock(spec=AsyncSession)
    mock_session_context.__aenter__.return_value = session_obj
    mock_session_context.__aexit__.return_value = None

    # Create a proper async generator for get_session
    async def mock_get_session():
        yield mock_session_context

    # Make get_session return our mock async generator
    mock_session.get_session = mock_get_session

    return mock_session


@pytest_asyncio.fixture
async def tv_matcher(
    mock_db_session: AsyncDatabaseSession, mock_logger: Logger
) -> AsyncGenerator[TVMatcher, None]:
    """Fixture for a TVMatcher instance with mocked dependencies"""
    with patch("src.workers.tv_matcher.AsyncHttpClient") as mock_http_client_class:
        # Create a mock for the HTTP client
        mock_http_client = AsyncMock()
        mock_http_client_class.return_value = mock_http_client

        # Create the matcher with mocked dependencies
        matcher = TVMatcher(mock_db_session, mock_logger)

        # Replace the HTTP client with our mock
        matcher.http_client = mock_http_client

        yield matcher


@pytest.fixture
def valid_params() -> TvMatcherParams:
    """Fixture for valid TV matcher parameters"""
    return TvMatcherParams(
        tmdb_id=12345, season_number=1, episode_number=2, file_id=uuid4()
    )


@pytest.fixture
def mock_tv_show_data() -> Dict[str, Any]:
    """Fixture for mock TV show data from TMDB"""
    return {
        "id": 12345,
        "name": "Test Show",
        "overview": "This is a test show",
        "poster_path": "/path/to/poster.jpg",
        "backdrop_path": "/path/to/backdrop.jpg",
        "first_air_date": "2020-01-01",
        "seasons": [
            {
                "id": 67890,
                "name": "Season 1",
                "season_number": 1,
                "episode_count": 10,
                "overview": "Season 1 overview",
                "poster_path": "/path/to/season_poster.jpg",
                "air_date": "2020-01-01",
            }
        ],
    }


@pytest.fixture
def mock_season_data() -> Dict[str, Any]:
    """Fixture for mock season data from TMDB"""
    return {
        "id": 67890,
        "name": "Season 1",
        "season_number": 1,
        "overview": "Season 1 overview",
        "poster_path": "/path/to/season_poster.jpg",
        "air_date": "2020-01-01",
        "episodes": [
            {
                "id": 111111,
                "name": "Episode 1",
                "episode_number": 1,
                "overview": "Episode 1 overview",
                "still_path": "/path/to/still1.jpg",
                "air_date": "2020-01-01",
            },
            {
                "id": 222222,
                "name": "Episode 2",
                "episode_number": 2,
                "overview": "Episode 2 overview",
                "still_path": "/path/to/still2.jpg",
                "air_date": "2020-01-08",
            },
        ],
    }


@pytest.fixture
def mock_episode_data() -> Dict[str, Any]:
    """Fixture for mock episode data from TMDB"""
    return {
        "id": 222222,
        "name": "Episode 2",
        "episode_number": 2,
        "season_number": 1,
        "overview": "Episode 2 overview",
        "still_path": "/path/to/still2.jpg",
        "air_date": "2020-01-08",
        "credits": {"cast": [], "crew": []},
        "images": {"stills": []},
        "videos": {"results": []},
    }


@pytest.mark.asyncio
async def test_validate_parameters_valid(
    tv_matcher: TVMatcher, valid_params: TvMatcherParams
) -> None:
    """Test parameter validation with valid parameters"""
    result = tv_matcher._validate_parameters(valid_params)
    assert result is valid_params


@pytest.mark.asyncio
async def test_validate_parameters_invalid(tv_matcher: TVMatcher) -> None:
    """Test parameter validation with invalid parameters"""
    with pytest.raises(ValueError, match="Parameters must be of type TvMatcherParams"):
        tv_matcher._validate_parameters(None)

    with pytest.raises(ValueError, match="Parameters must be of type TvMatcherParams"):
        tv_matcher._validate_parameters({"tmdb_id": 12345})  # type: ignore


@pytest.mark.asyncio
async def test_fetch_tv_show_details_success(
    tv_matcher: TVMatcher, mock_tv_show_data: Dict[str, Any]
) -> None:
    """Test successful fetching of TV show details"""
    # Setup mock response
    tv_matcher.http_client.fetch_json.return_value = mock_tv_show_data

    # Call the method
    result = await tv_matcher._fetch_tv_show_details(12345)

    # Verify results
    assert result == mock_tv_show_data
    tv_matcher.http_client.fetch_json.assert_called_once()

    # Verify the API endpoint and parameters
    call_args = tv_matcher.http_client.fetch_json.call_args[0]
    assert "tv/12345" in call_args[0]
    assert "api_key" in call_args[1]
    assert "append_to_response" in call_args[1]


@pytest.mark.asyncio
async def test_fetch_tv_show_details_failure(tv_matcher: TVMatcher) -> None:
    """Test handling of failed TV show details fetch"""
    # Setup mock to return None (failed request)
    tv_matcher.http_client.fetch_json.return_value = None

    # Call the method
    result = await tv_matcher._fetch_tv_show_details(12345)

    # Verify results
    assert result == {}
    tv_matcher.http_client.fetch_json.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_tv_show_details_exception(tv_matcher: TVMatcher) -> None:
    """Test handling of exceptions during TV show details fetch"""
    # Setup mock to raise an exception
    tv_matcher.http_client.fetch_json.side_effect = Exception("API error")

    # Call the method
    result = await tv_matcher._fetch_tv_show_details(12345)

    # Verify results
    assert result == {}
    tv_matcher.http_client.fetch_json.assert_called_once()
    tv_matcher.logger.error.assert_called_once()
    assert "Error fetching TV show details" in tv_matcher.logger.error.call_args[0][0]


@pytest.mark.asyncio
async def test_fetch_season_details_success(
    tv_matcher: TVMatcher, mock_season_data: Dict[str, Any]
) -> None:
    """Test successful fetching of season details"""
    # Setup mock response
    tv_matcher.http_client.fetch_json.return_value = mock_season_data

    # Call the method
    result = await tv_matcher._fetch_season_details(12345, 1)

    # Verify results
    assert result == mock_season_data
    tv_matcher.http_client.fetch_json.assert_called_once()

    # Verify the API endpoint and parameters
    call_args = tv_matcher.http_client.fetch_json.call_args[0]
    assert "tv/12345/season/1" in call_args[0]
    assert "api_key" in call_args[1]


@pytest.mark.asyncio
async def test_fetch_season_details_failure(tv_matcher: TVMatcher) -> None:
    """Test handling of failed season details fetch"""
    # Setup mock to return None (failed request)
    tv_matcher.http_client.fetch_json.return_value = None

    # Call the method
    result = await tv_matcher._fetch_season_details(12345, 1)

    # Verify results
    assert result == {}
    tv_matcher.http_client.fetch_json.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_episode_details_success(
    tv_matcher: TVMatcher, mock_episode_data: Dict[str, Any]
) -> None:
    """Test successful fetching of episode details"""
    # Setup mock response
    tv_matcher.http_client.fetch_json.return_value = mock_episode_data

    # Call the method
    result = await tv_matcher._fetch_episode_details(12345, 1, 2)

    # Verify results
    assert result == mock_episode_data
    tv_matcher.http_client.fetch_json.assert_called_once()

    # Verify the API endpoint and parameters
    call_args = tv_matcher.http_client.fetch_json.call_args[0]
    assert "tv/12345/season/1/episode/2" in call_args[0]
    assert "api_key" in call_args[1]
    assert "append_to_response" in call_args[1]


@pytest.mark.asyncio
async def test_create_tv_show_dto(
    tv_matcher: TVMatcher, mock_tv_show_data: Dict[str, Any]
) -> None:
    """Test creation of TVShowDTO from TMDB data"""
    # Call the method
    result = tv_matcher._create_tv_show_dto(mock_tv_show_data)

    # Verify results
    assert isinstance(result, TVShowDTO)
    assert result.tmdb_id == 12345
    assert result.title == "Test Show"
    assert result.overview == "This is a test show"
    assert result.poster_path == "/path/to/poster.jpg"
    assert result.backdrop_path == "/path/to/backdrop.jpg"
    assert result.year == 2020


@pytest.mark.asyncio
async def test_create_tv_show_dto_missing_data(tv_matcher: TVMatcher) -> None:
    """Test creation of TVShowDTO with missing data"""
    # Call the method with minimal data
    result = tv_matcher._create_tv_show_dto({"id": 12345, "name": "Test Show"})

    # Verify results
    assert isinstance(result, TVShowDTO)
    assert result.tmdb_id == 12345
    assert result.title == "Test Show"
    assert result.overview is None
    assert result.poster_path is None
    assert result.backdrop_path is None
    assert result.year is None


@pytest.mark.asyncio
async def test_create_tv_show_dto_invalid_date(tv_matcher: TVMatcher) -> None:
    """Test creation of TVShowDTO with invalid date"""
    # Call the method with invalid date
    result = tv_matcher._create_tv_show_dto(
        {"id": 12345, "name": "Test Show", "first_air_date": "invalid-date"}
    )

    # Verify results
    assert isinstance(result, TVShowDTO)
    assert result.year is None


@pytest.mark.asyncio
async def test_create_tv_season_dto(
    tv_matcher: TVMatcher, mock_season_data: Dict[str, Any]
) -> None:
    """Test creation of TVSeasonDTO from TMDB data"""
    # Call the method
    show_id = uuid4()
    result = tv_matcher._create_tv_season_dto(mock_season_data, show_id)

    # Verify results
    assert isinstance(result, TVSeasonDTO)
    assert result.show_id == show_id
    assert result.season_number == 1
    assert result.title == "Season 1"
    assert result.overview == "Season 1 overview"
    assert result.poster_path == "/path/to/season_poster.jpg"
    assert result.year == 2020


@pytest.mark.asyncio
async def test_create_tv_episode_dto(
    tv_matcher: TVMatcher, mock_episode_data: Dict[str, Any]
) -> None:
    """Test creation of TVEpisodeDTO from TMDB data"""
    # Call the method
    season_id = uuid4()
    result = tv_matcher._create_tv_episode_dto(mock_episode_data, season_id)

    # Verify results
    assert isinstance(result, TVEpisodeDTO)
    assert result.season_id == season_id
    assert result.episode_number == 2
    assert result.title == "Episode 2"
    assert result.overview == "Episode 2 overview"
    assert result.still_path == "/path/to/still2.jpg"
    assert result.air_date == date(2020, 1, 8)


@pytest.mark.asyncio
async def test_process_tv_show_success(
    tv_matcher: TVMatcher, mock_tv_show_data: Dict[str, Any]
) -> None:
    """Test successful processing of TV show data"""
    # Setup
    tmdb_id = 12345
    show_id = uuid4()

    # Mock the fetch_tv_show_details method
    with patch.object(
        tv_matcher, "_fetch_tv_show_details", return_value=mock_tv_show_data
    ) as mock_fetch:
        # Mock the insert_tv_show method
        with patch.object(
            tv_matcher, "_insert_tv_show", return_value=show_id
        ) as mock_insert:
            # Call the method
            result = await tv_matcher._process_tv_show(tmdb_id)

            # Verify results
            assert result is not None
            tv_show_dto, result_show_id, show_details = result
            assert isinstance(tv_show_dto, TVShowDTO)
            assert result_show_id == show_id
            assert show_details == mock_tv_show_data

            # Verify method calls
            mock_fetch.assert_called_once_with(tmdb_id)
            mock_insert.assert_called_once()


@pytest.mark.asyncio
async def test_process_tv_show_fetch_failure(tv_matcher: TVMatcher) -> None:
    """Test handling of TV show fetch failure"""
    # Setup
    tmdb_id = 12345

    # Mock the fetch_tv_show_details method to return empty dict
    with patch.object(
        tv_matcher, "_fetch_tv_show_details", return_value={}
    ) as mock_fetch:
        # Call the method
        result = await tv_matcher._process_tv_show(tmdb_id)

        # Verify results
        assert result is None
        mock_fetch.assert_called_once_with(tmdb_id)
        tv_matcher.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_process_tv_show_insert_failure(
    tv_matcher: TVMatcher, mock_tv_show_data: Dict[str, Any]
) -> None:
    """Test handling of TV show insert failure"""
    # Setup
    tmdb_id = 12345

    # Mock the fetch_tv_show_details method
    with patch.object(
        tv_matcher, "_fetch_tv_show_details", return_value=mock_tv_show_data
    ) as mock_fetch:
        # Mock the insert_tv_show method to return None (failure)
        with patch.object(
            tv_matcher, "_insert_tv_show", return_value=None
        ) as mock_insert:
            # Call the method
            result = await tv_matcher._process_tv_show(tmdb_id)

            # Verify results
            assert result is None
            mock_fetch.assert_called_once_with(tmdb_id)
            mock_insert.assert_called_once()
            tv_matcher.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_process_single_season_success(
    tv_matcher: TVMatcher, mock_season_data: Dict[str, Any]
) -> None:
    """Test successful processing of a single season"""
    # Setup
    tmdb_id = 12345
    show_id = uuid4()
    season_number = 1
    season_id = uuid4()

    # Mock the fetch_season_details method
    with patch.object(
        tv_matcher, "_fetch_season_details", return_value=mock_season_data
    ) as mock_fetch:
        # Mock the insert_tv_season method
        with patch.object(
            tv_matcher, "_insert_tv_season", return_value=season_id
        ) as mock_insert:
            # Mock the process_season_episodes method
            with patch.object(
                tv_matcher, "_process_season_episodes"
            ) as mock_process_episodes:
                # Call the method
                result = await tv_matcher._process_single_season(
                    tmdb_id, show_id, season_number
                )

                # Verify results
                assert result is True

                # Verify method calls
                mock_fetch.assert_called_once_with(tmdb_id, season_number)
                mock_insert.assert_called_once()
                mock_process_episodes.assert_called_once_with(
                    mock_season_data, season_id
                )


@pytest.mark.asyncio
async def test_process_single_season_fetch_failure(tv_matcher: TVMatcher) -> None:
    """Test handling of season fetch failure"""
    # Setup
    tmdb_id = 12345
    show_id = uuid4()
    season_number = 1

    # Mock the fetch_season_details method to return empty dict
    with patch.object(
        tv_matcher, "_fetch_season_details", return_value={}
    ) as mock_fetch:
        # Call the method
        result = await tv_matcher._process_single_season(
            tmdb_id, show_id, season_number
        )

        # Verify results
        assert result is False
        mock_fetch.assert_called_once_with(tmdb_id, season_number)
        tv_matcher.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_process_single_season_insert_failure(
    tv_matcher: TVMatcher, mock_season_data: Dict[str, Any]
) -> None:
    """Test handling of season insert failure"""
    # Setup
    tmdb_id = 12345
    show_id = uuid4()
    season_number = 1

    # Mock the fetch_season_details method
    with patch.object(
        tv_matcher, "_fetch_season_details", return_value=mock_season_data
    ) as mock_fetch:
        # Mock the insert_tv_season method to return None (failure)
        with patch.object(
            tv_matcher, "_insert_tv_season", return_value=None
        ) as mock_insert:
            # Call the method
            result = await tv_matcher._process_single_season(
                tmdb_id, show_id, season_number
            )

            # Verify results
            assert result is False
            mock_fetch.assert_called_once_with(tmdb_id, season_number)
            mock_insert.assert_called_once()
            tv_matcher.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_process_season_episodes(
    tv_matcher: TVMatcher, mock_season_data: Dict[str, Any]
) -> None:
    """Test processing of season episodes"""
    # Setup
    season_id = uuid4()
    episode_id = uuid4()

    # Mock the insert_tv_episode method
    with patch.object(
        tv_matcher, "_insert_tv_episode", return_value=episode_id
    ) as mock_insert:
        # Call the method
        await tv_matcher._process_season_episodes(mock_season_data, season_id)
        # Verify method calls
        mock_insert.assert_called()
        # We should have 2 episodes in our mock data
        assert mock_insert.call_count == 2


@pytest.mark.asyncio
async def test_process_all_seasons_and_episodes_success(
    tv_matcher: TVMatcher, mock_tv_show_data: Dict[str, Any]
) -> None:
    """Test successful processing of all seasons and episodes"""
    # Setup
    tmdb_id = 12345
    tv_show_id = uuid4()

    # Mock the process_single_season method to return True
    with patch.object(
        tv_matcher, "_process_single_season", return_value=True
    ) as mock_process_season:
        # Call the method
        result = await tv_matcher._process_all_seasons_and_episodes(
            tmdb_id, tv_show_id, mock_tv_show_data
        )

        # Verify results
        assert result is True

        # We should have one season in our mock data
        assert mock_process_season.call_count == 1
        mock_process_season.assert_called_with(tmdb_id, tv_show_id, 1)


@pytest.mark.asyncio
async def test_process_all_seasons_and_episodes_with_special_season(
    tv_matcher: TVMatcher,
) -> None:
    """Test processing of seasons with special season 0"""
    # Setup
    tmdb_id = 12345
    tv_show_id = uuid4()

    # Create show data with a special season (season 0)
    show_data = {
        "seasons": [
            {
                "id": 67890,
                "name": "Specials",
                "season_number": 0,
                "episode_count": 2,
                "overview": "Special episodes",
                "poster_path": "/path/to/special_poster.jpg",
                "air_date": "2020-01-01",
            },
            {
                "id": 67891,
                "name": "Season 1",
                "season_number": 1,
                "episode_count": 10,
                "overview": "Season 1 overview",
                "poster_path": "/path/to/season_poster.jpg",
                "air_date": "2020-01-01",
            },
        ]
    }

    # Mock the process_single_season method to return True
    with patch.object(
        tv_matcher, "_process_single_season", return_value=True
    ) as mock_process_season:
        # Call the method
        result = await tv_matcher._process_all_seasons_and_episodes(
            tmdb_id, tv_show_id, show_data
        )

        # Verify results
        assert result is True

        # We should process both seasons (including season 0)
        assert mock_process_season.call_count == 2
        # Check that both seasons were processed
        mock_process_season.assert_any_call(tmdb_id, tv_show_id, 0)
        mock_process_season.assert_any_call(tmdb_id, tv_show_id, 1)


@pytest.mark.asyncio
async def test_process_all_seasons_and_episodes_with_negative_season(
    tv_matcher: TVMatcher,
) -> None:
    """Test processing of seasons with negative season number (should be skipped)"""
    # Setup
    tmdb_id = 12345
    tv_show_id = uuid4()

    # Create show data with a negative season number (should be skipped)
    show_data = {
        "seasons": [
            {
                "id": 67890,
                "name": "Invalid Season",
                "season_number": -1,  # Negative season number
                "episode_count": 2,
                "overview": "Invalid season",
                "poster_path": "/path/to/invalid_poster.jpg",
                "air_date": "2020-01-01",
            },
            {
                "id": 67891,
                "name": "Season 1",
                "season_number": 1,
                "episode_count": 10,
                "overview": "Season 1 overview",
                "poster_path": "/path/to/season_poster.jpg",
                "air_date": "2020-01-01",
            },
        ]
    }

    # Mock the process_single_season method to return True
    with patch.object(
        tv_matcher, "_process_single_season", return_value=True
    ) as mock_process_season:
        # Call the method
        result = await tv_matcher._process_all_seasons_and_episodes(
            tmdb_id, tv_show_id, show_data
        )

        # Verify results
        assert result is True

        # We should only process season 1 (negative season should be skipped)
        assert mock_process_season.call_count == 1
        mock_process_season.assert_called_once_with(tmdb_id, tv_show_id, 1)


@pytest.mark.asyncio
async def test_process_all_seasons_and_episodes_with_season_failure(
    tv_matcher: TVMatcher, mock_tv_show_data: Dict[str, Any]
) -> None:
    """Test processing of all seasons with one season failing"""
    # Setup
    tmdb_id = 12345
    tv_show_id = uuid4()

    # Mock the process_single_season method to return False (failure)
    with patch.object(
        tv_matcher, "_process_single_season", return_value=False
    ) as mock_process_season:
        # Call the method
        result = await tv_matcher._process_all_seasons_and_episodes(
            tmdb_id, tv_show_id, mock_tv_show_data
        )

        # Verify results - should still return True even if a season fails
        assert result is True

        # We should have one season in our mock data
        assert mock_process_season.call_count == 1
        mock_process_season.assert_called_with(tmdb_id, tv_show_id, 1)

        # Logger should have recorded a warning
        tv_matcher.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_find_target_season_and_episode_season_not_found(
    tv_matcher: TVMatcher,
) -> None:
    """Test handling of season not found"""
    # Setup
    tv_show_id = uuid4()
    season_number = 1
    episode_number = 2
    file_id = uuid4()

    # Get the session object from the mock
    session_context = await anext(tv_matcher.db_session.get_session())
    session_obj = await session_context.__aenter__()

    # Mock the execute result for season query (not found)
    season_result = AsyncMock()
    season_result.scalar_one_or_none.return_value = None
    session_obj.execute.return_value = season_result

    # Call the method
    result = await tv_matcher._find_target_season_and_episode(
        tv_show_id, season_number, episode_number, file_id
    )

    # Verify results
    assert result is None

    # Verify logger was called
    tv_matcher.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_find_target_season_and_episode_episode_not_found(
    tv_matcher: TVMatcher,
) -> None:
    """Test handling of episode not found"""
    # Setup
    tv_show_id = uuid4()
    season_number = 1
    episode_number = 2
    file_id = uuid4()

    # Create mock season
    mock_season = Mock(spec=TVSeason)
    mock_season.id = uuid4()

    # Get the session object from the mock
    session_context = await anext(tv_matcher.db_session.get_session())
    session_obj = await session_context.__aenter__()

    # Mock the execute results
    season_result = AsyncMock()
    season_result.scalar_one_or_none.return_value = mock_season

    episode_result = AsyncMock()
    episode_result.scalar_one_or_none.return_value = None

    # Set up the execute method to return different results for different queries
    session_obj.execute.side_effect = [season_result, episode_result]

    # Call the method
    result = await tv_matcher._find_target_season_and_episode(
        tv_show_id, season_number, episode_number, file_id
    )

    # Verify results
    assert result is None

    # Verify logger was called
    tv_matcher.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_find_target_season_and_episode_database_error(
    tv_matcher: TVMatcher,
) -> None:
    """Test handling of database error"""
    # Setup
    tv_show_id = uuid4()
    season_number = 1
    episode_number = 2
    file_id = uuid4()

    # Get the session object from the mock
    session_context = await anext(tv_matcher.db_session.get_session())
    session_obj = await session_context.__aenter__()

    # Mock the execute method to raise an exception
    session_obj.execute.side_effect = Exception("Database error")

    # Call the method
    result = await tv_matcher._find_target_season_and_episode(
        tv_show_id, season_number, episode_number, file_id
    )

    # Verify results
    assert result is None

    # Verify logger was called
    tv_matcher.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_create_episode_entity_success(
    tv_matcher: TVMatcher, mock_episode_data: Dict[str, Any]
) -> None:
    """Test successful creation of episode entity"""
    # Setup
    file_id = uuid4()
    tv_episode_id = uuid4()
    season_number = 1
    episode_number = 2
    tv_show_dto = TVShowDTO(
        tmdb_id=12345,
        title="Test Show",
        overview="Test overview",
        poster_path="/path/to/poster.jpg",
        backdrop_path="/path/to/backdrop.jpg",
        year=2020,
    )
    tmdb_id = 12345
    entity_id = uuid4()

    # Mock the insert_entity method
    with patch.object(
        tv_matcher, "_insert_entity", return_value=entity_id
    ) as mock_insert:
        # Call the method
        result = await tv_matcher._create_episode_entity(
            file_id,
            tv_episode_id,
            mock_episode_data,
            season_number,
            episode_number,
            tv_show_dto,
            tmdb_id,
        )

        # Verify results
        assert result == entity_id

        # Verify method calls
        mock_insert.assert_called_once()

        # Verify the entity DTO passed to insert_entity
        entity_dto = mock_insert.call_args[0][0]
        assert isinstance(entity_dto, EntityDTO)
        assert entity_dto.file_id == file_id
        assert entity_dto.entity_type == EntityType.TV_EPISODE
        assert entity_dto.tv_episode_id == tv_episode_id
        assert entity_dto.matched_data == mock_episode_data
        assert entity_dto.metadata_status == MetadataStatus.CONFIRMED

        # Verify logger was called
        tv_matcher.logger.info.assert_called_once()


@pytest.mark.asyncio
async def test_create_episode_entity_failure(
    tv_matcher: TVMatcher, mock_episode_data: Dict[str, Any]
) -> None:
    """Test handling of entity creation failure"""
    # Setup
    file_id = uuid4()
    tv_episode_id = uuid4()
    season_number = 1
    episode_number = 2
    tv_show_dto = TVShowDTO(
        tmdb_id=12345,
        title="Test Show",
        overview="Test overview",
        poster_path="/path/to/poster.jpg",
        backdrop_path="/path/to/backdrop.jpg",
        year=2020,
    )
    tmdb_id = 12345

    # Mock the insert_entity method to return None (failure)
    with patch.object(tv_matcher, "_insert_entity", return_value=None) as mock_insert:
        # Call the method
        result = await tv_matcher._create_episode_entity(
            file_id,
            tv_episode_id,
            mock_episode_data,
            season_number,
            episode_number,
            tv_show_dto,
            tmdb_id,
        )

        # Verify results
        assert result is None

        # Verify method calls
        mock_insert.assert_called_once()

        # Verify logger was called
        tv_matcher.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_create_image_download_jobs(tv_matcher: TVMatcher) -> None:
    """Test creation of image download jobs"""
    # Setup
    entity_id = uuid4()

    # Create test data with all image paths
    tv_show_dto = TVShowDTO(
        tmdb_id=12345,
        title="Test Show",
        overview="Test overview",
        poster_path="/path/to/poster.jpg",
        backdrop_path="/path/to/backdrop.jpg",
        year=2020,
    )

    target_season = Mock(spec=TVSeason)
    target_season.poster_path = "/path/to/season_poster.jpg"

    target_episode = Mock(spec=TVEpisode)
    target_episode.still_path = "/path/to/episode_still.jpg"

    # Call the method
    result = tv_matcher._create_image_download_jobs(
        entity_id, tv_show_dto, target_season, target_episode
    )

    # Verify results
    assert isinstance(result, list)
    assert (
        len(result) == 4
    )  # Should have 4 jobs (show poster, backdrop, season poster, episode still)

    # Verify all jobs are of the correct type
    for job in result:
        assert isinstance(job, ChildJobRequest)
        assert job.job_type == JobType.IMAGE_DOWNLOADER
        assert isinstance(job.params, ImageDownloaderParams)
        assert job.params.entity_id == entity_id


@pytest.mark.asyncio
async def test_create_image_download_jobs_missing_images(tv_matcher: TVMatcher) -> None:
    """Test creation of image download jobs with missing image paths"""
    # Setup
    entity_id = uuid4()

    # Create test data with missing image paths
    tv_show_dto = TVShowDTO(
        tmdb_id=12345,
        title="Test Show",
        overview="Test overview",
        poster_path=None,  # Missing poster
        backdrop_path=None,  # Missing backdrop
        year=2020,
    )

    target_season = Mock(spec=TVSeason)
    target_season.poster_path = None  # Missing season poster

    target_episode = Mock(spec=TVEpisode)
    target_episode.still_path = None  # Missing episode still

    # Call the method
    result = tv_matcher._create_image_download_jobs(
        entity_id, tv_show_dto, target_season, target_episode
    )

    # Verify results
    assert isinstance(result, list)
    assert len(result) == 0  # Should have no jobs since all image paths are missing


@pytest.mark.asyncio
async def test_process_target_episode_success(
    tv_matcher: TVMatcher,
    mock_tv_show_data: Dict[str, Any],
    mock_season_data: Dict[str, Any],
    mock_episode_data: Dict[str, Any],
) -> None:
    """Test successful processing of target episode"""
    # Setup
    tmdb_id = 12345
    season_number = 1
    episode_number = 2
    tv_show_id = uuid4()
    file_id = uuid4()
    entity_id = uuid4()

    # Create TV show DTO
    tv_show_dto = TVShowDTO(
        tmdb_id=tmdb_id,
        title="Test Show",
        overview="Test overview",
        poster_path="/path/to/poster.jpg",
        backdrop_path="/path/to/backdrop.jpg",
        year=2020,
    )

    # Create mock season and episode
    mock_season = Mock(spec=TVSeason)
    mock_season.id = uuid4()

    mock_episode = Mock(spec=TVEpisode)
    mock_episode.id = uuid4()

    # Mock the fetch_season_details method
    with patch.object(
        tv_matcher, "_fetch_season_details", return_value=mock_season_data
    ) as mock_fetch_season:
        # Mock the find_target_season_and_episode method
        with patch.object(
            tv_matcher,
            "_find_target_season_and_episode",
            return_value=(mock_season, mock_episode, file_id),
        ) as mock_find:
            # Mock the fetch_episode_details method
            with patch.object(
                tv_matcher, "_fetch_episode_details", return_value=mock_episode_data
            ) as mock_fetch_episode:
                # Mock the create_episode_entity method
                with patch.object(
                    tv_matcher, "_create_episode_entity", return_value=entity_id
                ) as mock_create_entity:
                    # Call the method
                    result = await tv_matcher._process_target_episode(
                        tmdb_id,
                        season_number,
                        episode_number,
                        tv_show_id,
                        tv_show_dto,
                        file_id,
                    )

                    # Verify results
                    assert result is not None
                    result_entity_id, result_season, result_episode = result
                    assert result_entity_id == entity_id
                    assert result_season is mock_season
                    assert result_episode is mock_episode

                    # Verify method calls
                    mock_fetch_season.assert_called_once_with(tmdb_id, season_number)
                    mock_find.assert_called_once_with(
                        tv_show_id, season_number, episode_number, file_id
                    )
                    mock_fetch_episode.assert_called_once_with(
                        tmdb_id, season_number, episode_number
                    )
                    mock_create_entity.assert_called_once()


@pytest.mark.asyncio
async def test_process_target_episode_fetch_season_failure(
    tv_matcher: TVMatcher,
) -> None:
    """Test handling of season fetch failure in process_target_episode"""
    # Setup
    tmdb_id = 12345
    season_number = 1
    episode_number = 2
    tv_show_id = uuid4()
    file_id = uuid4()

    # Create TV show DTO
    tv_show_dto = TVShowDTO(
        tmdb_id=tmdb_id,
        title="Test Show",
        overview="Test overview",
        poster_path="/path/to/poster.jpg",
        backdrop_path="/path/to/backdrop.jpg",
        year=2020,
    )

    # Mock the fetch_season_details method to return empty dict (failure)
    with patch.object(
        tv_matcher, "_fetch_season_details", return_value={}
    ) as mock_fetch_season:
        # Call the method
        result = await tv_matcher._process_target_episode(
            tmdb_id, season_number, episode_number, tv_show_id, tv_show_dto, file_id
        )

        # Verify results
        assert result is None

        # Verify method calls
        mock_fetch_season.assert_called_once_with(tmdb_id, season_number)

        # Verify logger was called
        tv_matcher.logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_process_target_episode_find_target_failure(
    tv_matcher: TVMatcher, mock_season_data: Dict[str, Any]
) -> None:
    """Test handling of find_target_season_and_episode failure"""
    # Setup
    tmdb_id = 12345
    season_number = 1
    episode_number = 2
    tv_show_id = uuid4()
    file_id = uuid4()

    # Create TV show DTO
    tv_show_dto = TVShowDTO(
        tmdb_id=tmdb_id,
        title="Test Show",
        overview="Test overview",
        poster_path="/path/to/poster.jpg",
        backdrop_path="/path/to/backdrop.jpg",
        year=2020,
    )

    # Mock the fetch_season_details method
    with patch.object(
        tv_matcher, "_fetch_season_details", return_value=mock_season_data
    ) as mock_fetch_season:
        # Mock the find_target_season_and_episode method to return None (failure)
        with patch.object(
            tv_matcher, "_find_target_season_and_episode", return_value=None
        ) as mock_find:
            # Call the method
            result = await tv_matcher._process_target_episode(
                tmdb_id, season_number, episode_number, tv_show_id, tv_show_dto, file_id
            )

            # Verify results
            assert result is None

            # Verify method calls
            mock_fetch_season.assert_called_once_with(tmdb_id, season_number)
            mock_find.assert_called_once_with(
                tv_show_id, season_number, episode_number, file_id
            )


@pytest.mark.asyncio
async def test_process_target_episode_create_entity_failure(
    tv_matcher: TVMatcher,
    mock_season_data: Dict[str, Any],
    mock_episode_data: Dict[str, Any],
) -> None:
    """Test handling of create_episode_entity failure"""
    # Setup
    tmdb_id = 12345
    season_number = 1
    episode_number = 2
    tv_show_id = uuid4()
    file_id = uuid4()

    # Create TV show DTO
    tv_show_dto = TVShowDTO(
        tmdb_id=tmdb_id,
        title="Test Show",
        overview="Test overview",
        poster_path="/path/to/poster.jpg",
        backdrop_path="/path/to/backdrop.jpg",
        year=2020,
    )

    # Create mock season and episode
    mock_season = Mock(spec=TVSeason)
    mock_season.id = uuid4()

    mock_episode = Mock(spec=TVEpisode)
    mock_episode.id = uuid4()

    # Mock the fetch_season_details method
    with patch.object(
        tv_matcher, "_fetch_season_details", return_value=mock_season_data
    ) as mock_fetch_season:
        # Mock the find_target_season_and_episode method
        with patch.object(
            tv_matcher,
            "_find_target_season_and_episode",
            return_value=(mock_season, mock_episode, file_id),
        ) as mock_find:
            # Mock the fetch_episode_details method
            with patch.object(
                tv_matcher, "_fetch_episode_details", return_value=mock_episode_data
            ) as mock_fetch_episode:
                # Mock the create_episode_entity method to return None (failure)
                with patch.object(
                    tv_matcher, "_create_episode_entity", return_value=None
                ) as mock_create_entity:
                    # Call the method
                    result = await tv_matcher._process_target_episode(
                        tmdb_id,
                        season_number,
                        episode_number,
                        tv_show_id,
                        tv_show_dto,
                        file_id,
                    )

                    # Verify results
                    assert result is None

                    # Verify method calls
                    mock_fetch_season.assert_called_once_with(tmdb_id, season_number)
                    mock_find.assert_called_once_with(
                        tv_show_id, season_number, episode_number, file_id
                    )
                    mock_fetch_episode.assert_called_once_with(
                        tmdb_id, season_number, episode_number
                    )
                    mock_create_entity.assert_called_once()


@pytest.mark.asyncio
async def test_execute_success(
    tv_matcher: TVMatcher,
    valid_params: TvMatcherParams,
    mock_tv_show_data: Dict[str, Any],
    mock_season_data: Dict[str, Any],
    mock_episode_data: Dict[str, Any],
) -> None:
    """Test successful execution of the worker"""
    # Setup
    tv_show_id = uuid4()
    entity_id = uuid4()

    # Create mock season and episode
    mock_season = Mock(spec=TVSeason)
    mock_season.id = uuid4()
    mock_season.poster_path = "/path/to/season_poster.jpg"

    mock_episode = Mock(spec=TVEpisode)
    mock_episode.id = uuid4()
    mock_episode.still_path = "/path/to/episode_still.jpg"

    # Mock the validate_parameters method
    with patch.object(
        tv_matcher, "_validate_parameters", return_value=valid_params
    ) as mock_validate:
        # Mock the process_tv_show method
        with patch.object(
            tv_matcher,
            "_process_tv_show",
            return_value=(
                TVShowDTO(
                    tmdb_id=12345,
                    title="Test Show",
                    overview="Test overview",
                    poster_path="/path/to/poster.jpg",
                    backdrop_path="/path/to/backdrop.jpg",
                    year=2020,
                ),
                tv_show_id,
                mock_tv_show_data,
            ),
        ) as mock_process_show:
            # Mock the process_all_seasons_and_episodes method
            with patch.object(
                tv_matcher, "_process_all_seasons_and_episodes", return_value=True
            ) as mock_process_all:
                # Mock the process_target_episode method
                with patch.object(
                    tv_matcher,
                    "_process_target_episode",
                    return_value=(entity_id, mock_season, mock_episode),
                ) as mock_process_target:
                    # Mock the create_image_download_jobs method
                    with patch.object(
                        tv_matcher,
                        "_create_image_download_jobs",
                        return_value=[
                            ChildJobRequest(
                                job_type=JobType.IMAGE_DOWNLOADER,
                                params=ImageDownloaderParams(
                                    image_url="/path/to/poster.jpg", entity_id=entity_id
                                ),
                            )
                        ],
                    ) as mock_create_jobs:
                        # Call the method
                        result = await tv_matcher.execute(valid_params)

                        # Verify results
                        assert isinstance(result, list)
                        assert len(result) == 1
                        assert isinstance(result[0], ChildJobRequest)


@pytest.mark.asyncio
async def test_execute_invalid_parameters(tv_matcher: TVMatcher) -> None:
    """Test execution with invalid parameters"""
    # Mock the validate_parameters method to raise ValueError
    with patch.object(
        tv_matcher,
        "_validate_parameters",
        side_effect=ValueError("Parameters must be of type TvMatcherParams"),
    ):
        # Call the method with an invalid parameter (not None)
        invalid_params = {"tmdb_id": 12345}  # Not a TvMatcherParams object

        # The execute method should handle the ValueError internally and return an empty list
        result = await tv_matcher.execute(invalid_params)  # type: ignore

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 0


@pytest.mark.asyncio
async def test_execute_with_no_parameters(tv_matcher: TVMatcher) -> None:
    """Test execution with no parameters"""
    # Mock the validate_parameters method to handle None parameters
    with patch.object(
        tv_matcher, "_validate_parameters", return_value=None
    ) as mock_validate:
        # Call the method with None
        result = await tv_matcher.execute(None)

        # Verify results
        assert isinstance(result, list)
        assert len(result) == 0
        mock_validate.assert_called_once_with(None)
