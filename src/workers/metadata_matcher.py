from typing import Any, Optional, cast
from uuid import UUID

from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    ChildJobRequest,
    EntityDTO,
    JobType,
    MatchedData,
    MetadataMatcherParams,
    MovieMatcherParams,
    TvMatcherParams,
)
from src.common.http_client import AsyncHttpClient
from src.common.logger import Logger
from src.common.models import Entity
from src.common.system_types import MediaType
from src.workers.base import T_JobParams, Worker
from src.common.config import config


class MetadataMatcher(Worker):
    """Worker for matching media metadata using external APIs."""

    TMDB_API_URL: str = "https://api.themoviedb.org/3"

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        """
        Initialize the metadata matcher worker.

        Args:
            db_session: Database session for database operations
            api_key: TMDB API key
            logger: Optional logger instance
        """
        super().__init__(db_session, logger)
        self.http_client = AsyncHttpClient(retries=3, delay=2)

    async def execute(
        self, parameters: Optional[T_JobParams] = None
    ) -> list[ChildJobRequest]:
        """
        Execute the metadata matching process.

        Args:
            parameters: Job parameters containing media information to match

        Returns:
            List of child job requests for further processing

        Raises:
            ValueError: If parameters are not of the correct type
        """
        if not isinstance(parameters, MetadataMatcherParams):
            raise ValueError("Parameters must be of type MetadataMatcherParams")

        matched_data = parameters.matched_data

        # Search for the media in TMDB based on media type
        if matched_data.media_type == MediaType.MOVIE:
            search_results = await self._search_movie(matched_data)
        elif matched_data.media_type == MediaType.TV:
            search_results = await self._search_tv(matched_data)
        else:
            if self.logger:
                self.logger.error(f"Unsupported media type: {matched_data.media_type}")
            return []

        # If no results found, return empty child job list
        if not search_results:
            if self.logger:
                self.logger.info(f"No matches found for {matched_data.title}")
            return []

        # Take the first result as the match
        first_match = search_results[0]

        # Create appropriate child job based on media type
        child_jobs: list[ChildJobRequest] = []
        if matched_data.media_type == MediaType.MOVIE:
            child_jobs.append(
                ChildJobRequest(
                    job_type=JobType.MOVIE_MATCHER,
                    params=MovieMatcherParams(
                        tmdb_id=first_match["id"], file_id=parameters.file_id
                    ),
                )
            )
        elif matched_data.media_type == MediaType.TV:
            child_jobs.append(
                ChildJobRequest(
                    job_type=JobType.TV_MATCHER,
                    params=TvMatcherParams(
                        tmdb_id=first_match["id"],
                        file_id=parameters.file_id,
                        season_number=matched_data.season_number,
                        episode_number=matched_data.episode_number,
                    ),
                )
            )

        if self.logger:
            self.logger.info(
                f"Successfully matched {matched_data.title} to TMDB ID: {first_match['id']}"
            )

        return child_jobs

    async def _search_movie(self, matched_data: MatchedData) -> list[dict[str, Any]]:
        """
        Search for a movie in the TMDB API.

        Args:
            matched_data: Data containing title and year for search

        Returns:
            List of matching movie results
        """
        endpoint = f"{self.TMDB_API_URL}/search/movie"
        params: dict[str, Any] = {
            "api_key": config.TMDB_API_KEY,
            "query": matched_data.title,
        }

        if matched_data.year:
            params["year"] = matched_data.year

        async with self.http_client:
            response_data = await self.http_client.fetch_json(endpoint, params)

        if not response_data or "results" not in response_data:
            return []

        return cast(list[dict[str, Any]], response_data["results"])

    async def _search_tv(self, matched_data: MatchedData) -> list[dict[str, Any]]:
        """
        Search for a TV show in the TMDB API.

        Args:
            matched_data: Data containing title and year for search

        Returns:
            List of matching TV show results
        """
        endpoint = f"{self.TMDB_API_URL}/search/tv"
        params: dict[str, Any] = {
            "api_key": config.TMDB_API_KEY,
            "query": matched_data.title,
        }

        if matched_data.year:
            params["first_air_date_year"] = matched_data.year

        async with self.http_client:
            response_data = await self.http_client.fetch_json(endpoint, params)

        if not response_data or "results" not in response_data:
            return []

        return cast(list[dict[str, Any]], response_data["results"])

    async def _insert_entity(self, entity: EntityDTO) -> Optional[UUID]:
        """Insert movie or tv show into the database

        Args:
            entity (EntityDTO): Entity to insert

        Returns:
            uuid.UUID: ID of the inserted entity
        """
        async for session in self.db_session.get_session():
            session.add(Entity(**entity.model_dump()))
            # Flush to generate the ID without committing
            await session.flush()
            if entity.id is None:
                raise ValueError("Failed to generate entity ID")
            return entity.id
