from datetime import datetime, date
from typing import Any, Optional, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.config import config
from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    ChildJobRequest,
    EntityDTO,
    JobType,
    MovieDTO,
    MovieMatcherParams,
    NoChildJob,
)
from src.common.http_client import AsyncHttpClient
from src.common.logger import Logger
from src.common.models import Entity, File, Movie
from src.common.system_types import EntityType, MetadataStatus
from src.workers.base import T_JobParams, Worker


class MovieMatcher(Worker):
    """Worker for fetching and inserting detailed movie information from TMDB."""

    TMDB_API_URL: str = "https://api.themoviedb.org/3"

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        """
        Initialize the movie matcher worker.

        Args:
            db_session: Database session for database operations
            logger: Optional logger instance
        """
        super().__init__(db_session, logger)
        self.http_client = AsyncHttpClient(retries=3, delay=2)

    async def execute(
        self, parameters: Optional[T_JobParams] = None
    ) -> list[ChildJobRequest] | NoChildJob:
        """
        Execute the movie matching process.

        Args:
            parameters: Job parameters containing TMDB movie ID

        Returns:
            Empty list as no child jobs are spawned

        Raises:
            ValueError: If parameters are not of the correct type
        """
        if not isinstance(parameters, MovieMatcherParams):
            raise ValueError("Parameters must be of type MovieMatcherParams")

        tmdb_id = parameters.tmdb_id

        # Fetch detailed movie information from TMDB
        movie_details = await self._fetch_movie_details(tmdb_id)
        if not movie_details:
            if self.logger:
                self.logger.error(
                    f"Could not fetch movie details for TMDB ID: {tmdb_id}"
                )
            return []

        # Create MovieDTO from TMDB data
        movie_dto = self._create_movie_dto(movie_details)

        # Insert movie into database
        movie_id = await self._insert_movie(movie_dto)
        if not movie_id:
            if self.logger:
                self.logger.error(f"Failed to insert movie with TMDB ID: {tmdb_id}")
            return []

        # Create entity for this movie
        entity_dto = EntityDTO(
            file_id=parameters.file_id,
            entity_type=EntityType.MOVIE,
            movie_id=movie_id,
            matched_data=movie_details,
            metadata_status=MetadataStatus.CONFIRMED,
        )

        entity_id = await self._insert_entity(entity_dto)
        if entity_id:
            if self.logger:
                self.logger.info(
                    f"Successfully matched and inserted movie: {movie_dto.title} (TMDB ID: {tmdb_id})"
                )
        else:
            if self.logger:
                self.logger.error(
                    f"Failed to create entity for movie: {movie_dto.title} (TMDB ID: {tmdb_id})"
                )

        return []

    async def _fetch_movie_details(self, tmdb_id: int) -> dict[str, Any]:
        """
        Fetch detailed movie information from TMDB API.

        Args:
            tmdb_id: TMDB movie ID

        Returns:
            dict containing movie details
        """
        endpoint = f"{self.TMDB_API_URL}/movie/{tmdb_id}"
        params: dict[str, Any] = {
            "api_key": config.TMDB_API_KEY,
            "append_to_response": "credits,keywords,videos,images",
        }

        async with self.http_client:
            try:
                movie_data = await self.http_client.fetch_json(endpoint, params)
                if not movie_data:
                    return {}
                return movie_data
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error fetching movie details: {str(e)}")
                return {}

    def _create_movie_dto(self, movie_data: dict[str, Any]) -> MovieDTO:
        """
        Create MovieDTO from TMDB response data.

        Args:
            movie_data: Movie data from TMDB

        Returns:
            MovieDTO instance
        """
        # Extract year from release date if available
        year: Optional[int] = None
        release_date: Optional[date] = None

        if movie_data.get("release_date"):
            try:
                release_date = date.fromisoformat(movie_data["release_date"])
                year = release_date.year
            except (ValueError, IndexError):
                pass

        return MovieDTO(
            tmdb_id=movie_data.get("id"),
            title=movie_data.get("title", "Unknown Title"),
            overview=movie_data.get("overview"),
            poster_path=movie_data.get("poster_path"),
            backdrop_path=movie_data.get("backdrop_path"),
            release_date=release_date,
            year=year,
        )

    async def _insert_movie(self, movie_dto: MovieDTO) -> Optional[UUID]:
        """
        Insert movie into the database.

        Args:
            movie_dto: Movie data transfer object

        Returns:
            UUID of the inserted movie or None on failure
        """
        async for session in self.db_session.get_session():
            try:
                movie = Movie(**movie_dto.model_dump())

                session.add(movie)
                await session.commit()
                # refresh the movie to get the id
                await session.refresh(movie)
                return movie.id
            except Exception as e:
                await session.rollback()
                if self.logger:
                    self.logger.error(f"Error inserting movie: {str(e)}")
                return None

    async def _insert_entity(self, entity_dto: EntityDTO) -> Optional[str]:
        """
        Insert or update entity in the database.

        Args:
            entity_dto: Entity data transfer object

        Returns:
            UUID of the inserted entity or None on failure
        """
        async for session in self.db_session.get_session():
            try:
                # Convert the DTO to dict and remove None values
                entity_data = {
                    k: v
                    for k, v in entity_dto.model_dump().items()
                    if v is not None and k != "id"  # exclude id to let DB generate it
                }

                # Check for existing entity
                stmt = select(Entity).where(Entity.file_id == entity_data["file_id"])
                result = await session.execute(stmt)
                existing_entity = result.scalar_one_or_none()

                if existing_entity:
                    # Update existing entity
                    for key, value in entity_data.items():
                        setattr(existing_entity, key, value)
                    entity_id = str(existing_entity.id)
                else:
                    # Create new entity
                    entity = Entity(**entity_data)
                    session.add(entity)
                    await session.commit()
                    # refresh the entity to get the id
                    await session.refresh(entity)
                    entity_id = str(entity.id)

                return entity_id
            except Exception as e:
                await session.rollback()
                if self.logger:
                    self.logger.error(f"Error inserting/updating entity: {str(e)}")
                return None

    async def _find_pending_entity(self, tmdb_id: int) -> Optional[dict[str, Any]]:
        """
        Find pending entity that needs to be associated with this movie.

        Args:
            tmdb_id: TMDB movie ID

        Returns:
            dictionary with entity info or None if not found
        """
        async for session in self.db_session.get_session():
            try:
                # Using select() instead of query() for async compatibility
                stmt = (
                    select(Entity.id, Entity.file_id)
                    .join(File, Entity.file_id == File.id)
                    .where(Entity.metadata_status == MetadataStatus.PENDING)
                    .where(Entity.matched_data.contains({"id": tmdb_id}))
                )

                result = await session.execute(stmt)
                entity = result.mappings().first()

                if entity:
                    return cast(dict[str, Any], dict(entity))
                return None
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error finding pending entity: {str(e)}")
                return None
