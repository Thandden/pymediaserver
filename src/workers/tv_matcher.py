from datetime import datetime, date
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select

from src.common.config import config
from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    ChildJobRequest,
    EntityDTO,
    JobType,
    TVEpisodeDTO,
    TVSeasonDTO,
    TVShowDTO,
    TvMatcherParams,
    NoChildJob,
)
from src.common.http_client import AsyncHttpClient
from src.common.logger import Logger
from src.common.models import Entity, File, TVEpisode, TVSeason, TVShow
from src.common.system_types import EntityType, MetadataStatus
from src.workers.base import T_JobParams, Worker


class TVMatcher(Worker):
    """Worker for fetching and inserting detailed TV show, season, and episode information from TMDB."""

    TMDB_API_URL: str = "https://api.themoviedb.org/3"

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        """
        Initialize the TV matcher worker.

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
        Execute the TV matching process.

        Args:
            parameters: Job parameters containing TMDB show ID, season and episode numbers

        Returns:
            Empty list as no child jobs are spawned

        Raises:
            ValueError: If parameters are not of the correct type
        """
        if not isinstance(parameters, TvMatcherParams):
            raise ValueError("Parameters must be of type TvMatcherParams")

        tmdb_id = parameters.tmdb_id
        season_number = parameters.season_number
        episode_number = parameters.episode_number

        # Fetch TV show details from TMDB
        show_details = await self._fetch_tv_show_details(tmdb_id)
        if not show_details:
            if self.logger:
                self.logger.error(
                    f"Could not fetch TV show details for TMDB ID: {tmdb_id}"
                )
            return []

        # Create TVShowDTO from TMDB data
        tv_show_dto = self._create_tv_show_dto(show_details)

        # Insert TV show into database (or get existing)
        tv_show_id = await self._insert_tv_show(tv_show_dto)
        if not tv_show_id:
            if self.logger:
                self.logger.error(f"Failed to insert TV show with TMDB ID: {tmdb_id}")
            return []

        # Process all seasons data for the show
        seasons_data = show_details.get("seasons", [])

        for season_data in seasons_data:
            current_season_number = season_data.get("season_number", 0)

            # Skip special seasons (like season 0) if needed
            if current_season_number < 0:
                continue

            # Fetch detailed season information
            detailed_season = await self._fetch_season_details(
                tmdb_id, current_season_number
            )
            if not detailed_season:
                if self.logger:
                    self.logger.warning(
                        f"Could not fetch season {current_season_number} details for show ID: {tmdb_id}"
                    )
                continue

            # Create TVSeasonDTO from TMDB data
            tv_season_dto = self._create_tv_season_dto(detailed_season, tv_show_id)

            # Insert TV season into database
            tv_season_id = await self._insert_tv_season(tv_season_dto)
            if not tv_season_id:
                if self.logger:
                    self.logger.error(
                        f"Failed to insert season {current_season_number} for show ID: {tmdb_id}"
                    )
                continue

            # Process all episodes for this season
            episodes_data = detailed_season.get("episodes", [])

            for episode_data in episodes_data:
                current_episode_number = episode_data.get("episode_number", 0)

                # Create TVEpisodeDTO from TMDB data
                tv_episode_dto = self._create_tv_episode_dto(episode_data, tv_season_id)

                # Insert TV episode into database
                _ = await self._insert_tv_episode(tv_episode_dto)

        # Now that we've inserted all seasons and episodes, handle the specific episode for this file

        # Get the target season
        target_season_details = await self._fetch_season_details(tmdb_id, season_number)
        if not target_season_details:
            if self.logger:
                self.logger.error(
                    f"Could not fetch target season {season_number} details for show ID: {tmdb_id}"
                )
            return []

        # Find the season ID for our target season
        season_stmt = select(TVSeason).where(
            (TVSeason.show_id == tv_show_id) & (TVSeason.season_number == season_number)
        )

        async for session in self.db_session.get_session():
            try:
                result = await session.execute(season_stmt)
                target_season = result.scalar_one_or_none()

                if not target_season:
                    if self.logger:
                        self.logger.error(
                            f"Target season {season_number} not found in database"
                        )
                    return []

                # Find the episode ID for our target episode
                episode_stmt = select(TVEpisode).where(
                    (TVEpisode.season_id == target_season.id)
                    & (TVEpisode.episode_number == episode_number)
                )

                result = await session.execute(episode_stmt)
                target_episode = result.scalar_one_or_none()

                if not target_episode:
                    if self.logger:
                        self.logger.error(
                            f"Target episode {episode_number} not found in database"
                        )
                    return []

                # Fetch detailed episode information for matched_data
                episode_details = await self._fetch_episode_details(
                    tmdb_id, season_number, episode_number
                )

                # Create entity for this episode - now including tv_show_id
                entity_dto = EntityDTO(
                    file_id=parameters.file_id,
                    entity_type=EntityType.TV_EPISODE,
                    tv_show_id=tv_show_id,  # Add show ID to the entity
                    tv_episode_id=target_episode.id,
                    matched_data=episode_details,
                    metadata_status=MetadataStatus.CONFIRMED,
                )

                entity_id = await self._insert_entity(entity_dto)
                if entity_id:
                    if self.logger:
                        self.logger.info(
                            f"Successfully matched and inserted episode {episode_number} of season {season_number} "
                            f"for TV show: {tv_show_dto.title} (TMDB ID: {tmdb_id})"
                        )
                else:
                    if self.logger:
                        self.logger.error(
                            f"Failed to create entity for episode {episode_number} of season {season_number} "
                            f"for TV show: {tv_show_dto.title} (TMDB ID: {tmdb_id})"
                        )

                return []

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error processing target episode: {str(e)}")
                return []

        return []

    async def _fetch_tv_show_details(self, tmdb_id: int) -> dict[str, Any]:
        """
        Fetch detailed TV show information from TMDB API.

        Args:
            tmdb_id: TMDB TV show ID

        Returns:
            dict containing TV show details
        """
        endpoint = f"{self.TMDB_API_URL}/tv/{tmdb_id}"
        params: dict[str, Any] = {
            "api_key": config.TMDB_API_KEY,
            "append_to_response": "credits,keywords,videos,images",
        }

        async with self.http_client:
            try:
                show_data = await self.http_client.fetch_json(endpoint, params)
                if not show_data:
                    return {}
                return show_data
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error fetching TV show details: {str(e)}")
                return {}

    async def _fetch_season_details(
        self, tmdb_id: int, season_number: int
    ) -> dict[str, Any]:
        """
        Fetch detailed season information from TMDB API.

        Args:
            tmdb_id: TMDB TV show ID
            season_number: Season number

        Returns:
            dict containing season details
        """
        endpoint = f"{self.TMDB_API_URL}/tv/{tmdb_id}/season/{season_number}"
        params: dict[str, Any] = {"api_key": config.TMDB_API_KEY}

        async with self.http_client:
            try:
                season_data = await self.http_client.fetch_json(endpoint, params)
                if not season_data:
                    return {}
                return season_data
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error fetching season details: {str(e)}")
                return {}

    async def _fetch_episode_details(
        self, tmdb_id: int, season_number: int, episode_number: int
    ) -> dict[str, Any]:
        """
        Fetch detailed episode information from TMDB API.

        Args:
            tmdb_id: TMDB TV show ID
            season_number: Season number
            episode_number: Episode number

        Returns:
            dict containing episode details
        """
        endpoint = f"{self.TMDB_API_URL}/tv/{tmdb_id}/season/{season_number}/episode/{episode_number}"
        params: dict[str, Any] = {
            "api_key": config.TMDB_API_KEY,
            "append_to_response": "credits,images,videos",
        }

        async with self.http_client:
            try:
                episode_data = await self.http_client.fetch_json(endpoint, params)
                if not episode_data:
                    return {}
                return episode_data
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error fetching episode details: {str(e)}")
                return {}

    def _create_tv_show_dto(self, show_data: dict[str, Any]) -> TVShowDTO:
        """
        Create TVShowDTO from TMDB response data.

        Args:
            show_data: TV show data from TMDB

        Returns:
            TVShowDTO instance
        """
        # Extract year from first_air_date if available
        year: Optional[int] = None

        if show_data.get("first_air_date"):
            try:
                air_date = date.fromisoformat(show_data["first_air_date"])
                year = air_date.year
            except (ValueError, IndexError):
                pass

        return TVShowDTO(
            tmdb_id=show_data.get("id"),
            title=show_data.get("name", "Unknown Title"),
            overview=show_data.get("overview"),
            poster_path=show_data.get("poster_path"),
            backdrop_path=show_data.get("backdrop_path"),
            year=year,
        )

    def _create_tv_season_dto(
        self, season_data: dict[str, Any], show_id: UUID
    ) -> TVSeasonDTO:
        """
        Create TVSeasonDTO from TMDB response data.

        Args:
            season_data: Season data from TMDB
            show_id: UUID of the TV show

        Returns:
            TVSeasonDTO instance
        """
        # Extract year from air_date if available
        year: Optional[int] = None

        if season_data.get("air_date"):
            try:
                air_date = date.fromisoformat(season_data["air_date"])
                year = air_date.year
            except (ValueError, IndexError):
                pass

        return TVSeasonDTO(
            show_id=show_id,
            season_number=season_data.get("season_number", 0),
            title=season_data.get("name"),
            overview=season_data.get("overview"),
            poster_path=season_data.get("poster_path"),
            year=year,
        )

    def _create_tv_episode_dto(
        self, episode_data: dict[str, Any], season_id: UUID
    ) -> TVEpisodeDTO:
        """
        Create TVEpisodeDTO from TMDB response data.

        Args:
            episode_data: Episode data from TMDB
            season_id: UUID of the TV season

        Returns:
            TVEpisodeDTO instance
        """
        # Parse air_date if available
        air_date: Optional[date] = None

        if episode_data.get("air_date"):
            try:
                air_date = date.fromisoformat(episode_data["air_date"])
            except ValueError:
                pass

        return TVEpisodeDTO(
            season_id=season_id,
            episode_number=episode_data.get("episode_number", 0),
            title=episode_data.get("name"),
            overview=episode_data.get("overview"),
            still_path=episode_data.get("still_path"),
            air_date=air_date,
        )

    async def _insert_tv_show(self, tv_show_dto: TVShowDTO) -> Optional[UUID]:
        """
        Insert TV show into the database or get existing one.

        Args:
            tv_show_dto: TV show data transfer object

        Returns:
            UUID of the inserted/existing TV show or None on failure
        """
        async for session in self.db_session.get_session():
            try:
                # Check if show already exists with this TMDB ID
                if tv_show_dto.tmdb_id:
                    stmt = select(TVShow).where(TVShow.tmdb_id == tv_show_dto.tmdb_id)
                    result = await session.execute(stmt)
                    existing_show = result.scalar_one_or_none()

                    if existing_show:
                        return existing_show.id

                # Create new TV show
                tv_show = TVShow(**tv_show_dto.model_dump())

                session.add(tv_show)
                await session.commit()
                # Refresh the show to get the id
                await session.refresh(tv_show)
                return tv_show.id
            except Exception as e:
                await session.rollback()
                if self.logger:
                    self.logger.error(f"Error inserting TV show: {str(e)}")
                return None

    async def _insert_tv_season(self, tv_season_dto: TVSeasonDTO) -> Optional[UUID]:
        """
        Insert TV season into the database or get existing one.

        Args:
            tv_season_dto: TV season data transfer object

        Returns:
            UUID of the inserted/existing TV season or None on failure
        """
        async for session in self.db_session.get_session():
            try:
                # Check if season already exists for this show
                stmt = select(TVSeason).where(
                    (TVSeason.show_id == tv_season_dto.show_id)
                    & (TVSeason.season_number == tv_season_dto.season_number)
                )
                result = await session.execute(stmt)
                existing_season = result.scalar_one_or_none()

                if existing_season:
                    return existing_season.id

                # Create new TV season
                tv_season = TVSeason(**tv_season_dto.model_dump())

                session.add(tv_season)
                await session.commit()
                # Refresh the season to get the id
                await session.refresh(tv_season)
                return tv_season.id
            except Exception as e:
                await session.rollback()
                if self.logger:
                    self.logger.error(f"Error inserting TV season: {str(e)}")
                return None

    async def _insert_tv_episode(self, tv_episode_dto: TVEpisodeDTO) -> Optional[UUID]:
        """
        Insert TV episode into the database or get existing one.

        Args:
            tv_episode_dto: TV episode data transfer object

        Returns:
            UUID of the inserted/existing TV episode or None on failure
        """
        async for session in self.db_session.get_session():
            try:
                # Check if episode already exists for this season
                stmt = select(TVEpisode).where(
                    (TVEpisode.season_id == tv_episode_dto.season_id)
                    & (TVEpisode.episode_number == tv_episode_dto.episode_number)
                )
                result = await session.execute(stmt)
                existing_episode = result.scalar_one_or_none()

                if existing_episode:
                    return existing_episode.id

                # Create new TV episode
                tv_episode = TVEpisode(**tv_episode_dto.model_dump())

                session.add(tv_episode)
                await session.commit()
                # Refresh the episode to get the id
                await session.refresh(tv_episode)
                return tv_episode.id
            except Exception as e:
                await session.rollback()
                if self.logger:
                    self.logger.error(f"Error inserting TV episode: {str(e)}")
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
                    # Refresh the entity to get the id
                    await session.refresh(entity)
                    entity_id = str(entity.id)

                return entity_id
            except Exception as e:
                await session.rollback()
                if self.logger:
                    self.logger.error(f"Error inserting/updating entity: {str(e)}")
                return None
