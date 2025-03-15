from __future__ import annotations

import os
import shutil
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, cast
import uuid

from sqlalchemy import select, and_
from sqlalchemy.orm import joinedload

from src.common.base_service import Service, T_ServiceParams
from src.common.db import AsyncDatabaseSession
from src.common.logger import Logger
from src.common.models import PlaybackSession, TranscodingSession
from src.common.config import config
from src.common.types import TranscodeState
from src.common.dto import CleanupParams


class Cleanup(Service[CleanupParams]):
    """Worker implementation for cleaning up transcoding sessions and their files."""

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        """
        Initialize the cleanup worker.

        Args:
            db_session: Database session
            logger: Optional logger
        """
        super().__init__(db_session, logger)
        self.transcode_directory = config.TRANSCODE_DIRECTORY
        self.cleanup_interval = 3600  # Default to 1 hour between cleanup runs

    async def start(self, parameters: Optional[CleanupParams] = None) -> None:
        """
        Start the cleanup service.

        Args:
            parameters: Job parameters (optional)
        """
        if self.logger:
            self.logger.info("Starting transcoding session cleanup service")

        # If parameters are provided, check for cleanup_interval
        if parameters is not None:
            self.cleanup_interval = parameters.cleanup_interval

        if self.logger:
            self.logger.info(f"Cleanup interval set to {self.cleanup_interval} seconds")

    async def process_iteration(
        self, parameters: Optional[CleanupParams] = None
    ) -> None:
        """
        Process a single cleanup iteration.

        Args:
            parameters: Job parameters (optional)
        """
        if self.logger:
            self.logger.info("Starting cleanup cycle")

        try:
            # Get current time for comparison
            now = datetime.now(timezone.utc)

            # Cleanup completed sessions (3 hours after completion)
            await self._cleanup_completed_sessions(now)

            # Cleanup inactive/abandoned sessions (24 hours of inactivity)
            await self._cleanup_inactive_sessions(now)

            # Cleanup failed transcodes (12 hours after error)
            await self._cleanup_failed_transcodes(now)

            # Cleanup pending transcodes that never started (6 hours since creation)
            await self._cleanup_stalled_pending_transcodes(now)

            if self.logger:
                self.logger.info("Cleanup cycle completed")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in cleanup iteration: {e}")
            raise

    async def _cleanup_completed_sessions(self, now: datetime) -> None:
        """
        Cleanup completed playback sessions.

        Args:
            now: Current datetime for comparison
        """
        if self.logger:
            self.logger.debug("Cleaning up completed playback sessions")

        # Calculate the cutoff time (3 hours ago)
        cutoff_time = now - timedelta(hours=3)

        try:
            async for session in self.db_session.get_session():
                # Find completed playback sessions with transcoding sessions
                query = (
                    select(PlaybackSession)
                    .options(joinedload(PlaybackSession.transcoding_session))
                    .join(TranscodingSession)
                    .where(
                        and_(
                            PlaybackSession.watched_percentage
                            >= 0.9,  # is_completed property
                            PlaybackSession.last_heartbeat <= cutoff_time,
                        )
                    )
                )

                result = await session.execute(query)
                playback_sessions = result.scalars().all()

                for playback_session in playback_sessions:
                    if playback_session.transcoding_session:
                        # Delete the transcoding files
                        await self._delete_transcode_files(
                            playback_session.transcoding_session.id
                        )

                        # Delete the transcoding session from the database
                        await session.delete(playback_session.transcoding_session)

                        if self.logger:
                            self.logger.info(
                                f"Cleaned up completed session: {playback_session.id}, "
                                f"transcode: {playback_session.transcoding_session.id}"
                            )

                await session.commit()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error cleaning up completed sessions: {e}")

    async def _cleanup_inactive_sessions(self, now: datetime) -> None:
        """
        Cleanup inactive or abandoned transcoding sessions.

        Args:
            now: Current datetime for comparison
        """
        if self.logger:
            self.logger.debug("Cleaning up inactive transcoding sessions")

        # Calculate the cutoff time (24 hours ago)
        cutoff_time = now - timedelta(hours=24)

        try:
            async for session in self.db_session.get_session():
                # Find inactive transcoding sessions
                query = (
                    select(TranscodingSession)
                    .options(joinedload(TranscodingSession.playback_session))
                    .where(TranscodingSession.last_activity <= cutoff_time)
                )

                result = await session.execute(query)
                transcoding_sessions = result.scalars().all()

                for transcoding_session in transcoding_sessions:
                    # Delete the transcoding files
                    await self._delete_transcode_files(transcoding_session.id)

                    # Delete the transcoding session from the database
                    await session.delete(transcoding_session)

                    if self.logger:
                        self.logger.info(
                            f"Cleaned up inactive transcoding session: {transcoding_session.id}, "
                            f"last activity: {transcoding_session.last_activity}"
                        )

                await session.commit()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error cleaning up inactive sessions: {e}")

    async def _cleanup_failed_transcodes(self, now: datetime) -> None:
        """
        Cleanup failed transcoding sessions.

        Args:
            now: Current datetime for comparison
        """
        if self.logger:
            self.logger.debug("Cleaning up failed transcoding sessions")

        # Calculate the cutoff time (12 hours ago)
        cutoff_time = now - timedelta(hours=12)

        try:
            async for session in self.db_session.get_session():
                # Find failed transcoding sessions
                query = select(TranscodingSession).where(
                    and_(
                        TranscodingSession.state == TranscodeState.ERROR,
                        TranscodingSession.last_activity <= cutoff_time,
                    )
                )

                result = await session.execute(query)
                transcoding_sessions = result.scalars().all()

                for transcoding_session in transcoding_sessions:
                    # Delete the transcoding files
                    await self._delete_transcode_files(transcoding_session.id)

                    # Delete the transcoding session from the database
                    await session.delete(transcoding_session)

                    if self.logger:
                        self.logger.info(
                            f"Cleaned up failed transcoding session: {transcoding_session.id}, "
                            f"error: {transcoding_session.error_message}"
                        )

                await session.commit()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error cleaning up failed transcodes: {e}")

    async def _cleanup_stalled_pending_transcodes(self, now: datetime) -> None:
        """
        Cleanup pending transcoding sessions that never started.

        Args:
            now: Current datetime for comparison
        """
        if self.logger:
            self.logger.debug("Cleaning up stalled pending transcoding sessions")

        # Calculate the cutoff time (6 hours ago)
        cutoff_time = now - timedelta(hours=6)

        try:
            async for session in self.db_session.get_session():
                # Find stalled pending transcoding sessions
                query = select(TranscodingSession).where(
                    and_(
                        TranscodingSession.state == TranscodeState.PENDING,
                        TranscodingSession.current_segment == 0,
                        TranscodingSession.created_at <= cutoff_time,
                    )
                )

                result = await session.execute(query)
                transcoding_sessions = result.scalars().all()

                for transcoding_session in transcoding_sessions:
                    # Delete the transcoding files
                    await self._delete_transcode_files(transcoding_session.id)

                    # Delete the transcoding session from the database
                    await session.delete(transcoding_session)

                    if self.logger:
                        self.logger.info(
                            f"Cleaned up stalled pending transcoding session: {transcoding_session.id}, "
                            f"created at: {transcoding_session.created_at}"
                        )

                await session.commit()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error cleaning up stalled pending transcodes: {e}")

    async def _delete_transcode_files(self, transcode_id: uuid.UUID) -> None:
        """
        Delete the transcoding files for a specific transcoding session.

        Args:
            transcode_id: ID of the transcoding session
        """
        transcode_path = os.path.join(self.transcode_directory, str(transcode_id))

        try:
            if os.path.exists(transcode_path):
                # Use shutil.rmtree to delete the directory and all its contents
                shutil.rmtree(transcode_path)

                if self.logger:
                    self.logger.debug(f"Deleted transcode directory: {transcode_path}")
            else:
                if self.logger:
                    self.logger.debug(
                        f"Transcode directory not found: {transcode_path}"
                    )
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error deleting transcode directory {transcode_path}: {e}"
                )
