from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import cast, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.db import AsyncDatabaseSession
from src.common.logger import Logger
from src.services.cleanup import Cleanup


class AsyncIteratorMock:
    """Mock for async iterator that yields a session."""

    def __init__(self, session: MagicMock):
        self.session = session
        self.called = False

    def __aiter__(self) -> AsyncIterator[MagicMock]:
        return self

    async def __anext__(self):
        if not self.called:
            self.called = True
            return self.session
        raise StopAsyncIteration


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock database session."""
    session = MagicMock(spec=AsyncSession)
    # Make all relevant methods async mocks
    session.execute = AsyncMock()
    session.delete = AsyncMock()  # This needs to be awaited in the code
    session.commit = AsyncMock()  # This needs to be awaited in the code

    # Set up the result chain for execute
    mock_result = AsyncMock()
    mock_result.scalars.return_value.all.return_value = []
    session.execute.return_value = mock_result

    return session


@pytest.fixture
def mock_db_session(mock_session: MagicMock) -> AsyncDatabaseSession:
    """Create a mock database session with context manager."""
    mock_db = AsyncMock(spec=AsyncDatabaseSession)

    # Create a function that returns an async iterator
    def get_session() -> AsyncIteratorMock:
        return AsyncIteratorMock(mock_session)

    mock_db.get_session = get_session
    return mock_db


@pytest.fixture
def mock_logger() -> Logger:
    """Create a mock logger."""
    return MagicMock(spec=Logger)


@pytest.fixture
def cleanup_service(
    mock_db_session: AsyncDatabaseSession, mock_logger: Logger
) -> Cleanup:
    """Create a cleanup service with mocked dependencies."""
    with patch("src.services.cleanup.config") as mock_config:
        mock_config.TRANSCODE_DIRECTORY = "/mock/transcode/dir"
        service = Cleanup(mock_db_session, mock_logger)
        return service


@pytest.mark.asyncio
async def test_execute_calls_all_cleanup_methods(cleanup_service: Cleanup) -> None:
    """Test that execute calls all cleanup methods."""
    # Mock all the cleanup methods
    with (
        patch.object(
            cleanup_service, "_cleanup_completed_sessions", new_callable=AsyncMock
        ) as mock_completed,
        patch.object(
            cleanup_service, "_cleanup_inactive_sessions", new_callable=AsyncMock
        ) as mock_inactive,
        patch.object(
            cleanup_service, "_cleanup_failed_transcodes", new_callable=AsyncMock
        ) as mock_failed,
        patch.object(
            cleanup_service,
            "_cleanup_stalled_pending_transcodes",
            new_callable=AsyncMock,
        ) as mock_stalled,
    ):
        # Call the process_iteration method which performs the cleanup
        await cleanup_service.process_iteration()

        # Assert all cleanup methods were called
        mock_completed.assert_called_once()
        mock_inactive.assert_called_once()
        mock_failed.assert_called_once()
        mock_stalled.assert_called_once()


@pytest.mark.asyncio
async def test_delete_transcode_files_existing_directory(
    cleanup_service: Cleanup,
) -> None:
    """Test deleting existing transcode files."""
    # Mock os.path.exists and shutil.rmtree
    with (
        patch("os.path.exists", return_value=True),
        patch("shutil.rmtree") as mock_rmtree,
    ):
        transcode_id = uuid.uuid4()
        await cleanup_service._delete_transcode_files(transcode_id)  # type: ignore

        # Assert shutil.rmtree was called with the correct path
        expected_path = os.path.join("/mock/transcode/dir", str(transcode_id))
        mock_rmtree.assert_called_once_with(expected_path)

        # Assert the debug log was called
        if cleanup_service.logger:
            cast(MagicMock, cleanup_service.logger).debug.assert_called_once_with(
                f"Deleted transcode directory: {expected_path}"
            )


@pytest.mark.asyncio
async def test_delete_transcode_files_nonexistent_directory(
    cleanup_service: Cleanup,
) -> None:
    """Test attempting to delete a nonexistent transcode directory."""
    # Mock os.path.exists to return False
    with patch("os.path.exists", return_value=False):
        transcode_id = uuid.uuid4()
        await cleanup_service._delete_transcode_files(transcode_id)  # type: ignore

        # Assert the debug log was called
        expected_path = os.path.join("/mock/transcode/dir", str(transcode_id))
        if cleanup_service.logger:
            cast(MagicMock, cleanup_service.logger).debug.assert_called_once_with(
                f"Transcode directory not found: {expected_path}"
            )


@pytest.mark.asyncio
async def test_delete_transcode_files_error(cleanup_service: Cleanup) -> None:
    """Test error handling when deleting transcode files."""
    # Mock os.path.exists to return True and shutil.rmtree to raise an exception
    with (
        patch("os.path.exists", return_value=True),
        patch("shutil.rmtree", side_effect=Exception("Test error")),
    ):
        transcode_id = uuid.uuid4()
        await cleanup_service._delete_transcode_files(transcode_id)  # type: ignore

        # Assert the error log was called
        if cleanup_service.logger:
            cast(MagicMock, cleanup_service.logger).error.assert_called_once()
            error_message = cast(MagicMock, cleanup_service.logger).error.call_args[0][
                0
            ]
            assert "Error deleting transcode directory" in error_message


@pytest.mark.asyncio
async def test_cleanup_completed_sessions_exception(
    cleanup_service: Cleanup, mock_session: MagicMock
) -> None:
    """Test error handling in cleanup of completed sessions."""
    # Mock session to raise an exception
    mock_session.execute.side_effect = Exception("Test database error")

    # Call the method
    now = datetime.now(timezone.utc)
    await cleanup_service._cleanup_completed_sessions(now)  # type: ignore

    # Assert the error was logged
    if cleanup_service.logger:
        cast(MagicMock, cleanup_service.logger).error.assert_called_once()
        error_message = cast(MagicMock, cleanup_service.logger).error.call_args[0][0]
        assert "Error cleaning up completed sessions" in error_message
