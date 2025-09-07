import asyncio
import os
import uuid
import hashlib
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, AsyncGenerator, cast, Type
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest
from sqlalchemy import select, Select
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    WatchDogParams,
    ChildJobRequest,
    FileDTO,
    FFProbeParams,
    FileMatcherParams,
)
from src.common.logger import Logger
from src.common.models import File
from src.common.system_types import JobType, MediaType
from src.services.watchdog import WatchDog, FileEventHandler


class MockAsyncSession:
    """Mock for AsyncSession to use in tests."""

    def __init__(self) -> None:
        self.add_calls: List[Any] = []
        self.execute_results: Dict[str, Any] = {}
        self.execute_calls: List[Any] = []
        self.commit_called = False
        self.rollback_called = False
        self.closed = False

    async def add(self, obj: Any) -> None:
        """Mock add method."""
        self.add_calls.append(obj)

    async def execute(self, statement: Any) -> Any:
        """Mock execute method."""
        self.execute_calls.append(statement)
        result = MagicMock()

        # If it's a select statement for File objects, return mock files
        if (
            hasattr(statement, "entity_description")
            and hasattr(statement.entity_description, "entity")
            and statement.entity_description.entity is File
        ):
            mock_result = self.execute_results.get("files", [])
            result.scalars.return_value.all.return_value = mock_result

        return result

    async def commit(self) -> None:
        """Mock commit method."""
        self.commit_called = True

    async def rollback(self) -> None:
        """Mock rollback method."""
        self.rollback_called = True

    async def close(self) -> None:
        """Mock close method."""
        self.closed = True

    def __await__(self) -> Any:
        async def _await_impl() -> Any:
            return self

        return _await_impl().__await__()


class MockAsyncDatabaseSession:
    """Mock for AsyncDatabaseSession to use in tests."""

    def __init__(self) -> None:
        self.session = MockAsyncSession()

    async def get_session(self) -> AsyncGenerator[MockAsyncSession, None]:
        """Mock get_session method."""
        yield self.session


@pytest.fixture
def mock_db_session() -> MockAsyncDatabaseSession:
    """Fixture for mock database session."""
    return MockAsyncDatabaseSession()


@pytest.fixture
def mock_logger() -> MagicMock:
    """Fixture for mock logger."""
    logger = MagicMock(spec=Logger)
    return logger


@pytest.fixture
def watchdog(
    mock_db_session: MockAsyncDatabaseSession, mock_logger: MagicMock
) -> WatchDog:
    """Fixture for WatchDog instance."""
    # Cast to AsyncDatabaseSession to satisfy type checker
    return WatchDog(cast(AsyncDatabaseSession, mock_db_session), mock_logger)


@pytest.fixture
def mock_observer() -> MagicMock:
    """Fixture for mock Observer."""
    observer = MagicMock()
    observer.start = MagicMock()
    observer.stop = MagicMock()
    observer.join = MagicMock()
    return observer


@pytest.fixture
def mock_file_event_handler() -> MagicMock:
    """Fixture for mock FileEventHandler."""
    handler = MagicMock(spec=FileEventHandler)
    return handler


@pytest.fixture
def watchdog_params() -> WatchDogParams:
    """Fixture for WatchDogParams."""
    return WatchDogParams(
        dir_path=Path("/test/path"),
        media_type=MediaType.MOVIE,
        file_extensions=[".mp4", ".mkv"],
    )


class TestFileEventHandler:
    """Tests for the FileEventHandler class."""

    @pytest.fixture
    def file_event_handler(self, mock_logger: MagicMock) -> FileEventHandler:
        """Fixture for FileEventHandler instance."""
        file_extensions = [".mp4", ".mkv"]
        known_files: Set[str] = set()
        known_hashes: Set[str] = set()
        new_files: Dict[str, Dict[str, Any]] = {}
        file_detected_event = threading.Event()

        return FileEventHandler(
            file_extensions=file_extensions,
            known_files=known_files,
            known_hashes=known_hashes,
            new_files=new_files,
            file_detected_event=file_detected_event,
            logger=mock_logger,
        )

    def test_init(self, file_event_handler: FileEventHandler) -> None:
        """Test initialization of FileEventHandler."""
        assert file_event_handler.file_extensions == [".mp4", ".mkv"]
        assert file_event_handler.known_files == set()
        assert file_event_handler.known_hashes == set()
        assert file_event_handler.new_files == {}
        assert isinstance(file_event_handler.file_detected_event, threading.Event)

    def test_on_created_with_matching_extension(
        self, file_event_handler: FileEventHandler
    ) -> None:
        """Test on_created method with a file that has a matching extension."""
        # Create a mock event
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/path/movie.mp4"

        # Call the method
        file_event_handler.on_created(event)

        # Check that the file was added to new_files
        assert "/test/path/movie.mp4" in file_event_handler.new_files
        assert (
            file_event_handler.new_files["/test/path/movie.mp4"]["path"]
            == "/test/path/movie.mp4"
        )
        assert file_event_handler.new_files["/test/path/movie.mp4"]["hash"] is None
        assert isinstance(
            file_event_handler.new_files["/test/path/movie.mp4"]["time"], float
        )

        # Check that the event was set
        assert file_event_handler.file_detected_event.is_set()

    def test_on_created_with_non_matching_extension(
        self, file_event_handler: FileEventHandler
    ) -> None:
        """Test on_created method with a file that has a non-matching extension."""
        # Create a mock event
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/path/movie.txt"

        # Call the method
        file_event_handler.on_created(event)

        # Check that the file was not added to new_files
        assert "/test/path/movie.txt" not in file_event_handler.new_files

        # Check that the event was not set
        assert not file_event_handler.file_detected_event.is_set()

    def test_on_created_with_directory(
        self, file_event_handler: FileEventHandler
    ) -> None:
        """Test on_created method with a directory."""
        # Create a mock event
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/test/path/directory"

        # Call the method
        file_event_handler.on_created(event)

        # Check that the directory was not added to new_files
        assert "/test/path/directory" not in file_event_handler.new_files

        # Check that the event was not set
        assert not file_event_handler.file_detected_event.is_set()

    def test_on_created_with_known_file(
        self, file_event_handler: FileEventHandler
    ) -> None:
        """Test on_created method with a file that is already known."""
        # Add a file to known_files
        file_event_handler.known_files.add("/test/path/known_movie.mp4")

        # Create a mock event
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/path/known_movie.mp4"

        # Call the method
        file_event_handler.on_created(event)

        # Check that the file was not added to new_files
        assert "/test/path/known_movie.mp4" not in file_event_handler.new_files

        # Check that the event was not set
        assert not file_event_handler.file_detected_event.is_set()

    def test_on_created_with_empty_extensions(self, mock_logger: MagicMock) -> None:
        """Test on_created method with empty file extensions list."""
        # Create a handler with empty file_extensions
        file_extensions: List[str] = []
        known_files: Set[str] = set()
        known_hashes: Set[str] = set()
        new_files: Dict[str, Dict[str, Any]] = {}
        file_detected_event = threading.Event()

        handler = FileEventHandler(
            file_extensions=file_extensions,
            known_files=known_files,
            known_hashes=known_hashes,
            new_files=new_files,
            file_detected_event=file_detected_event,
            logger=mock_logger,
        )

        # Create a mock event
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/path/movie.any_extension"

        # Call the method
        handler.on_created(event)

        # Check that the file was added to new_files (since any extension is allowed)
        assert "/test/path/movie.any_extension" in handler.new_files

        # Check that the event was set
        assert handler.file_detected_event.is_set()

    def test_on_created_case_insensitive_extension(
        self, file_event_handler: FileEventHandler
    ) -> None:
        """Test on_created method with case-insensitive extension matching."""
        # Create a mock event with uppercase extension
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/path/movie.MP4"

        # Call the method
        file_event_handler.on_created(event)

        # Check that the file was added to new_files
        assert "/test/path/movie.MP4" in file_event_handler.new_files

        # Check that the event was set
        assert file_event_handler.file_detected_event.is_set()


class TestWatchDog:
    """Tests for the WatchDog class."""

    @pytest.mark.asyncio
    async def test_init(self, watchdog: WatchDog) -> None:
        """Test initialization of WatchDog."""
        assert watchdog.observer is None
        assert watchdog.event_handler is None
        assert watchdog.known_files == set()
        assert watchdog.known_hashes == set()
        assert watchdog.new_files == {}
        assert isinstance(watchdog.processing_lock, asyncio.Lock)
        assert isinstance(watchdog.file_detected_event, threading.Event)
        assert watchdog._running is False

    @pytest.mark.asyncio
    async def test_start_invalid_params(self, watchdog: WatchDog) -> None:
        """Test start method with invalid parameters."""
        with pytest.raises(
            ValueError, match="Parameters must be of type WatchDogParams"
        ):
            await watchdog.start(None)

    @patch("src.services.watchdog.Observer")
    @patch("src.services.watchdog.FileEventHandler")
    @pytest.mark.asyncio
    async def test_start(
        self,
        mock_file_event_handler_class: MagicMock,
        mock_observer_class: MagicMock,
        watchdog: WatchDog,
        watchdog_params: WatchDogParams,
    ) -> None:
        """Test start method."""
        # Setup mocks
        mock_observer = mock_observer_class.return_value
        mock_file_event_handler = mock_file_event_handler_class.return_value

        # Mock the _get_known_files method
        watchdog._get_known_files = AsyncMock()

        # Call the method
        await watchdog.start(watchdog_params)

        # Check that _get_known_files was called
        watchdog._get_known_files.assert_called_once()

        # Check that the observer was set up correctly
        mock_observer_class.assert_called_once()
        mock_file_event_handler_class.assert_called_once_with(
            watchdog_params.file_extensions,
            watchdog.known_files,
            watchdog.known_hashes,
            watchdog.new_files,
            watchdog.file_detected_event,
            watchdog.logger,
        )
        mock_observer.schedule.assert_called_once_with(
            mock_file_event_handler, str(watchdog_params.dir_path), recursive=True
        )
        mock_observer.start.assert_called_once()

        # Check that the background task was started
        assert watchdog._running is True
        assert watchdog._check_task is not None

        # Clean up
        watchdog._running = False
        if watchdog._check_task:
            watchdog._check_task.cancel()

    @pytest.mark.asyncio
    async def test_process_iteration_invalid_params(self, watchdog: WatchDog) -> None:
        """Test process_iteration method with invalid parameters."""
        with pytest.raises(
            ValueError, match="Parameters must be of type WatchDogParams"
        ):
            await watchdog.process_iteration(None)

    @pytest.mark.asyncio
    async def test_process_iteration_with_event(
        self,
        watchdog: WatchDog,
        watchdog_params: WatchDogParams,
    ) -> None:
        """Test process_iteration method when an event is triggered."""
        # Set up the event
        watchdog.file_event = asyncio.Event()
        watchdog.file_event.set()

        # Mock the methods
        watchdog._calculate_hashes = AsyncMock()
        watchdog._process_new_files = AsyncMock(return_value=[])

        # Call the method
        await watchdog.process_iteration(watchdog_params)

        # Check that the methods were called
        watchdog._calculate_hashes.assert_called_once()
        watchdog._process_new_files.assert_called_once_with(watchdog_params.media_type)

        # Check that the event was cleared
        assert not watchdog.file_event.is_set()

    @pytest.mark.asyncio
    async def test_process_iteration_with_child_jobs(
        self,
        watchdog: WatchDog,
        watchdog_params: WatchDogParams,
    ) -> None:
        """Test process_iteration method when child jobs are created."""
        # Set up the event
        watchdog.file_event = asyncio.Event()
        watchdog.file_event.set()

        # Create mock child jobs
        child_jobs = [
            ChildJobRequest(
                job_type=JobType.FILE_MATCHER,
                params=FileMatcherParams(
                    path="/test/path/file.mp4",
                    media_type=MediaType.MOVIE,
                    file_id=uuid.uuid4(),
                ),
            ),
            ChildJobRequest(
                job_type=JobType.FFPROBE,
                params=FFProbeParams(
                    file_id=uuid.uuid4(),
                    path="/test/path/file.mp4",
                ),
            ),
        ]

        # Mock the methods
        watchdog._calculate_hashes = AsyncMock()
        watchdog._process_new_files = AsyncMock(return_value=child_jobs)
        watchdog._save_jobs_to_db = AsyncMock()

        # Call the method
        await watchdog.process_iteration(watchdog_params)

        # Check that the methods were called
        watchdog._calculate_hashes.assert_called_once()
        watchdog._process_new_files.assert_called_once_with(watchdog_params.media_type)
        watchdog._save_jobs_to_db.assert_called_once_with(child_jobs)

    @pytest.mark.asyncio
    async def test_process_iteration_with_exception(
        self,
        watchdog: WatchDog,
        watchdog_params: WatchDogParams,
    ) -> None:
        """Test process_iteration method when an exception occurs."""
        # Set up the event
        watchdog.file_event = asyncio.Event()
        watchdog.file_event.set()

        # Mock the methods to raise an exception
        watchdog._calculate_hashes = AsyncMock(side_effect=Exception("Test exception"))

        # Call the method and check that the exception is raised
        with pytest.raises(Exception, match="Test exception"):
            await watchdog.process_iteration(watchdog_params)

        # Check that the logger was called
        watchdog.logger.error.assert_called_with(
            "Error in watchdog processing iteration: Test exception"
        )

    @pytest.mark.asyncio
    async def test_stop(self, watchdog: WatchDog) -> None:
        """Test stop method."""
        # Set up the watchdog
        watchdog._running = True
        watchdog._check_task = asyncio.create_task(asyncio.sleep(0))
        watchdog.observer = MagicMock()

        # Call the method
        await watchdog.stop()

        # Check that everything was stopped
        assert watchdog._running is False
        assert watchdog.observer.stop.called
        assert watchdog.observer.join.called

    @pytest.mark.asyncio
    async def test_check_file_detected_event(self, watchdog: WatchDog) -> None:
        """Test _check_file_detected_event method."""
        # Set up the watchdog
        watchdog._running = True
        watchdog.file_detected_event = threading.Event()
        watchdog.file_event = asyncio.Event()

        # Create a task for the method
        task = asyncio.create_task(watchdog._check_file_detected_event())

        # Set the threading event
        watchdog.file_detected_event.set()

        # Wait a bit for the task to process
        await asyncio.sleep(0.2)

        # Check that the asyncio event was set
        assert watchdog.file_event.is_set()

        # Clean up
        watchdog._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
