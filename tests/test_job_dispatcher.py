"""Unit tests for the JobDispatcher class."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, List, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import Result, ScalarResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    ChildJobRequest,
    FileMatcherParams,
)
from src.common.logger import LogLevel, Logger
from src.common.models import Job
from src.common.system_types import JobStatus, JobType, MediaType
from src.job_dispatcher import JobDispatcher, run_job_dispatcher
from src.job_manager import JobContext


class MockAsyncDatabaseSession(AsyncDatabaseSession):
    """Mock implementation of AsyncDatabaseSession for testing."""

    def __init__(self) -> None:
        """Initialize the mock database session."""
        self._session = AsyncMock(spec=AsyncSession)
        self._engine = MagicMock()

    @property
    def session(self) -> AsyncSession:
        """Get the mock session.

        Returns:
            AsyncSession: The mock session
        """
        return self._session

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a mock database session.

        Yields:
            AsyncSession: Mock database session
        """
        yield self._session

    async def close(self) -> None:
        """Close the mock database session."""
        pass


class TestJobDispatcher:
    """Test suite for the JobDispatcher class."""

    @pytest_asyncio.fixture(scope="function")
    async def db_session(self) -> MockAsyncDatabaseSession:
        """Create a mock database session.

        Returns:
            MockAsyncDatabaseSession: Mock database session
        """
        session = MockAsyncDatabaseSession()
        yield session
        await session.close()

    @pytest.fixture
    def logger(self) -> Logger:
        """Create a mock logger.

        Returns:
            Logger: Mock logger
        """
        return Logger("test_logger", LogLevel.DEBUG)

    @pytest.fixture
    def dispatcher(
        self, db_session: MockAsyncDatabaseSession, logger: Logger
    ) -> JobDispatcher:
        """Create a JobDispatcher instance for testing.

        Args:
            db_session: Mock database session
            logger: Mock logger

        Returns:
            JobDispatcher: JobDispatcher instance
        """
        return JobDispatcher(
            db_session=db_session,
            poll_interval=1,
            max_concurrent_jobs=2,
            logger=logger,
        )

    @pytest.fixture
    def mock_job(self) -> Job:
        """Create a mock job.

        Returns:
            Job: Mock job
        """
        job_id = uuid.uuid4()
        return Job(
            id=job_id,
            job_type=JobType.MEDIA_SCAN,
            status=JobStatus.OPEN,
            parameters={
                "dir_path": "/test",
                "media_type": "MOVIE",
                "file_extensions": [".mp4"],
            },
            priority=1,
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_init(self, dispatcher: JobDispatcher) -> None:
        """Test JobDispatcher initialization.

        Args:
            dispatcher: JobDispatcher instance
        """
        assert dispatcher.poll_interval == 1
        assert dispatcher.max_concurrent_jobs == 2
        assert dispatcher.running is False
        assert isinstance(dispatcher.active_jobs, set)
        assert len(dispatcher.active_jobs) == 0
        assert isinstance(dispatcher.semaphore, asyncio.Semaphore)
        assert dispatcher.semaphore._value == 2  # type: ignore

    @pytest.mark.asyncio
    async def test_get_open_jobs_empty(self, dispatcher: JobDispatcher) -> None:
        """Test getting open jobs when none exist."""
        # Mock the database query result to return no jobs
        mock_result = AsyncMock(spec=Result)
        mock_scalars = AsyncMock(spec=ScalarResult)
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars

        # Set up the mock to return our result
        mock_session = cast(MockAsyncDatabaseSession, dispatcher.db_session)
        mock_session.session.execute = AsyncMock(return_value=mock_result)

        jobs = await dispatcher._get_open_jobs()  # type: ignore

        assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_get_open_jobs_with_jobs(
        self, dispatcher: JobDispatcher, mock_job: Job
    ) -> None:
        """Test getting open jobs when jobs exist."""
        # Mock the database query result to return a job
        mock_result = AsyncMock(spec=Result)
        mock_scalars = AsyncMock(spec=ScalarResult)
        mock_scalars.all.return_value = [mock_job]
        mock_result.scalars.return_value = mock_scalars

        # Set up the mock to return our result
        mock_session = cast(MockAsyncDatabaseSession, dispatcher.db_session)
        mock_session.session.execute = AsyncMock(return_value=mock_result)

        jobs = await dispatcher._get_open_jobs()  # type: ignore

        assert len(jobs) == 1
        assert jobs[0].id == mock_job.id

    @pytest.mark.asyncio
    async def test_get_open_jobs_max_concurrent(
        self, dispatcher: JobDispatcher, mock_job: Job
    ) -> None:
        """Test getting open jobs when max concurrent jobs are already running.

        Args:
            dispatcher: JobDispatcher instance
            mock_job: Mock job
        """
        # Add max_concurrent_jobs to active_jobs
        for _ in range(dispatcher.max_concurrent_jobs):
            dispatcher.active_jobs.add(uuid.uuid4())

        # Mock the database query result to return a job
        mock_result = AsyncMock(spec=Result)
        mock_scalars = AsyncMock(spec=ScalarResult)
        mock_scalars.all.return_value = [mock_job]
        mock_result.scalars.return_value = mock_scalars

        # Set up the mock to return our result
        mock_session = cast(MockAsyncDatabaseSession, dispatcher.db_session)
        mock_session.session.execute = AsyncMock(return_value=mock_result)

        # Call the method directly
        jobs = await dispatcher._get_open_jobs()  # type: ignore

        assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_get_open_jobs_filter_already_active(
        self, dispatcher: JobDispatcher, mock_job: Job
    ) -> None:
        """Test filtering out jobs that are already active.

        Args:
            dispatcher: JobDispatcher instance
            mock_job: Mock job
        """
        # Add the mock job to active_jobs
        dispatcher.active_jobs.add(mock_job.id)

        # Mock the database query result to return the same job
        mock_result = AsyncMock(spec=Result)
        mock_scalars = AsyncMock(spec=ScalarResult)
        mock_scalars.all.return_value = [mock_job]
        mock_result.scalars.return_value = mock_scalars

        # Set up the mock to return our result
        mock_session = cast(MockAsyncDatabaseSession, dispatcher.db_session)
        mock_session.session.execute = AsyncMock(return_value=mock_result)

        jobs = await dispatcher._get_open_jobs()  # type: ignore

        assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_get_open_jobs_filter_changed_status(
        self, dispatcher: JobDispatcher, mock_job: Job
    ) -> None:
        """Test filtering out jobs that have changed status.

        Args:
            dispatcher: JobDispatcher instance
            mock_job: Mock job
        """
        # Change the job status
        mock_job.status = JobStatus.RUNNING

        # Mock the database query result to return the job with changed status
        mock_result = AsyncMock(spec=Result)
        mock_scalars = AsyncMock(spec=ScalarResult)
        mock_scalars.all.return_value = [mock_job]
        mock_result.scalars.return_value = mock_scalars

        # Set up the mock to return our result
        mock_session = cast(MockAsyncDatabaseSession, dispatcher.db_session)
        mock_session.session.execute = AsyncMock(return_value=mock_result)

        jobs = await dispatcher._get_open_jobs()  # type: ignore

        assert len(jobs) == 0

    @pytest.mark.asyncio
    async def test_process_job_success(
        self,
        dispatcher: JobDispatcher,
        db_session: MockAsyncDatabaseSession,
        mock_job: Job,
    ) -> None:
        """Test successful job processing.

        Args:
            dispatcher: JobDispatcher instance
            db_session: Mock database session
            mock_job: Mock job
        """
        # Mock the job_manager context
        mock_context = AsyncMock(spec=JobContext)
        mock_context.execute_job.return_value = []  # No child jobs

        # Patch the job_manager to return our mock context
        with patch(
            "src.job_dispatcher.job_manager",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_context),
                __aexit__=AsyncMock(return_value=None),
            ),
        ):
            # Call the method directly
            await dispatcher._process_job(mock_job)  # type: ignore

        # Verify that the job was added to active_jobs and then removed
        assert mock_job.id not in dispatcher.active_jobs

        # Verify that execute_job was called with the correct parameters
        mock_context.execute_job.assert_called_once()

        # Verify that create_child_jobs was not called (no child jobs)
        mock_context.create_child_jobs.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_job_with_child_jobs(
        self,
        dispatcher: JobDispatcher,
        db_session: MockAsyncDatabaseSession,
        mock_job: Job,
    ) -> None:
        """Test job processing with child jobs.

        Args:
            dispatcher: JobDispatcher instance
            db_session: Mock database session
            mock_job: Mock job
        """
        # Create mock child job requests
        child_job_params = FileMatcherParams(
            path="/test/file.mp4",
            media_type=MediaType.MOVIE,
            file_id=uuid.uuid4(),
        )
        child_jobs = [
            ChildJobRequest(
                job_type=JobType.FILE_MATCHER,
                params=child_job_params,
                priority=1,
            )
        ]

        # Mock the job_manager context
        mock_context = AsyncMock(spec=JobContext)
        mock_context.execute_job.return_value = child_jobs

        # Patch the job_manager to return our mock context
        with patch(
            "src.job_dispatcher.job_manager",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_context),
                __aexit__=AsyncMock(return_value=None),
            ),
        ):
            # Call the method directly
            await dispatcher._process_job(mock_job)  # type: ignore

        # Verify that the job was added to active_jobs and then removed
        assert mock_job.id not in dispatcher.active_jobs

        # Verify that execute_job was called with the correct parameters
        mock_context.execute_job.assert_called_once()

        # Verify that create_child_jobs was called with the child jobs
        mock_context.create_child_jobs.assert_called_once_with(child_jobs)

    @pytest.mark.asyncio
    async def test_process_job_error(
        self,
        dispatcher: JobDispatcher,
        db_session: MockAsyncDatabaseSession,
        mock_job: Job,
    ) -> None:
        """Test job processing with an error.

        Args:
            dispatcher: JobDispatcher instance
            db_session: Mock database session
            mock_job: Mock job
        """
        # Mock the job_manager context to raise an exception
        mock_context = AsyncMock(spec=JobContext)
        mock_context.execute_job.side_effect = ValueError("Test error")

        # Patch the job_manager to return our mock context
        with patch(
            "src.job_dispatcher.job_manager",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_context),
                __aexit__=AsyncMock(return_value=None),
            ),
        ):
            # Call the method directly
            await dispatcher._process_job(mock_job)  # type: ignore

        # Verify that the job was added to active_jobs and then removed
        assert mock_job.id not in dispatcher.active_jobs

    @pytest.mark.asyncio
    async def test_start_stop(
        self,
        dispatcher: JobDispatcher,
        db_session: MockAsyncDatabaseSession,
        mock_job: Job,
    ) -> None:
        """Test starting and stopping the dispatcher.

        Args:
            dispatcher: JobDispatcher instance
            db_session: Mock database session
            mock_job: Mock job
        """
        # Mock _get_open_jobs to return a job once, then empty list
        with patch.object(
            JobDispatcher, "_get_open_jobs", autospec=True
        ) as mock_get_open_jobs:
            mock_get_open_jobs.side_effect = [[mock_job], [], []]

            # Mock _process_job
            with patch.object(
                JobDispatcher, "_process_job", autospec=True
            ) as mock_process_job:
                # Start the dispatcher in a task so we can stop it
                task = asyncio.create_task(dispatcher.start())

                # Wait a bit for the dispatcher to process the job
                await asyncio.sleep(0.2)

                # Stop the dispatcher
                await dispatcher.stop()

                # Wait for the task to complete
                await task

                # Verify that _get_open_jobs was called
                assert mock_get_open_jobs.call_count >= 1

                # Verify that _process_job was called with the mock job
                mock_process_job.assert_called_once()

                # Verify that the dispatcher is stopped
                assert dispatcher.running is False

    @pytest.mark.asyncio
    async def test_start_with_error(
        self, dispatcher: JobDispatcher, db_session: MockAsyncDatabaseSession
    ) -> None:
        """Test dispatcher handling errors in the main loop.

        Args:
            dispatcher: JobDispatcher instance
            db_session: Mock database session
        """
        # Mock _get_open_jobs to raise an exception, then return empty list
        with patch.object(
            JobDispatcher, "_get_open_jobs", autospec=True
        ) as mock_get_open_jobs:
            mock_get_open_jobs.side_effect = [ValueError("Test error"), [], []]

            # Start the dispatcher in a task so we can stop it
            task = asyncio.create_task(dispatcher.start())

            # Wait a bit for the dispatcher to handle the error
            await asyncio.sleep(0.2)

            # Stop the dispatcher
            await dispatcher.stop()

            # Wait for the task to complete
            await task

            # Verify that _get_open_jobs was called
            assert mock_get_open_jobs.call_count >= 1

            # Verify that the dispatcher is stopped
            assert dispatcher.running is False

    @pytest.mark.asyncio
    async def test_start_with_cancellation(
        self, dispatcher: JobDispatcher, db_session: MockAsyncDatabaseSession
    ) -> None:
        """Test dispatcher handling cancellation.

        Args:
            dispatcher: JobDispatcher instance
            db_session: Mock database session
        """

        # Mock _get_open_jobs to block indefinitely
        async def block_forever(*args: object, **kwargs: object) -> List[Job]:
            await asyncio.sleep(1000)
            return []

        with patch.object(
            JobDispatcher, "_get_open_jobs", autospec=True, side_effect=block_forever
        ):
            # Start the dispatcher in a task so we can cancel it
            task = asyncio.create_task(dispatcher.start())

            # Wait a bit for the dispatcher to start
            await asyncio.sleep(0.1)

            # Cancel the task
            task.cancel()

            # Wait for the task to complete (should handle cancellation)
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify that the dispatcher is stopped
            assert dispatcher.running is False

    @pytest.mark.asyncio
    async def test_run_job_dispatcher(
        self, db_session: MockAsyncDatabaseSession, logger: Logger
    ) -> None:
        """Test the run_job_dispatcher function."""
        mock_dispatcher = AsyncMock(spec=JobDispatcher)

        # Create an event to track when start is called
        start_called = asyncio.Event()

        async def mock_start() -> None:
            start_called.set()
            await asyncio.sleep(0.1)  # Simulate some work

        mock_dispatcher.start.side_effect = mock_start
        mock_dispatcher.stop = AsyncMock()

        with patch("src.job_dispatcher.JobDispatcher", return_value=mock_dispatcher):
            task = asyncio.create_task(
                run_job_dispatcher(
                    db_session=db_session,
                    poll_interval=1,
                    max_concurrent_jobs=2,
                    logger=logger,
                )
            )

            # Wait for start to be called
            await start_called.wait()

            # Cancel and wait for clean shutdown
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Wait a bit to ensure stop is called
            await asyncio.sleep(0.1)

            # Verify lifecycle methods were called
            mock_dispatcher.start.assert_awaited_once()
            mock_dispatcher.stop.assert_awaited_once()
