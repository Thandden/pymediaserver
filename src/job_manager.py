"""Job management and processing system."""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Type, TypeVar
from uuid import UUID
from datetime import datetime, timezone

from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.db import AsyncDatabaseSession
from src.common.dto import ChildJobRequest, JobParams
from src.common.logger import Logger
from src.common.models import Job
from src.common.system_types import JobStatus, JobType
from src.common.base import Worker
from src.workers.ffprobe import FFProbe
from src.workers.file_matcher import FileMatcher
from src.workers.image_downloader import ImageDownloader
from src.workers.media_scanner import MediaScanner
from src.workers.metadata_matcher import MetadataMatcher
from src.workers.movie_matcher import MovieMatcher
from src.workers.transcoder import Transcoder
from src.workers.tv_matcher import TVMatcher

T_Params = TypeVar("T_Params", bound=JobParams)


class JobContext:
    """Async context manager for job processing lifecycle."""

    # Map job types to their worker classes
    WORKER_MAP: Dict[JobType, Type[Worker]] = {
        JobType.MEDIA_SCAN: MediaScanner,
        JobType.FILE_MATCHER: FileMatcher,
        JobType.METADATA_MATCHER: MetadataMatcher,
        JobType.FFPROBE: FFProbe,
        JobType.MOVIE_MATCHER: MovieMatcher,
        JobType.TV_MATCHER: TVMatcher,
        JobType.TRANSCODER: Transcoder,
        JobType.IMAGE_DOWNLOADER: ImageDownloader,
        # Add other job types and their workers here
    }

    def __init__(
        self,
        db_session: AsyncDatabaseSession,
        job_id: UUID,
        logger: Optional[Logger] = None,
    ) -> None:
        self.db_session = db_session
        self.job_id = job_id
        self.logger = logger
        self.job: Optional[Job] = None
        self.worker: Optional[Worker] = None
        self.session: Optional[AsyncSession] = None
        self.job_type: Optional[JobType] = None
        self.params: Optional[JobParams] = None

    async def __aenter__(self) -> "JobContext":
        """Enter the job context and mark job as running."""
        async for session in self.db_session.get_session():
            self.session = session
            # Get and lock the job
            result = await session.execute(
                select(Job).where(Job.id == self.job_id).with_for_update()
            )
            self.job = result.scalar_one_or_none()

            if not self.job:
                raise ValueError(f"Job {self.job_id} not found")
            if self.job.status != JobStatus.OPEN:
                raise ValueError(f"Job {self.job_id} is not in OPEN state")

            # Update job status to running
            self.job.status = JobStatus.RUNNING
            self.job.started_at = datetime.now(timezone.utc)
            await session.commit()

            # Create the appropriate worker for this job type
            self._create_worker(self.job.job_type)

            # Store job type and parameters as instance attributes
            self.job_type = self.job.job_type

            # Use TypeAdapter instead of deprecated parse_obj_as
            job_params_adapter = TypeAdapter(JobParams)
            self.params = job_params_adapter.validate_python(self.job.parameters)

            return self  # Return the context instance instead of tuple

        # This should never be reached if the database session is working correctly
        raise RuntimeError("Failed to acquire database session")

    def _create_worker(self, job_type: JobType) -> None:
        """Create the appropriate worker for the job type.

        Args:
            job_type: The type of job to create a worker for

        Raises:
            ValueError: If no worker is registered for the job type
        """
        if job_type not in self.WORKER_MAP:
            raise ValueError(f"No worker registered for job type {job_type}")

        worker_class = self.WORKER_MAP[job_type]
        self.worker = worker_class(self.db_session, self.logger)

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit the job context and update final status."""
        if not self.job or not self.session:
            return

        async for session in self.db_session.get_session():
            # Get a fresh copy of the job
            result = await session.execute(select(Job).where(Job.id == self.job_id))
            job = result.scalar_one_or_none()

            if not job:
                if self.logger:
                    self.logger.error(f"Job {self.job_id} not found during exit")
                return

            if exc_val:
                # Handle error case
                job.status = JobStatus.FAILED
                job.error = str(exc_val)
                if self.logger:
                    self.logger.error(f"Job failed: {exc_val}")
            else:
                # Handle success case
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                if self.logger:
                    self.logger.info(f"Job completed successfully: {self.job_id}")

            await session.commit()

    async def execute_job(self, params: JobParams) -> list[ChildJobRequest]:
        """Execute the job using the appropriate worker.

        Args:
            params: The parameters for the job

        Returns:
            List of child job requests created by the worker

        Raises:
            ValueError: If no worker has been created for this job
        """
        if not self.worker:
            raise ValueError("No worker available for job execution")

        if self.logger:
            self.logger.info(
                f"Executing job {self.job_id} with worker {type(self.worker).__name__}"
            )

        # Execute the worker with the provided parameters
        result = await self.worker.execute(params)
        # Ensure we always return a list, even if the worker returns None
        return result if result is not None else []

    async def create_child_jobs(self, child_requests: list[ChildJobRequest]) -> None:
        """Create child jobs from worker requests."""
        if not child_requests:
            return

        if self.logger:
            self.logger.info(
                f"Creating {len(child_requests)} child jobs for job {self.job_id}"
            )

        async for session in self.db_session.get_session():
            for request in child_requests:
                # Ensure params is not None before calling model_dump
                if request.params is None:
                    if self.logger:
                        self.logger.error(
                            f"Child job request has None params: {request}"
                        )
                    continue

                # Convert parameters to a serializable dict, ensuring enum values are converted to strings
                params_dict = request.params.model_dump(mode="json")

                child_job = Job(
                    job_type=request.job_type,
                    parameters=params_dict,  # Use the JSON-serializable dict
                    status=JobStatus.OPEN,
                    priority=request.priority,
                    parent_job_id=self.job_id,
                )
                session.add(child_job)
            await session.commit()


@asynccontextmanager
async def job_manager(
    db_session: AsyncDatabaseSession,
    job_id: UUID,
    logger: Optional[Logger] = None,
) -> AsyncGenerator[JobContext, None]:
    """Create a job processing context.

    Args:
        db_session: Database session for database operations
        job_id: ID of the job to process
        logger: Optional logger instance

    Yields:
        JobContext: Context manager for job processing
    """
    context = JobContext(db_session, job_id, logger)
    try:
        await context.__aenter__()
        yield context
    except Exception as e:
        if logger:
            logger.error(f"Error in job context: {str(e)}")
        await context.__aexit__(type(e), e, e.__traceback__)
        raise
    else:
        await context.__aexit__(None, None, None)
