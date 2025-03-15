"""Job dispatcher for processing open jobs from the queue."""

import asyncio
from typing import Optional, List
from uuid import UUID
import signal

from sqlalchemy import select, desc

from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    FileMatcherParams,
    FFProbeParams,
    ImageDownloaderParams,
    MetadataMatcherParams,
    MediaScannerParams,
    MovieMatcherParams,
    TranscodeParams,
    TvMatcherParams,
)
from src.common.logger import Logger, LogLevel
from src.common.models import Job
from src.common.system_types import JobStatus, JobType
from src.job_manager import job_manager


class JobDispatcher:
    """Dispatcher that polls for open jobs and processes them."""

    PARAMS_MAP = {
        JobType.MEDIA_SCAN: MediaScannerParams,
        JobType.FILE_MATCHER: FileMatcherParams,
        JobType.METADATA_MATCHER: MetadataMatcherParams,
        JobType.FFPROBE: FFProbeParams,
        JobType.MOVIE_MATCHER: MovieMatcherParams,
        JobType.TV_MATCHER: TvMatcherParams,
        JobType.TRANSCODER: TranscodeParams,
        JobType.IMAGE_DOWNLOADER: ImageDownloaderParams,
    }

    def __init__(
        self,
        db_session: AsyncDatabaseSession,
        poll_interval: int = 5,
        max_concurrent_jobs: int = 5,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the job dispatcher.

        Args:
            db_session: Database session for database operations
            poll_interval: Interval in seconds between job polling
            max_concurrent_jobs: Maximum number of jobs to process concurrently
            logger: Optional logger instance
        """
        self.db_session = db_session
        self.poll_interval = poll_interval
        self.max_concurrent_jobs = max_concurrent_jobs
        self.logger = logger or Logger("job_dispatcher", LogLevel.INFO)
        self.running = False
        self.active_jobs: set[UUID] = set()
        self.semaphore = asyncio.Semaphore(max_concurrent_jobs)

    async def start(self) -> None:
        """Start the job dispatcher loop."""
        self.running = True
        self.logger.info(
            f"Starting job dispatcher with poll interval {self.poll_interval}s"
        )

        while self.running:
            try:
                # Get open jobs with their full data
                jobs = await self._get_open_jobs()

                if jobs:
                    self.logger.info(f"Found {len(jobs)} open jobs to process")

                    # Process each job with its full data
                    await asyncio.gather(*[self._process_job(job) for job in jobs])
                else:
                    self.logger.debug("No open jobs found")

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                self.logger.info("Job dispatcher received cancellation")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Error in job dispatcher loop: {str(e)}")
                await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        """Stop the job dispatcher."""
        self.logger.info("Stopping job dispatcher")
        self.running = False

    async def _get_open_jobs(self) -> List[Job]:
        """Get open jobs from the database, ordered by priority and creation time.

        Returns:
            List of open jobs
        """
        async for session in self.db_session.get_session():
            # Calculate how many more jobs we can process
            available_slots = self.max_concurrent_jobs - len(self.active_jobs)

            if available_slots <= 0:
                return []

            # Query for open jobs, ordered by priority (desc) and creation time
            stmt = (
                select(Job)
                .where(Job.status == JobStatus.OPEN)
                .where(
                    Job.id.notin_(self.active_jobs)
                )  # Exclude already processing jobs
                .order_by(desc(Job.priority), Job.created_at)
                .limit(available_slots)
            )

            result = await session.execute(stmt)
            jobs = list(result.scalars().all())

            # Filter out any jobs that might have changed state since query
            valid_jobs: List[Job] = []
            for job in jobs:
                if job.id in self.active_jobs:
                    self.logger.warning(
                        f"Job {job.id} is already in active_jobs but was returned in query"
                    )
                    continue
                if job.status != JobStatus.OPEN:
                    self.logger.warning(
                        f"Job {job.id} status changed from OPEN to {job.status} between query and processing"
                    )
                    continue
                valid_jobs.append(job)

            self.logger.debug(f"Found {len(valid_jobs)} open jobs")
            return valid_jobs

        return []

    async def _process_job(self, job: Job) -> None:
        """Process a single job using the job manager."""
        self.active_jobs.add(job.id)

        try:
            async with self.semaphore:
                job_logger = self.logger.bind(job_id=str(job.id))
                job_logger.info(f"Processing job {job.id}")

                try:
                    async with job_manager(
                        self.db_session, job.id, job_logger
                    ) as context:
                        job_logger.debug(
                            f"Executing job {job.id} of type {job.job_type}"
                        )

                        # Get parameters using the job object passed to this method
                        params_class = self.PARAMS_MAP[job.job_type]
                        params = params_class.model_validate(job.parameters)

                        child_jobs = await context.execute_job(params)

                        if child_jobs:
                            job_logger.info(f"Creating {len(child_jobs)} child jobs")
                            await context.create_child_jobs(child_jobs)
                            job_logger.debug(
                                f"Created child jobs: {[j.job_type for j in child_jobs]}"
                            )
                        else:
                            job_logger.debug("No child jobs created")

                        job_logger.info(f"Job {job.id} completed successfully")
                except Exception as e:
                    job_logger.error(f"Error in job context: {str(e)}")
                    raise
        except Exception as e:
            self.logger.error(f"Error processing job {job.id}: {str(e)}")
        finally:
            self.active_jobs.discard(job.id)


async def run_job_dispatcher(
    db_session: AsyncDatabaseSession,
    poll_interval: int = 5,
    max_concurrent_jobs: int = 5,
    logger: Optional[Logger] = None,
) -> None:
    """Run the job dispatcher as a standalone process.

    Args:
        db_session: Database session for database operations
        poll_interval: Interval in seconds between job polling
        max_concurrent_jobs: Maximum number of jobs to process concurrently
        logger: Optional logger instance
    """
    dispatcher = JobDispatcher(
        db_session=db_session,
        poll_interval=poll_interval,
        max_concurrent_jobs=max_concurrent_jobs,
        logger=logger,
    )

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    async def shutdown(signal: Optional[signal.Signals] = None) -> None:
        if signal and logger:
            logger.info(f"Received exit signal {signal.name}")
        await dispatcher.stop()

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    try:
        await dispatcher.start()
    except asyncio.CancelledError:
        await dispatcher.stop()
