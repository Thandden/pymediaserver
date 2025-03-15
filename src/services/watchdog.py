from typing import Optional, Dict, Set, Any
import uuid
import os
import asyncio
import time
import hashlib
from datetime import datetime, timezone
import threading
from concurrent.futures import ThreadPoolExecutor

import aiofiles
import aiofiles.os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from sqlalchemy import select
from src.common.base_service import Service, T_ServiceParams
from src.common.dto import (
    WatchDogParams,
    ChildJobRequest,
    FileDTO,
    FFProbeParams,
    FileMatcherParams,
)
from src.common.models import File, Job
from src.common.db import AsyncDatabaseSession
from src.common.logger import Logger
from src.common.system_types import JobType, MediaType


class FileEventHandler(FileSystemEventHandler):
    """Handler for file system events."""

    def __init__(
        self,
        file_extensions: list[str],
        known_files: Set[str],
        known_hashes: Set[str],
        new_files: Dict[str, Dict[str, Any]],
        file_detected_event: threading.Event,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initialize the file event handler.

        Args:
            file_extensions: List of file extensions to monitor
            known_files: Set of known file paths
            known_hashes: Set of known file hashes
            new_files: Dictionary to store new files
            file_detected_event: Threading event to signal when a file is detected
            logger: Optional logger
        """
        self.file_extensions = [ext.lower() for ext in file_extensions]
        self.known_files = known_files
        self.known_hashes = known_hashes
        self.new_files = new_files
        self.file_detected_event = file_detected_event
        self.logger = logger

    def on_created(self, event: FileSystemEvent) -> None:
        """
        Handle file creation events.

        Args:
            event: File system event
        """
        if not event.is_directory:
            file_path = str(event.src_path)
            file_ext = os.path.splitext(file_path)[1].lower()

            # Check if this file has a matching extension and is not already known
            if (
                not self.file_extensions or file_ext in self.file_extensions
            ) and file_path not in self.known_files:
                if self.logger:
                    self.logger.debug(f"New file detected: {file_path}")

                # Store the file path and creation time
                self.new_files[file_path] = {
                    "path": file_path,
                    "time": time.time(),
                    "hash": None,  # Will be calculated later
                }

                # Set the threading event to signal a file was detected
                self.file_detected_event.set()


class WatchDog(Service[WatchDogParams]):
    """Watchdog worker implementation for monitoring directories for new files."""

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        """
        Initialize the watchdog worker.

        Args:
            db_session: Database session
            logger: Optional logger
        """
        super().__init__(db_session, logger)
        # Using Any type to avoid linter errors with Observer
        self.observer: Any = None
        self.event_handler: Optional[FileEventHandler] = None
        self.known_files: Set[str] = set()
        self.known_hashes: Set[str] = set()
        self.new_files: Dict[str, Dict[str, Any]] = {}
        self.processing_lock: asyncio.Lock = asyncio.Lock()

        # Use a threading.Event for cross-thread signaling
        self.file_detected_event = threading.Event()

        # Create a thread pool for checking the threading event
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Asyncio event for the main loop
        self.file_event = asyncio.Event()

        # Flag to control the event checking loop
        self._running = False

        # Background task for checking the threading event
        self._check_task = None

    async def start(self, parameters: Optional[WatchDogParams] = None) -> None:
        """
        Start the watchdog service.

        Args:
            parameters: Job parameters
        """
        if not isinstance(parameters, WatchDogParams):
            raise ValueError("Parameters must be of type WatchDogParams")

        if self.logger:
            self.logger.debug(f"Starting watchdog for {parameters.dir_path}")

        # Convert Path to string for compatibility
        dir_path = str(parameters.dir_path)

        # Get existing files from database
        await self._get_known_files()

        # Set up the file event handler with the threading event
        self.event_handler = FileEventHandler(
            parameters.file_extensions,
            self.known_files,
            self.known_hashes,
            self.new_files,
            self.file_detected_event,
            self.logger,
        )

        # Set up the observer
        observer = Observer()
        observer.schedule(self.event_handler, dir_path, recursive=True)
        observer.start()
        self.observer = observer

        # Start the background task to check for file events
        self._running = True
        self._check_task = asyncio.create_task(self._check_file_detected_event())

        if self.logger:
            self.logger.info(f"Watchdog started for directory: {dir_path}")

    async def _check_file_detected_event(self) -> None:
        """Background task to check the threading event and set the asyncio event."""
        while self._running:
            try:
                # Check if the threading event is set using the executor
                is_set = await asyncio.get_event_loop().run_in_executor(
                    self.executor, lambda: self.file_detected_event.is_set()
                )

                if is_set:
                    # Clear the threading event
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, self.file_detected_event.clear
                    )

                    # Set the asyncio event
                    self.file_event.set()

                    if self.logger:
                        self.logger.debug(
                            "File detection event transferred from thread to asyncio"
                        )

                # Short sleep to avoid CPU spinning
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in event checking task: {e}")
                await asyncio.sleep(1)  # Longer sleep on error

    async def process_iteration(
        self, parameters: Optional[WatchDogParams] = None
    ) -> None:
        """
        Process a single iteration of the watchdog service.

        Args:
            parameters: Job parameters
        """
        if not isinstance(parameters, WatchDogParams):
            raise ValueError("Parameters must be of type WatchDogParams")

        try:
            # Wait for the file event with a timeout
            try:
                # Use a shorter timeout for responsiveness, but still have a backup check
                await asyncio.wait_for(
                    self.file_event.wait(),
                    timeout=parameters.scan_interval
                    if hasattr(parameters, "scan_interval")
                    else 60,
                )
                # Clear the event for next time
                self.file_event.clear()
                if self.logger:
                    self.logger.debug("Processing files due to file detection event")
            except asyncio.TimeoutError:
                # Timeout is normal - we'll check for files anyway as a backup
                if self.logger:
                    self.logger.debug("Processing files due to periodic check")

            # Use a lock to prevent concurrent processing
            async with self.processing_lock:
                # Calculate MD5 hashes for new files
                await self._calculate_hashes()

                # Process any new files found
                child_jobs = await self._process_new_files(parameters.media_type)

                # Submit child jobs if any were created
                if child_jobs:
                    # Save the child jobs to the database
                    await self._save_jobs_to_db(child_jobs)
                    if self.logger:
                        self.logger.info(
                            f"Created and saved {len(child_jobs)} child jobs to database"
                        )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in watchdog processing iteration: {e}")
            raise

    async def stop(self) -> None:
        """Stop the watchdog service."""
        self._running = False

        # Cancel the check task
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        # Shutdown the executor
        self.executor.shutdown(wait=False)

        # Stop the observer
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()

        if self.logger:
            self.logger.info("Watchdog service stopped")

    async def _get_known_files(self) -> None:
        """
        Get all known files and their hashes from the database.
        """
        try:
            async for session in self.db_session.get_session():
                result = await session.execute(select(File))
                files = result.scalars().all()
                if files:
                    self.known_files = {file.path for file in files}
                    # Only add non-None hashes to the set
                    self.known_hashes = {
                        file.hash for file in files if file.hash is not None
                    }

                    if self.logger:
                        self.logger.debug(
                            f"Loaded {len(self.known_files)} known files and {len(self.known_hashes)} hashes from database"
                        )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error fetching files: {e}")

    async def _calculate_hashes(self) -> None:
        """
        Calculate MD5 hashes for all new files.
        """
        if not self.new_files:
            return

        if self.logger:
            self.logger.debug(f"Calculating MD5 hashes for {len(self.new_files)} files")

        for file_path in list(self.new_files.keys()):
            try:
                md5_hash = await self._calculate_md5(file_path)
                self.new_files[file_path]["hash"] = md5_hash

                # If the hash already exists in the database, remove this file from new_files
                if md5_hash in self.known_hashes:
                    if self.logger:
                        self.logger.debug(
                            f"File with hash {md5_hash} already exists in database, skipping: {file_path}"
                        )
                    del self.new_files[file_path]
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error calculating hash for {file_path}: {e}")
                # Remove the file from new_files if we can't calculate its hash
                if file_path in self.new_files:
                    del self.new_files[file_path]

    async def _calculate_md5(self, file_path: str, chunk_size: int = 4096) -> str:
        """
        Calculate MD5 hash of a file using async IO operations

        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to read from file

        Returns:
            Hexadecimal string representation of the MD5 hash

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file cannot be accessed
            IOError: For other IO-related errors
        """
        md5_hash = hashlib.md5()

        # Check if file exists and is accessible
        if not await aiofiles.os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Open file in binary mode for reading
        async with aiofiles.open(file_path, "rb") as file:
            # Read and update hash in chunks
            while chunk := await file.read(chunk_size):
                md5_hash.update(chunk)

        # Return the hexadecimal representation of the hash
        return md5_hash.hexdigest()

    async def _process_new_files(self, media_type: MediaType) -> list[ChildJobRequest]:
        """
        Process new files and create child jobs.

        Args:
            media_type: Media type for the files

        Returns:
            List of child job requests
        """
        if not self.new_files:
            if self.logger:
                self.logger.info("No new files found")
            return []

        if self.logger:
            self.logger.info(f"Found {len(self.new_files)} new files to process")

        # Create a job for each new file
        indexed_files: list[FileDTO] = []
        child_jobs: list[ChildJobRequest] = []

        for file_path, file_info in self.new_files.items():
            # Generate a UUID for the file
            file_id = uuid.uuid4()

            # Get the calculated hash
            md5_hash = file_info.get("hash")

            # Create a FileDTO for the database
            indexed_files.append(
                FileDTO(
                    id=file_id,
                    path=file_path,
                    media_type=media_type,
                    hash=md5_hash,
                    indexed=True,
                )
            )

            # Create child jobs for file matching and probe
            file_matcher_job = ChildJobRequest(
                job_type=JobType.FILE_MATCHER,
                params=FileMatcherParams(
                    path=file_path,
                    media_type=media_type,
                    file_id=file_id,
                ),
            )

            ffprobe_job = ChildJobRequest(
                job_type=JobType.FFPROBE,
                params=FFProbeParams(file_id=file_id, path=file_path),
            )

            child_jobs.extend([file_matcher_job, ffprobe_job])

            if self.logger:
                self.logger.debug(f"Created child jobs for file: {file_path}")

        # Update the database with the new files
        if indexed_files:
            await self._update_db(indexed_files)
            if self.logger:
                self.logger.info(
                    f"Updated database with {len(indexed_files)} new files"
                )

        # Clear the new files dictionary
        self.new_files.clear()

        if self.logger:
            self.logger.info(f"Created {len(child_jobs)} child jobs")

        return child_jobs

    async def _update_db(self, files: list[FileDTO]) -> None:
        """
        Update the database with the indexed files.

        Args:
            files: List of file DTOs
        """
        async for session in self.db_session.get_session():
            try:
                for file_dto in files:
                    file_model = File(
                        id=file_dto.id,
                        path=file_dto.path,
                        media_type=file_dto.media_type,
                        hash=file_dto.hash,
                    )
                    session.add(file_model)

                    # Update tracking sets
                    self.known_files.add(file_dto.path)
                    if file_dto.hash:
                        self.known_hashes.add(file_dto.hash)

                await session.commit()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error updating database: {e}")
                raise

    async def _save_jobs_to_db(self, jobs: list[ChildJobRequest]) -> None:
        """
        Save child jobs to the database.

        Args:
            jobs: List of child job requests
        """
        if not jobs:
            return

        if self.logger:
            self.logger.debug(f"Saving {len(jobs)} jobs to database")

        async for session in self.db_session.get_session():
            try:
                for job_request in jobs:
                    # Convert parameters to a serializable dict, ensuring enum values are converted to strings
                    params_dict = (
                        job_request.params.model_dump(mode="json")
                        if job_request.params
                        else None
                    )

                    # Create job model with the correct fields
                    job_model = Job(
                        id=job_request.id or uuid.uuid4(),
                        job_type=job_request.job_type,
                        status=job_request.status,
                        parameters=params_dict,  # Use the JSON-serializable dict
                        priority=job_request.priority,
                        created_at=job_request.created_at or datetime.now(timezone.utc),
                        updated_at=job_request.updated_at,
                        started_at=job_request.started_at,
                        completed_at=job_request.completed_at,
                        error=job_request.error,
                        retry_count=job_request.retry_count,
                        parent_job_id=job_request.parent_job_id,
                    )
                    session.add(job_model)

                await session.commit()
                if self.logger:
                    self.logger.info(f"Successfully saved {len(jobs)} jobs to database")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error saving jobs to database: {e}")
                # Don't raise the exception to avoid crashing the service
                await session.rollback()
