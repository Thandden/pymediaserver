from typing import Optional
import uuid

from sqlalchemy import select, Result
from src.common.dto import FileDTO, FileMatcherParams, MediaScannerParams
from src.workers.base import T_JobParams, Worker
from src.common.dto import ChildJobRequest, JobType
import aiofiles
import aiofiles.os
import hashlib
import os
from src.common.models import File
from src.common.db import AsyncDatabaseSession
from src.common.logger import Logger


class MediaScanner(Worker):
    """Media scanning worker implementation."""

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        super().__init__(db_session, logger)

    async def execute(
        self, parameters: Optional[T_JobParams] = None
    ) -> list[ChildJobRequest]:
        """
        Execute the media scanning worker

        Args:
            parameters: Job parameters

        Returns:
            ChildJobRequest[JobDTO]: Request for child jobs
        """
        if not isinstance(parameters, MediaScannerParams):
            raise ValueError("Parameters must be of type MediaScannerParams")

        found_files = await self._scan_directory(
            parameters.dir_path, parameters.file_extensions
        )

        # Get existing files from database and convert to set for efficient lookups
        existing_files = await self._get_all_files()
        existing_file_paths = set(file.path for file in existing_files)

        # Find unique files (those not already in the database)
        unique_files = [
            file_path
            for file_path in found_files
            if file_path not in existing_file_paths
        ]

        # Create a job for each unique file found
        indexed_files: list[FileDTO] = []
        child_jobs: list[ChildJobRequest] = []

        for file_path in unique_files:
            file_id = str(uuid.uuid4())
            md5_hash = await self._calculate_md5(file_path)
            indexed_files.append(
                FileDTO(
                    id=file_id,
                    path=file_path,
                    hash=md5_hash,
                    media_type=parameters.media_type,
                )
            )

            child_jobs.append(
                ChildJobRequest(
                    job_type=JobType.FILE_MATCHER,
                    params=FileMatcherParams(
                        path=file_path,
                        media_type=parameters.media_type,
                        file_id=file_id,
                    ),
                )
            )

        if indexed_files:
            await self._update_db(indexed_files)

        return child_jobs

    async def _get_all_files(self) -> list[FileDTO]:
        """
        Get all files from the database
        """
        try:
            async for session in self.db_session.get_session():
                result: Result[tuple[File]] = await session.execute(select(File))
                files = result.scalars().all()
                if not files:
                    return []
                return [
                    FileDTO(
                        id=file.id,
                        path=file.path,
                        hash=file.hash,
                        media_type=file.media_type,
                    )
                    for file in files
                ]
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error fetching files: {e}")
            return []  # Return empty list on exception

        # Return empty list if no session was yielded
        return []

    async def _update_db(self, files: list[FileDTO]) -> None:
        """
        Update the database with the indexed files
        """
        async for session in self.db_session.get_session():
            for file in files:
                # add to file table
                file_model: File = File(**file.model_dump())
                session.add(file_model)

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

    async def _scan_directory(
        self, directory_path: str, file_extensions: list[str]
    ) -> list[str]:
        """
        Scan a directory for files with specified extensions

        Args:
            directory_path: Path to scan
            file_extensions: List of file extensions to look for
            logger: Optional logger for debug information

        Returns:
            List of file paths that match the criteria
        """
        matching_files: list[str] = []
        lowercase_extensions: set[str] = {ext.lower() for ext in file_extensions}

        if self.logger:
            self.logger.debug(
                f"Scanning {directory_path} for files with extensions: {file_extensions}"
            )

        # Verify directory exists
        if not await aiofiles.os.path.exists(directory_path):
            if self.logger:
                self.logger.error(f"Directory does not exist: {directory_path}")
            return matching_files

        # Walk through the directory tree synchronously (os.walk is not async)
        # For large directories, this could be optimized further
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_ext = os.path.splitext(filename)[1].lower()

                # Check if this file has a matching extension
                if not file_extensions or file_ext in lowercase_extensions:
                    full_path = os.path.join(root, filename)
                    matching_files.append(full_path)
                    if self.logger:
                        self.logger.debug(f"Found matching file: {full_path}")

        if self.logger:
            self.logger.info(
                f"Found {len(matching_files)} matching files in {directory_path}"
            )

        return matching_files
