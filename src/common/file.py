import os
import hashlib
from src.common.logger import Logger
from typing import Optional, List, Set

import aiofiles
import aiofiles.os


async def calculate_md5(file_path: str, chunk_size: int = 4096) -> str:
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


async def scan_directory(
    directory_path: str, file_extensions: List[str], logger: Optional[Logger] = None
) -> List[str]:
    """
    Scan a directory for files with specified extensions

    Args:
        directory_path: Path to scan
        file_extensions: List of file extensions to look for
        logger: Optional logger for debug information

    Returns:
        List of file paths that match the criteria
    """
    matching_files: List[str] = []
    lowercase_extensions: Set[str] = {ext.lower() for ext in file_extensions}

    if logger:
        logger.debug(
            f"Scanning {directory_path} for files with extensions: {file_extensions}"
        )

    # Verify directory exists
    if not await aiofiles.os.path.exists(directory_path):
        if logger:
            logger.error(f"Directory does not exist: {directory_path}")
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
                if logger:
                    logger.debug(f"Found matching file: {full_path}")

    if logger:
        logger.info(f"Found {len(matching_files)} matching files in {directory_path}")

    return matching_files
