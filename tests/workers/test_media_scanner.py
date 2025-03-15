import os
import uuid
import hashlib
from typing import Any, AsyncGenerator, List, Optional, cast, TypeVar
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from sqlalchemy import Result
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.db import AsyncDatabaseSession
from src.common.dto import (
    ChildJobRequest,
    FileDTO,
    FileMatcherParams,
    JobType,
    MediaScannerParams,
)
from src.common.logger import Logger
from src.common.models import File
from src.common.system_types import MediaType
from src.workers.media_scanner import MediaScanner

T = TypeVar("T")


class MockAsyncSession(AsyncSession):
    def __init__(self) -> None:
        self.add_called = False
        self.execute_result: Optional[List[File]] = None

    async def execute(self, query: Any) -> Result[Any]:
        mock_result = MagicMock(spec=Result)
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = (
            self.execute_result if self.execute_result is not None else []
        )
        mock_result.scalars.return_value = mock_scalars
        return mock_result

    def add(self, obj: Any) -> None:
        self.add_called = True


class MockAsyncDatabaseSession(AsyncDatabaseSession):
    def __init__(self) -> None:
        self.session = MockAsyncSession()

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        yield self.session


@pytest.fixture
def mock_db_session() -> MockAsyncDatabaseSession:
    return MockAsyncDatabaseSession()


@pytest.fixture
def mock_logger() -> Logger:
    return MagicMock(spec=Logger)


@pytest.fixture
def media_scanner(
    mock_db_session: MockAsyncDatabaseSession, mock_logger: Logger
) -> MediaScanner:
    return MediaScanner(mock_db_session, mock_logger)


@pytest.mark.asyncio
async def test_execute_with_invalid_parameters(media_scanner: MediaScanner) -> None:
    """Test execute method with invalid parameters."""
    with pytest.raises(
        ValueError, match="Parameters must be of type MediaScannerParams"
    ):
        await media_scanner.execute(None)


@pytest.mark.asyncio
async def test_execute_with_no_files(media_scanner: MediaScanner) -> None:
    """Test execute method when no files are found."""
    # Mock _scan_directory to return empty list
    with patch.object(media_scanner, "_scan_directory", return_value=[]):
        result = await media_scanner.execute(
            MediaScannerParams(
                dir_path="/fake/path",
                file_extensions=[".mp3"],
                media_type=MediaType.MUSIC,
            )
        )

        assert isinstance(result, list)
        assert len(result) == 0


@pytest.mark.asyncio
async def test_execute_with_new_files(media_scanner: MediaScanner) -> None:
    """Test execute method with new files found."""
    # Generate a single UUID for predictability
    test_uuid = uuid.uuid4()

    with (
        patch.object(
            media_scanner,
            "_scan_directory",
            return_value=["/fake/path/file1.mp3", "/fake/path/file2.mp3"],
        ),
        patch.object(media_scanner, "_get_all_files", return_value=[]),
        patch.object(media_scanner, "_calculate_md5", return_value="fakehash"),
        patch.object(media_scanner, "_update_db", return_value=None),
        # Use a single UUID for all calls to ensure consistent behavior
        patch("uuid.uuid4", return_value=test_uuid),
    ):
        result = await media_scanner.execute(
            MediaScannerParams(
                dir_path="/fake/path",
                file_extensions=[".mp3"],
                media_type=MediaType.MUSIC,
            )
        )

        assert isinstance(result, list)
        assert len(result) == 4  # Two jobs per file (FILE_MATCHER and FFPROBE)

        # Group jobs by file path
        jobs_by_path: dict[str, list[ChildJobRequest]] = {}
        for job in result:
            path = (
                job.params.path if hasattr(job.params, "path") else job.params.file_id
            )
            if path not in jobs_by_path:
                jobs_by_path[path] = []
            jobs_by_path[path].append(job)

        # Verify we have jobs for both files
        assert set(jobs_by_path.keys()) == {
            "/fake/path/file1.mp3",
            "/fake/path/file2.mp3",
        }

        # Verify each file has both types of jobs
        for file_jobs in jobs_by_path.values():
            assert len(file_jobs) == 2
            job_types = {job.job_type for job in file_jobs}
            assert job_types == {JobType.FILE_MATCHER, JobType.FFPROBE}


@pytest.mark.asyncio
async def test_execute_with_existing_files(media_scanner: MediaScanner) -> None:
    """Test execute method with some existing files."""
    # Generate fixed UUIDs for predictability
    existing_uuid = uuid.uuid4()
    new_uuid = uuid.uuid4()

    existing_file = FileDTO(
        id=existing_uuid,
        path="/fake/path/file1.mp3",
        hash="fakehash",
        media_type=MediaType.MUSIC,
    )

    async def mock_calculate_md5(path: str) -> str:
        return "fakehash" if path == "/fake/path/file1.mp3" else "newhash"

    with (
        patch.object(
            media_scanner,
            "_scan_directory",
            return_value=["/fake/path/file1.mp3", "/fake/path/file2.mp3"],
        ),
        patch.object(media_scanner, "_get_all_files", return_value=[existing_file]),
        patch.object(media_scanner, "_calculate_md5", side_effect=mock_calculate_md5),
        patch.object(media_scanner, "_update_db", return_value=None),
        # Use a single new UUID for consistency
        patch("uuid.uuid4", return_value=new_uuid),
    ):
        result = await media_scanner.execute(
            MediaScannerParams(
                dir_path="/fake/path",
                file_extensions=[".mp3"],
                media_type=MediaType.MUSIC,
            )
        )

        assert isinstance(result, list)
        assert len(result) == 2  # Two jobs for the one new file

        # Verify both jobs are for the new file
        for job in result:
            if isinstance(job.params, FileMatcherParams):
                assert job.job_type == JobType.FILE_MATCHER
                assert job.params.path == "/fake/path/file2.mp3"
                assert job.params.media_type == MediaType.MUSIC
            else:  # FFProbeParams
                assert job.job_type == JobType.FFPROBE
                assert job.params.path == "/fake/path/file2.mp3"


@pytest.mark.asyncio
async def test_get_all_files_empty(
    media_scanner: MediaScanner, mock_db_session: MockAsyncDatabaseSession
) -> None:
    """Test _get_all_files method when no files exist."""
    mock_db_session.session.execute_result = []

    # We're intentionally accessing protected method for testing
    result = await media_scanner._get_all_files()
    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_get_all_files_with_data(
    media_scanner: MediaScanner, mock_db_session: MockAsyncDatabaseSession
) -> None:
    """Test _get_all_files method when files exist."""
    # Create mock File objects with real version 4 UUIDs
    file1 = MagicMock(spec=File)
    file1.id = uuid.uuid4()
    file1.path = "/path/to/file1.mp3"
    file1.hash = "hash1"
    file1.media_type = MediaType.MUSIC

    file2 = MagicMock(spec=File)
    file2.id = uuid.uuid4()
    file2.path = "/path/to/file2.mp3"
    file2.hash = "hash2"
    file2.media_type = MediaType.MUSIC

    mock_db_session.session.execute_result = [file1, file2]

    result = await media_scanner._get_all_files()
    assert isinstance(result, list)
    assert len(result) == 2

    assert result[0].id == file1.id
    assert result[0].path == "/path/to/file1.mp3"
    assert result[0].hash == "hash1"
    assert result[0].media_type == MediaType.MUSIC

    assert result[1].id == file2.id
    assert result[1].path == "/path/to/file2.mp3"
    assert result[1].hash == "hash2"
    assert result[1].media_type == MediaType.MUSIC


@pytest.mark.asyncio
async def test_update_db(
    media_scanner: MediaScanner, mock_db_session: MockAsyncDatabaseSession
) -> None:
    """Test _update_db method."""
    files = [
        FileDTO(
            id=uuid.uuid4(),
            path="/path/to/file1.mp3",
            hash="hash1",
            media_type=MediaType.MUSIC,
        ),
        FileDTO(
            id=uuid.uuid4(),
            path="/path/to/file2.mp3",
            hash="hash2",
            media_type=MediaType.MUSIC,
        ),
    ]

    await media_scanner._update_db(files)
    assert mock_db_session.session.add_called


@pytest.mark.asyncio
async def test_calculate_md5_file_not_found(media_scanner: MediaScanner) -> None:
    """Test _calculate_md5 method with non-existent file."""
    with patch("aiofiles.os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            # We're intentionally accessing protected method for testing
            await media_scanner._calculate_md5("/fake/path/nonexistent.mp3")


@pytest.mark.asyncio
async def test_calculate_md5_success(media_scanner: MediaScanner) -> None:
    """Test _calculate_md5 method with successful hash calculation."""
    test_content = b"test content"
    expected_hash = hashlib.md5(test_content).hexdigest()

    mock_file = AsyncMock()
    mock_file.read.side_effect = [test_content, b""]  # Return content then EOF

    with (
        patch("aiofiles.os.path.exists", return_value=True),
        patch(
            "aiofiles.open",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_file)),
        ),
    ):
        # We're intentionally accessing protected method for testing
        result = await media_scanner._calculate_md5("/fake/path/file.mp3")
        assert result == expected_hash


@pytest.mark.asyncio
async def test_scan_directory_not_exists(media_scanner: MediaScanner) -> None:
    """Test _scan_directory method with non-existent directory."""
    with patch("aiofiles.os.path.exists", return_value=False):
        # We're intentionally accessing protected method for testing
        result = await media_scanner._scan_directory("/fake/path", [".mp3"])
        assert isinstance(result, list)
        assert len(result) == 0


@pytest.mark.asyncio
async def test_scan_directory_with_matching_files(media_scanner: MediaScanner) -> None:
    """Test _scan_directory method with matching files."""
    mock_files = [
        ("root", [], ["file1.mp3", "file2.txt", "file3.mp3"]),
        ("root/subdir", [], ["file4.mp3", "file5.wav"]),
    ]

    with (
        patch("aiofiles.os.path.exists", return_value=True),
        patch("os.walk", return_value=mock_files),
        patch(
            "os.path.splitext",
            side_effect=lambda f: (
                cast(str, f.split(".")[0]),
                "." + cast(str, f.split(".")[1]),
            ),
        ),
        patch("os.path.join", side_effect=lambda root, file: f"{root}/{file}"),
    ):
        # We're intentionally accessing protected method for testing
        result = await media_scanner._scan_directory("/fake/path", [".mp3"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert "root/file1.mp3" in result
        assert "root/file3.mp3" in result
        assert "root/subdir/file4.mp3" in result
        assert "root/file2.txt" not in result
        assert "root/subdir/file5.wav" not in result


@pytest.mark.asyncio
async def test_scan_directory_with_no_extensions_filter(
    media_scanner: MediaScanner,
) -> None:
    """Test _scan_directory method with no extensions filter."""
    mock_files = [
        ("root", [], ["file1.mp3", "file2.txt", "file3.mp3"]),
        ("root/subdir", [], ["file4.mp3", "file5.wav"]),
    ]

    with (
        patch("aiofiles.os.path.exists", return_value=True),
        patch("os.walk", return_value=mock_files),
        patch(
            "os.path.splitext",
            side_effect=lambda f: (
                cast(str, f.split(".")[0]),
                "." + cast(str, f.split(".")[1]),
            ),
        ),
        patch("os.path.join", side_effect=lambda root, file: f"{root}/{file}"),
    ):
        # We're intentionally accessing protected method for testing
        result = await media_scanner._scan_directory("/fake/path", [])
        assert isinstance(result, list)
        assert len(result) == 5
        assert "root/file1.mp3" in result
        assert "root/file2.txt" in result
        assert "root/file3.mp3" in result
        assert "root/subdir/file4.mp3" in result
        assert "root/subdir/file5.wav" in result
