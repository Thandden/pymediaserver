from typing import Any
import uuid
import pytest
from datetime import datetime
from src.workers.file_matcher import FileMatcher
from src.common.system_types import MediaType
from src.common.db import AsyncDatabaseSession
from src.common.dto import MatchedData, FileMatcherParams

@pytest.fixture
def file_matcher() -> FileMatcher:
    """Create a FileMatcher instance for testing."""
    return FileMatcher(AsyncDatabaseSession(db_url="sqlite+aiosqlite:///"), None)

@pytest.mark.parametrize("test_input,expected", [
    # Movies with years in parentheses
    (
        "/media/movies/The Shawshank Redemption (1994)/shawshank.redemption.1080p.mkv",
        MatchedData(title="The Shawshank Redemption", year=1994, media_type=MediaType.MOVIE)
    ),
    (
        "/data/movies/The Matrix (1999) [1080p]/the.matrix.x264.mp4",
        MatchedData(title="The Matrix", year=1999, media_type=MediaType.MOVIE)
    ),
    (
        "/movies/12 Angry Men (1957).mkv",
        MatchedData(title="12 Angry Men", year=1957, media_type=MediaType.MOVIE)
    ),
    
    # Movies with years in filenames
    (
        "/media/films/inception.2010.bluray.x264.mkv",
        MatchedData(title="Inception", year=2010, media_type=MediaType.MOVIE)
    ),
    (
        "/media/Pulp.Fiction.1994.DVDRip.x264.mkv",
        MatchedData(title="Pulp Fiction", year=1994, media_type=MediaType.MOVIE)
    ),
    
    # Movies with special characters and quality tags
    (
        "/movies/Spider-Man.Far.From.Home.2019.2160p.WEB-DL.x265-[RARBG].mkv",
        MatchedData(title="Spider Man Far From Home", year=2019, media_type=MediaType.MOVIE)
    ),
])
def test_movie_matching(file_matcher: FileMatcher, test_input: str, expected: MatchedData) -> None:
    """Test movie file path matching."""
    result = file_matcher.match_file_path(test_input, MediaType.MOVIE)
    assert result.title == expected.title
    assert result.year == expected.year
    assert result.media_type == expected.media_type

@pytest.mark.parametrize("test_input,expected", [
    # TV Shows with years in directory
    (
        "/media/tv/Breaking Bad (2008)/Season.1/breaking.bad.s01e01.720p.mkv",
        MatchedData(title="Breaking Bad", year=2008, media_type=MediaType.TV)
    ),
    (
        "/shows/The Wire (2002)/season 1/the.wire.1x01.hdtv.x264.mp4",
        MatchedData(title="The Wire", year=2002, media_type=MediaType.TV)
    ),
    
    # TV Shows without years
    (
        "/media/series/Friends/friends.s01e01.720p.WEB-DL.mkv",
        MatchedData(title="Friends", year=None, media_type=MediaType.TV)
    ),
    (
        "/tv/Game of Thrones/S01/got.s01e01.1080p.mkv",
        MatchedData(title="Game Of Thrones", year=None, media_type=MediaType.TV)
    ),
    
    # TV Shows with special characters and nested directories
    (
        "/media/Stranger.Things/Season.01/ST.S01E01.2160p.HDR.Netflix.WEBRip.mkv",
        MatchedData(title="Stranger Things", year=None, media_type=MediaType.TV)
    ),
    (
        "/shows/Rick.and.Morty.S01.2160p/Rick.and.Morty.S01E01.HDR.mkv",
        MatchedData(title="Rick And Morty", year=None, media_type=MediaType.TV)
    ),
])
def test_tv_show_matching(file_matcher: FileMatcher, test_input: str, expected: MatchedData) -> None:
    """Test TV show file path matching."""
    result = file_matcher.match_file_path(test_input, MediaType.TV)
    assert result.title == expected.title
    assert result.year == expected.year
    assert result.media_type == expected.media_type

@pytest.mark.parametrize("test_input", [
    # Test various quality indicators
    "1080p", "720p", "2160p", "4K", "UHD",
    # Test codecs
    "x264", "x.264", "h.264", "HEVC",
    # Test audio formats
    "DTS", "AC3", "AAC", "DTS-HD",
    # Test source tags
    "BluRay", "WEB-DL", "BRRip", "HDRip",
    # Test release info
    "EXTENDED", "UNRATED", "PROPER", "REPACK",
    # Test other common patterns
    "5.1", "10bit", "[RARBG]", "{SPARKS}"
])
def test_pattern_cleaning(file_matcher: FileMatcher, test_input: str) -> None:
    """Test that common patterns are properly cleaned from titles."""
    result = file_matcher._clean_title(test_input)
    assert result.strip() == ""

def test_invalid_year_handling(file_matcher: FileMatcher) -> None:
    """Test handling of invalid years."""
    # Test future year
    future_year = datetime.now().year + 1
    result = file_matcher.match_file_path(
        f"/movies/Future Movie ({future_year}).mkv",
        MediaType.MOVIE
    )
    assert result.year is None
    
    # Test too old year
    result = file_matcher.match_file_path(
        "/movies/Ancient Movie (1799).mkv",
        MediaType.MOVIE
    )
    assert result.year is None

def test_nested_brackets_handling(file_matcher: FileMatcher) -> None:
    """Test handling of nested brackets and multiple year patterns."""
    result = file_matcher.match_file_path(
        "/movies/Inception [2010] (2010) {2010}.mkv",
        MediaType.MOVIE
    )
    assert result.title == "Inception"
    assert result.year == 2010

def test_multiple_dots_handling(file_matcher: FileMatcher) -> None:
    """Test handling of multiple dots and separators."""
    result = file_matcher.match_file_path(
        "/movies/The...Matrix...1999...Extended.mkv",
        MediaType.MOVIE
    )
    assert result.title == "The Matrix"
    assert result.year == 1999 

@pytest.mark.asyncio
async def test_correct_type_parameter(file_matcher: FileMatcher) -> None:
    """Test that correct type parameter is passed to metadata matcher."""
    await file_matcher.execute(FileMatcherParams(path="test", media_type=MediaType.MOVIE, file_id="6699163b-43ff-41d1-8ce9-0e25630b99f4"))

