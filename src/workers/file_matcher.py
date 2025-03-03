from typing import Optional
import re
from datetime import datetime

from src.common.dto import (
    MatchedData,
    FileMatcherParams,
    ChildJobRequest,
    MetadataMatcherParams,
)
from src.workers.base import T_JobParams, Worker
from src.common.db import AsyncDatabaseSession
from src.common.logger import Logger
from src.common.system_types import MediaType, JobType


class FileMatcher(Worker):
    """Worker that matches files to media titles based on their file paths."""

    # Common patterns to remove from titles
    CLEAN_PATTERNS = [
        # Video Quality and Bit Depth
        r"\b\d{3,4}p\b",  # 720p, 1080p, 2160p
        r"\b[48]k\b",  # 4k, 8k
        r"\b(?:UHD|HD|FHD)\b",  # UHD, HD, FHD
        r"\b(?:72|108|216)0\b",  # 720, 1080, 2160
        r"\b10[- ]?bit\b",  # 10bit, 10-bit
        # Video Codecs
        r"\bx[.-]?26[45]\b",  # x264, x.264, x-264, x265
        r"\bxvid\b",  # xvid
        r"\bhevc\b",  # HEVC
        r"\bh[.-]?26[45]\b",  # h264, h.264, h265
        r"\bavc\b",  # AVC
        r"\bmpeg[-._ ]?\d?\b",  # mpeg, mpeg2, mpeg-4
        # Audio
        r"\baac(?:2\.0)?\b",  # AAC, AAC2.0
        r"\bac3\b",  # AC3
        r"\beac3\b",  # EAC3
        r"\bdts(?:-hd)?\b",  # DTS, DTS-HD
        r"\bdd[.-]?5[.-]1\b",  # DD5.1, DD.5.1
        r"\b5[.-]1\b",  # 5.1
        r"\b7[.-]1\b",  # 7.1
        r"\batmos\b",  # ATMOS
        r"\bflac\b",  # FLAC
        r"\bma\b",  # MA (Master Audio)
        # Source/Quality Tags
        r"\bblu[- ]?ray\b",  # BluRay, Blu-ray
        r"\b(?:bd|br|dvd)[- ]?rip\b",  # BDRip, BRRip, DVDRip
        r"\bdvd(?:scr)?\b",  # DVD, DVDScr
        r"\bweb[- ]?(?:dl|rip)\b",  # WEB-DL, WEBDL, WEBRip
        r"\bhd[- ]?rip\b",  # HDRip
        r"\bhdr(?:10)?\+?\b",  # HDR, HDR10, HDR10+
        r"\bdolby\b",  # Dolby
        r"\bremux\b",  # REMUX
        r"\bproper\b",  # PROPER
        r"\brepack\b",  # REPACK
        r"\bunrated\b",  # UNRATED
        r"\bextended\b",  # EXTENDED
        r"\bdirectors?[. ]?cut\b",  # Directors.Cut, Director's Cut
        r"\bdc\b",  # DC (Directors Cut)
        # TV Show Patterns
        r"\bS\d{1,2}E\d{1,2}\b",  # S01E01
        r"\bs\d{1,2}e\d{1,2}\b",  # s01e01
        r"\b\d{1,2}x\d{1,2}\b",  # 1x01
        r"\bSeason[. ]?\d+\b",  # Season1, Season.1
        r"\bS\d{1,2}\b",  # S01
        # Release Groups & Other
        r"\[[^\]]+\]",  # Anything in []
        r"\{[^}]+\}",  # Anything in {}
        r"\([^)]*\)",  # Anything in ()
        r"\brus\b",  # RUS
        r"\beng\b",  # ENG
        r"\bsubs?\b",  # sub, subs
        r"\bcc\b",  # CC (Closed Captions)
        r"\bsdr\b",  # SDR
        # File Extensions
        r"\.(?:mp4|mkv|avi|mov|wmv|flv|webm|m4v|mpg|mpeg|iso)$",
    ]

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        super().__init__(db_session, logger)

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract a year between 1800 and current year from the given text.

        Args:
            text (str): The text to search for a year.

        Returns:
            Optional[int]: The extracted year if found and valid, None otherwise.

        Note:
            Sets _last_matched_year_pattern and _last_matched_year as instance attributes
            when a valid year is found.
        """
        year_patterns = [
            r"\((\d{4})\)",  # (1999)
            r"[. ](\d{4})[. ]",  # .1999. or space 1999 space
            r"[. ](\d{4})(?=[. ]|$)",  # .1999 at end or before dot/space
            r"\.(\d{4})\.",  # .1999.
            r"(?:^|\s)(\d{4})(?:\s|$)",  # 1999 surrounded by spaces or at start/end
        ]

        current_year = datetime.now().year

        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                if 1800 <= year <= current_year:
                    # Remove the year from the text to help with title cleaning
                    self._last_matched_year_pattern = pattern
                    self._last_matched_year = year
                    return year

        return None

    def _clean_title(self, title: str) -> str:
        """Clean and normalize a media title by removing common patterns and formatting.

        Args:
            title (str): The raw title string to clean.

        Returns:
            str: The cleaned and normalized title in Title Case.

        Note:
            Uses _last_matched_year_pattern and _last_matched_year if available to
            remove previously detected years from the title.
        """
        # Convert to lowercase for better pattern matching
        cleaned = title.lower()

        # Remove the year if we found one
        if hasattr(self, "_last_matched_year_pattern") and hasattr(
            self, "_last_matched_year"
        ):
            year_str = str(self._last_matched_year)
            # Remove the year and its surrounding characters
            cleaned = re.sub(self._last_matched_year_pattern, " ", cleaned)
            # Also try to remove the year if it appears differently
            cleaned = re.sub(rf"\b{year_str}\b", " ", cleaned)

        # Apply all cleaning patterns
        for pattern in self.CLEAN_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        # Replace dots, underscores, hyphens with spaces
        cleaned = cleaned.replace(".", " ").replace("_", " ").replace("-", " ")

        # Remove multiple spaces and trim
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Title case for final result
        return cleaned.title()

    def _extract_season_episode(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """Extract season and episode numbers from text.

        Args:
            text (str): The text to search for season and episode numbers.

        Returns:
            tuple[Optional[int], Optional[int]]: Season and episode numbers if found.
        """
        # Common episode patterns
        patterns = [
            # S01E01 format (most common)
            r"[Ss](\d{1,2})[Ee](\d{1,2})",
            # 1x01 format
            r"(\d{1,2})x(\d{1,2})",
            # season1/episode1 format
            r"season[. ]?(\d{1,2}).*?episode[. ]?(\d{1,2})",
            # Folder structure implied patterns - will be handled separately
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    season = int(match.group(1))
                    episode = int(match.group(2))
                    return season, episode
                except (ValueError, IndexError):
                    continue

        return None, None

    def _match_file_path(self, file_path: str, media_type: MediaType) -> MatchedData:
        """Extract title and year information from a media file path.

        Args:
            file_path (str): The full path to the media file.
            media_type (MediaType): The type of media (TV or MOVIE).

        Returns:
            MatchedData: Object containing the extracted title, year, and media type.

        Note:
            For TV shows, tries to extract show name from grandparent directory first,
            then parent directory, then filename.
            For movies, prefers parent directory with year, then directory after movies/films,
            then filename.
        """
        # Reset year matching state
        if hasattr(self, "_last_matched_year_pattern"):
            delattr(self, "_last_matched_year_pattern")
        if hasattr(self, "_last_matched_year"):
            delattr(self, "_last_matched_year")

        # Split path into parts and get relevant parts
        parts = file_path.split("/")
        filename = parts[-1]
        parent_dir = parts[-2] if len(parts) > 1 else ""
        grandparent_dir = parts[-3] if len(parts) > 2 else ""

        # Try to extract year from various parts
        year = None
        for part in [parent_dir, grandparent_dir, filename]:
            year = self._extract_year(part)
            if year:
                break

        season_number = None
        episode_number = None

        # TV Show specific handling
        if media_type == MediaType.TV:
            # Try to extract season/episode from filename first
            season_number, episode_number = self._extract_season_episode(filename)

            # If not found in filename, check parent directory for season info
            if season_number is None and parent_dir:
                # Check for 'Season X' or 'SXX' pattern in parent directory
                season_match = re.search(
                    r"[Ss]eason[. ]?(\d{1,2})", parent_dir, re.IGNORECASE
                )
                if not season_match:
                    season_match = re.search(r"^[Ss](\d{1,2})$", parent_dir)

                if season_match:
                    season_number = int(season_match.group(1))

                    # Look for episode in filename again now that we know it's in a season folder
                    _, ep_maybe = self._extract_season_episode(filename)
                    if ep_maybe:
                        episode_number = ep_maybe
                    else:
                        # Try to find just episode number (e.g., "01.mkv" or "E01")
                        ep_match = re.search(r"[Ee]?(\d{1,2})(?:\.\w+)?$", filename)
                        if ep_match:
                            episode_number = int(ep_match.group(1))

            # Try to get show name from grandparent directory first
            if grandparent_dir and grandparent_dir.lower() not in {
                "tv",
                "shows",
                "series",
                "television",
            }:
                title = grandparent_dir
            # Then try parent directory
            elif parent_dir and not re.search(
                r"^(?:[Ss]eason[. ]?\d+|[Ss]\d+)$", parent_dir, re.IGNORECASE
            ):
                title = parent_dir
            # Finally try filename
            else:
                title = filename

        else:  # Movie handling
            # For movies, prefer parent directory name if it has a year
            if re.search(r"\(\d{4}\)", parent_dir):
                title = parent_dir
            # Next try the directory containing "movies" or "films"
            elif any(x.lower() in ["movies", "films"] for x in parts[:-1]):
                movie_dir_index = next(
                    (
                        i
                        for i, x in enumerate(parts)
                        if x.lower() in ["movies", "films"]
                    ),
                    -1,
                )
                if movie_dir_index != -1 and len(parts) > movie_dir_index + 1:
                    title = parts[movie_dir_index + 1]
                else:
                    title = filename
            # Finally fall back to filename
            else:
                title = filename
        # Clean the title
        title = self._clean_title(title)

        return MatchedData(
            title=title,
            year=year,
            media_type=media_type,
            season_number=season_number,
            episode_number=episode_number,
        )

    async def execute(
        self, parameters: Optional[T_JobParams] = None
    ) -> list[ChildJobRequest]:
        """Execute the file matching process for a given file.

        Args:
            parameters (Optional[T_JobParams]): Must be FileMatcherParams containing file_id
                and media_type.

        Returns:
            list[ChildJobRequest]: List containing a metadata matcher job request if successful,
                empty list otherwise.

        Raises:
            ValueError: If parameters is not of type FileMatcherParams.
        """
        if not isinstance(parameters, FileMatcherParams):
            raise ValueError("Parameters must be of type FileMatcherParams")

        # Match file path to extract metadata
        matched_data = self._match_file_path(parameters.path, parameters.media_type)

        # Create child job for metadata matching
        return [
            ChildJobRequest(
                job_type=JobType.METADATA_MATCHER,
                params=MetadataMatcherParams(
                    matched_data=matched_data,
                    file_id=parameters.file_id,
                ),
            )
        ]

    def match_file_path(self, file_path: str, media_type: MediaType) -> MatchedData:
        """Public method to extract media information from a file path.

        Args:
            file_path (str): The full path to the media file.
            media_type (MediaType): The type of media (TV or MOVIE).

        Returns:
            MatchedData: Object containing the extracted title, year, and media type.
        """
        return self._match_file_path(file_path, media_type)
