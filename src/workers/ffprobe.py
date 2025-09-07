from typing import Optional, Any
import json
import asyncio
import uuid

from src.common.dto import (
    ChildJobRequest,
    FFProbeParams,
    MediaTechnicalInfoDTO,
    VideoTrackDTO,
    AudioTrackDTO,
)
from src.common.models import MediaTechnicalInfo, VideoTrack, AudioTrack
from src.workers.base import T_JobParams, Worker
from src.common.db import AsyncDatabaseSession
from src.common.logger import Logger
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession


class FFProbe(Worker):
    """Worker that extracts media technical information using ffprobe."""

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        super().__init__(db_session, logger)

    async def execute(
        self, parameters: Optional[T_JobParams] = None
    ) -> list[ChildJobRequest]:
        """Execute the ffprobe analysis process for a given file.

        Args:
            parameters (Optional[T_JobParams]): Must be FFProbeParams containing path and file_id.

        Returns:
            list[ChildJobRequest]: Empty list as this worker doesn't create child jobs.

        Raises:
            ValueError: If parameters is not of type FFProbeParams.
        """
        if not isinstance(parameters, FFProbeParams):
            raise ValueError("Parameters must be of type FFProbeParams")

        # Run ffprobe and parse results
        probe_data = await self._run_ffprobe(parameters.path)
        if not probe_data:
            if self.logger:
                self.logger.error(f"Failed to probe file: {parameters.path}")
            return []

        # Extract technical info from probe data, passing the file_id
        tech_info_dto = self._extract_technical_info(probe_data, parameters.path, parameters.file_id)
        
        # Save to database
        await self._save_technical_info(tech_info_dto)
        
        # Return empty list as we don't need to create child jobs
        return []

    async def _run_ffprobe(self, file_path: str) -> Optional[dict[str, Any]]:
        """Run ffprobe command on file path and return the parsed JSON output.

        Args:
            file_path (str): Path to the media file.

        Returns:
            Optional[dict[str, Any]]: Parsed ffprobe output as dictionary or None if failed.
        """
        try:
            # Construct ffprobe command with all needed information
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                file_path
            ]
            
            # Run ffprobe command asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                if self.logger:
                    self.logger.error(f"ffprobe error: {stderr.decode()}")
                return None
            
            # Parse JSON output
            return json.loads(stdout.decode())
        except Exception as e:
            if self.logger:
                self.logger.error(f"Exception running ffprobe: {str(e)}")
            return None

    def _extract_technical_info(
        self, probe_data: dict[str, Any], file_path: str, file_id: uuid.UUID
    ) -> MediaTechnicalInfoDTO:
        """Extract technical information from ffprobe output.

        Args:
            probe_data (dict[str, Any]): ffprobe output data.
            file_path (str): Path to the file.
            file_id (uuid.UUID): ID of the file from FFProbeParams.

        Returns:
            MediaTechnicalInfoDTO: Technical information DTO.
        """
        # Extract format information
        format_info = probe_data.get("format", {})
        
        # Create technical info DTO with the provided file_id
        tech_info = MediaTechnicalInfoDTO(
            file_id=file_id,  # Use the provided file_id instead of generating one
            duration=int(float(format_info.get("duration", "0")) * 1000) if "duration" in format_info else None,
            bitrate=int(format_info.get("bit_rate", "0")) if "bit_rate" in format_info else None,
            container_format=format_info.get("format_name"),
            codec_data=format_info.get("tags", {}),
            video_tracks=[],
            audio_tracks=[]
        )
        
        # Extract stream information
        streams = probe_data.get("streams", [])
        for stream_index, stream in enumerate(streams):
            codec_type = stream.get("codec_type")
            
            if codec_type == "video":
                video_track = self._extract_video_track(stream, stream_index, tech_info.id or uuid.uuid4())
                tech_info.video_tracks.append(video_track)
            elif codec_type == "audio":
                audio_track = self._extract_audio_track(stream, stream_index, tech_info.id or uuid.uuid4())
                tech_info.audio_tracks.append(audio_track)
        
        return tech_info

    def _extract_video_track(
        self, stream: dict[str, Any], index: int, technical_info_id: uuid.UUID
    ) -> VideoTrackDTO:
        """Extract video track information from a stream.

        Args:
            stream (dict[str, Any]): Stream data from ffprobe.
            index (int): Stream index.
            technical_info_id (uuid.UUID): ID of the parent technical info.

        Returns:
            VideoTrackDTO: Video track information.
        """
        # Calculate framerate
        frame_rate: Optional[float] = None
        if "r_frame_rate" in stream:
            try:
                num, den = stream["r_frame_rate"].split("/")
                frame_rate = float(num) / float(den) if float(den) != 0 else None
            except (ValueError, ZeroDivisionError):
                pass
        
        # Extract HDR format
        hdr_format: Optional[str] = None
        if "side_data_list" in stream:
            for side_data in stream["side_data_list"]:
                if side_data.get("side_data_type") == "Content light level metadata":
                    hdr_format = "HDR10"
                elif side_data.get("side_data_type") == "Mastering display metadata":
                    if hdr_format != "Dolby Vision":  # Prioritize Dolby Vision
                        hdr_format = "HDR10"
        
        tags = stream.get("tags", {})
        if any(tag.lower().startswith("dovi") for tag in tags.values()):
            hdr_format = "Dolby Vision"
        
        return VideoTrackDTO(
            technical_info_id=technical_info_id,
            track_index=index,
            width=stream.get("width"),
            height=stream.get("height"),
            codec=stream.get("codec_name"),
            frame_rate=frame_rate,
            bit_depth=int(stream.get("bits_per_raw_sample", "0")) if "bits_per_raw_sample" in stream else None,
            color_space=stream.get("color_space"),
            hdr_format=hdr_format,
            bitrate=int(stream.get("bit_rate", "0")) if "bit_rate" in stream else None,
            is_default=stream.get("disposition", {}).get("default", 0) == 1,
            metadata=stream.get("tags", {})
        )

    def _extract_audio_track(
        self, stream: dict[str, Any], index: int, technical_info_id: uuid.UUID
    ) -> AudioTrackDTO:
        """Extract audio track information from a stream.

        Args:
            stream (dict[str, Any]): Stream data from ffprobe.
            index (int): Stream index.
            technical_info_id (uuid.UUID): ID of the parent technical info.

        Returns:
            AudioTrackDTO: Audio track information.
        """
        tags = stream.get("tags", {})
        title = tags.get("title")
        
        return AudioTrackDTO(
            technical_info_id=technical_info_id,
            track_index=index,
            codec=stream.get("codec_name"),
            language=tags.get("language"),
            channels=stream.get("channels"),
            sample_rate=int(stream.get("sample_rate", "0")) if "sample_rate" in stream else None,
            bitrate=int(stream.get("bit_rate", "0")) if "bit_rate" in stream else None,
            title=title,
            is_default=stream.get("disposition", {}).get("default", 0) == 1,
            metadata=tags
        )

    async def _save_technical_info(self, tech_info_dto: MediaTechnicalInfoDTO) -> None:
        """Save technical information to the database.

        Args:
            tech_info_dto (MediaTechnicalInfoDTO): Technical information to save.
        """
        async for session in self.db_session.get_session():
            async with session.begin():
                # Check if technical info already exists for this file
                stmt = select(MediaTechnicalInfo).where(
                    MediaTechnicalInfo.file_id == tech_info_dto.file_id
                )
                existing_tech_info = await session.scalar(stmt)
                
                if existing_tech_info:
                    # Update existing technical info
                    existing_tech_info.duration = tech_info_dto.duration
                    existing_tech_info.bitrate = tech_info_dto.bitrate
                    existing_tech_info.container_format = tech_info_dto.container_format
                    existing_tech_info.codec_data = tech_info_dto.codec_data
                    
                    # Delete existing tracks
                    await session.execute(
                        delete(VideoTrack).where(
                            VideoTrack.technical_info_id == existing_tech_info.id
                        )
                    )
                    await session.execute(
                        delete(AudioTrack).where(
                            AudioTrack.technical_info_id == existing_tech_info.id
                        )
                    )
                    
                    technical_info_id = existing_tech_info.id
                else:
                    # Create new technical info
                    new_tech_info = MediaTechnicalInfo(
                        file_id=tech_info_dto.file_id,
                        duration=tech_info_dto.duration,
                        bitrate=tech_info_dto.bitrate,
                        container_format=tech_info_dto.container_format,
                        codec_data=tech_info_dto.codec_data
                    )
                    session.add(new_tech_info)
                    await session.flush()
                    technical_info_id = new_tech_info.id

                # Add video tracks
                for track_dto in tech_info_dto.video_tracks:
                    video_track = VideoTrack(
                        technical_info_id=technical_info_id,
                        track_index=track_dto.track_index,
                        width=track_dto.width,
                        height=track_dto.height,
                        codec=track_dto.codec,
                        frame_rate=track_dto.frame_rate,
                        bit_depth=track_dto.bit_depth,
                        color_space=track_dto.color_space,
                        hdr_format=track_dto.hdr_format,
                        bitrate=track_dto.bitrate,
                        is_default=track_dto.is_default,
                        metadata_info=track_dto.metadata
                    )
                    session.add(video_track)
                
                # Add audio tracks
                for track_dto in tech_info_dto.audio_tracks:
                    audio_track = AudioTrack(
                        technical_info_id=technical_info_id,
                        track_index=track_dto.track_index,
                        codec=track_dto.codec,
                        language=track_dto.language,
                        channels=track_dto.channels,
                        sample_rate=track_dto.sample_rate,
                        bitrate=track_dto.bitrate,
                        title=track_dto.title,
                        is_default=track_dto.is_default,
                        metadata_info=track_dto.metadata
                    )
                    session.add(audio_track)

    async def get_file_id_from_path(self, file_path: str) -> Optional[uuid.UUID]:
        """Get file ID from file path.

        Args:
            file_path (str): Path to the file.

        Returns:
            Optional[uuid.UUID]: File ID if found, None otherwise.
        """
        from src.common.models import File
        
        async with self.db_session.get_session() as session:
            stmt = select(File.id).where(File.path == file_path)
            result = await session.execute(stmt)
            file_id = result.scalar_one_or_none()
            return file_id 