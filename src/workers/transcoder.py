import asyncio
import os
import signal
from pathlib import Path
from typing import Any, Callable, Optional, cast
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.db import AsyncDatabaseSession
from src.common.dto import ChildJobRequest, NoChildJob, TranscodeParams
from src.common.ffmpeg_builder import ColorDepth, FFMpegCommandBuilder
from src.common.logger import Logger
from src.common.models import File, PlaybackSession, TranscodingSession
from src.common.types import TranscodeState
from src.workers.base import T_JobParams, Worker
from src.common.config import config


class Transcoder(Worker):
    """Worker that handles media transcoding sessions using FFmpeg."""

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        super().__init__(db_session, logger)
        self._ffmpeg_pid: int = 0
        self._task_running: bool = False
        self._ffmpeg_task: Optional[asyncio.Task[None]] = None
        self._watch_task: Optional[asyncio.Task[None]] = None

    async def execute(
        self, parameters: Optional[T_JobParams] = None
    ) -> list[ChildJobRequest] | NoChildJob:
        """Execute the transcoding process for a session.

        Args:
            parameters: Must be TranscodeParams containing transcode_session_id.

        Returns:
            NoChildJob: This worker doesn't create child jobs.

        Raises:
            ValueError: If parameters is not of type TranscodeParams.
        """
        if not isinstance(parameters, TranscodeParams):
            raise ValueError("Parameters must be of type TranscodeParams")

        session_id = parameters.transcode_session_id
        
        if self.logger:
            self.logger.info(f"Starting transcoder worker for session {session_id}")

        self._task_running = True
        
        try:
            while self._task_running:
                # Get the current transcode session state
                session_data = await self._get_transcode_session(session_id)
                
                if not session_data:
                    if self.logger:
                        self.logger.error(f"Transcoding session {session_id} not found")
                    return None
                
                # Handle the session based on its state
                match session_data["state"]:
                    case TranscodeState.PENDING:
                        if self.logger:
                            self.logger.info(f"Starting new transcoding session {session_id}")
                        
                        # Prepare output directory
                        output_dir = Path(session_data["output_path"]).parent
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Start transcoding and update state to ACTIVE
                        await self._start_transcoding_tasks(session_data)
                        await self._update_transcode_state(session_id, TranscodeState.ACTIVE)
                        
                    case TranscodeState.ACTIVE:
                        # Session is already active, just monitor
                        await asyncio.sleep(1)
                        
                    case TranscodeState.SEEK:
                        if self.logger:
                            self.logger.info(f"Seeking in transcoding session {session_id}")
                        
                        # Get playback position from the playback session
                        playback_data = await self._get_playback_session(session_data["playback_session_id"])
                        self.logger.debug(f"Playback data: {playback_data}")
                        if not playback_data:
                            continue
                        
                        # Stop current transcoding
                        await self._stop_transcoding_tasks()
                        
                        # Start new transcoding from the seek position
                        await self._start_transcoding_tasks(
                            session_data, 
                            start_timestamp=str(playback_data["current_position"])
                        )
                        
                        # Update state back to ACTIVE
                        await self._update_transcode_state(session_id, TranscodeState.ACTIVE)
                        
                    case TranscodeState.ERROR:
                        if self.logger:
                            self.logger.error(
                                f"Transcoding session {session_id} is in ERROR state: "
                                f"{session_data.get('error_message', 'Unknown error')}"
                            )
                        await self._stop_transcoding_tasks()
                        self._task_running = False
                        
                    case TranscodeState.INACTIVE:
                        if self.logger:
                            self.logger.info(f"Transcoding session {session_id} is INACTIVE")
                        await self._stop_transcoding_tasks()
                        self._task_running = False
                
                # Check if tasks are still running
                if self._ffmpeg_task and self._ffmpeg_task.done():
                    exception = self._ffmpeg_task.exception()
                    if exception:
                        if self.logger:
                            self.logger.error(f"FFmpeg task failed: {str(exception)}")
                        await self._update_transcode_state(
                            session_id, 
                            TranscodeState.ERROR, 
                            error_message=str(exception)
                        )
                        self._task_running = False
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in transcoder worker: {str(e)}")
            await self._update_transcode_state(
                session_id, 
                TranscodeState.ERROR, 
                error_message=str(e)
            )
            await self._stop_transcoding_tasks()
        finally:
            # Ensure tasks are properly cleaned up
            await self._stop_transcoding_tasks()
            
        return None

    async def _get_transcode_session(self, session_id: uuid.UUID) -> Optional[dict[str, Any]]:
        """Get transcoding session data from the database.

        Args:
            session_id: ID of the transcoding session.

        Returns:
            Optional[dict[str, Any]]: Session data or None if not found.
        """
        async for session in self.db_session.get_session():
            stmt = select(TranscodingSession).where(TranscodingSession.id == session_id)
            result = await session.execute(stmt)
            transcode_session = result.scalar_one_or_none()
            
            if transcode_session:
                # Include file path in the returned data
                file_stmt = select(File.path).where(File.id == transcode_session.file_id)
                file_result = await session.execute(file_stmt)
                file_path = file_result.scalar_one_or_none()
                
                return {
                    "id": transcode_session.id,
                    "playback_session_id": transcode_session.playback_session_id,
                    "file_id": transcode_session.file_id,
                    "state": transcode_session.state,
                    "current_segment": transcode_session.current_segment,
                    "target_segment": transcode_session.target_segment,
                    "output_path": transcode_session.output_path,
                    "target_codec": transcode_session.target_codec,
                    "target_resolution": transcode_session.target_resolution,
                    "target_bitrate": transcode_session.target_bitrate,
                    "error_message": transcode_session.error_message,
                    "file_path": file_path
                }
            return None

    async def _get_playback_session(self, session_id: uuid.UUID) -> Optional[dict[str, Any]]:
        """Get playback session data from the database.

        Args:
            session_id: ID of the playback session.

        Returns:
            Optional[dict[str, Any]]: Session data or None if not found.
        """
        async for session in self.db_session.get_session():
            stmt = select(PlaybackSession).where(PlaybackSession.id == session_id)
            result = await session.execute(stmt)
            playback_session = result.scalar_one_or_none()
            
            if playback_session:
                return {
                    "id": playback_session.id,
                    "current_position": playback_session.current_position,
                    "duration": playback_session.duration,
                    "watched_percentage": playback_session.watched_percentage
                }
            return None

    async def _update_transcode_state(
        self, 
        session_id: uuid.UUID, 
        state: TranscodeState, 
        error_message: Optional[str] = None
    ) -> None:
        """Update the state of a transcoding session.

        Args:
            session_id: ID of the transcoding session.
            state: New state to set.
            error_message: Optional error message for ERROR state.
        """
        async for session in self.db_session.get_session():
            update_values = {"state": state}
            if error_message is not None:
                update_values["error_message"] = error_message
                
            stmt = (
                update(TranscodingSession)
                .where(TranscodingSession.id == session_id)
                .values(**update_values)
            )
            await session.execute(stmt)

    async def _start_transcoding_tasks(
        self, 
        session_data: dict[str, Any], 
        start_timestamp: Optional[str] = None
    ) -> None:
        """Start the FFmpeg transcoding process and the segment watcher.

        Args:
            session_data: Transcoding session data.
            start_timestamp: Optional timestamp to start from (for seeking).
        """
        # Stop any existing tasks
        await self._stop_transcoding_tasks()
        
        # Use configured transcode directory + session ID for output
        transcode_base_path = Path(config.TRANSCODE_DIRECTORY)
        output_path = transcode_base_path / str(session_data["id"])
        
        # Delete existing directory if it exists and create a fresh one
        if output_path.exists():
            if self.logger:
                self.logger.info(f"Removing existing transcode directory: {output_path}")
            import shutil
            shutil.rmtree(output_path, ignore_errors=True)
        
        # Create fresh output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Update output path in session data
        session_data["output_path"] = str(output_path)
        
        # Create and start tasks
        self._ffmpeg_task = asyncio.create_task(
            self._run_ffmpeg(
                session_data,
                self._store_ffmpeg_pid,
                start_timestamp
            )
        )
        
        # Create an initial empty m3u8 file to ensure it exists immediately
        m3u8_path = output_path / f"{output_path.stem}.m3u8"
        with open(m3u8_path, "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            f.write("#EXT-X-PLAYLIST-TYPE:VOD\n")
            f.write("#EXT-X-TARGETDURATION:5\n")
            f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
        
        self._watch_task = asyncio.create_task(
            self._watch_segments(output_path, output_path.stem)
        )

    def _store_ffmpeg_pid(self, pid: int) -> None:
        """Callback to store the FFmpeg process ID.

        Args:
            pid: Process ID of the FFmpeg process.
        """
        self._ffmpeg_pid = pid
        if self.logger:
            self.logger.info(f"FFmpeg process started with PID: {pid}")

    async def _run_ffmpeg(
        self, 
        session_data: dict[str, Any], 
        pid_callback: Callable[[int], None],
        start_timestamp: Optional[str] = None
    ) -> None:
        """Run FFmpeg command to generate segments.
        
        Args:
            session_data: Dictionary containing transcoding session data.
            pid_callback: Callback function to store the FFmpeg process ID.
            start_timestamp: Optional timestamp to start from (for seeking).
        """
        try:
            input_path = str(Path(session_data["file_path"]).resolve())
            output_path = str(Path(session_data["output_path"]).resolve())
            
            if self.logger:
                self.logger.debug(f"Input path: {input_path}")
                self.logger.debug(f"Output path: {output_path}")
            
            # Parse resolution
            width, height = session_data["target_resolution"].split("x")
            width_int, height_int = int(width), int(height)
            
            # Use the FFMpegCommandBuilder to build the command
            builder = FFMpegCommandBuilder()
            
            # Set input file with optional seek
            builder.set_input_path(input_path)
            if start_timestamp:
                builder.set_start_timestamp(start_timestamp)
            
            # Add codec settings
            builder.set_video_codec(session_data["target_codec"])
            builder.set_audio_codec("aac")
            
            # Add resolution
            builder.set_resolution(width_int, height_int)
            
            # Set segment duration (5 seconds)
            builder.set_segment_duration(5)
            
            # Set output path for segments
            builder.set_output_path(str(Path(output_path) / Path(output_path).stem))
            
            # Get the final command as a list (safer for subprocess)
            cmd_list = builder.build_list()
            
            if self.logger:
                self.logger.debug(f"Command list: {cmd_list}")
            
            # Start FFmpeg process
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Call the PID callback
            pid_callback(process.pid)
            
            # Wait for process to complete
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0 and self.logger:
                self.logger.error(f"FFmpeg error: {stderr.decode() if stderr else 'Unknown error'}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error running FFmpeg: {str(e)}")
            raise

    async def _watch_segments(self, media_dir: Path, base_name: str) -> None:
        """Watch media directory for new segments and manage m3u8 playlist."""
        m3u8_path = media_dir / f"{base_name}.m3u8"
        
        try:
            # Get duration from session data or use a default
            # We should properly get this from the video file metadata in production
            total_duration = 6755.0  # Make this a float
            segment_duration = 5.0   # Make this a float
            total_segments = int(total_duration // segment_duration)  # Ensure integer division
            
            # Generate the complete m3u8 file
            with open(m3u8_path, "w") as f:
                f.write("#EXTM3U\n")
                f.write("#EXT-X-VERSION:3\n")
                f.write("#EXT-X-PLAYLIST-TYPE:VOD\n")
                f.write(f"#EXT-X-TARGETDURATION:{int(segment_duration)}\n")  # Duration must be integer
                f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
                
                for segment_num in range(total_segments):
                    segment_name = f"{base_name}_{segment_num:03d}.ts"
                    f.write(f"#EXTINF:{segment_duration:.1f},\n{segment_name}\n")  # Format duration with 1 decimal
                
                f.write("#EXT-X-ENDLIST\n")

            # Keep the task running but do nothing
            while self._task_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error watching segments: {str(e)}")
            raise

    async def _stop_transcoding_tasks(self) -> None:
        """Stop running transcoding tasks and kill FFmpeg process."""
        # Kill FFmpeg process if running
        if self._ffmpeg_pid > 0:
            try:
                if self.logger:
                    self.logger.info(f"Killing FFmpeg process with PID: {self._ffmpeg_pid}")
                os.kill(self._ffmpeg_pid, signal.SIGTERM)
            except ProcessLookupError:
                # Process already terminated
                pass
            finally:
                self._ffmpeg_pid = 0

        # Cancel tasks if running
        for task_name, task in [("FFmpeg", self._ffmpeg_task), ("Watch", self._watch_task)]:
            if task and not task.done():
                if self.logger:
                    self.logger.info(f"Cancelling {task_name} task")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
        self._ffmpeg_task = None
        self._watch_task = None
