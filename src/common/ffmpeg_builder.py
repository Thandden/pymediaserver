from typing import Optional, Self, List, Union, Literal
from enum import Enum
from dataclasses import dataclass
import math


class FFMpegCommandBuilderError(Exception):
    """Base exception for command builder errors"""


class InvalidCodecConfigurationError(FFMpegCommandBuilderError):
    """Raised when neither video nor audio codec is specified"""


class ColorDepth(str, Enum):
    """Enum representing supported color depths"""

    BIT_8 = "8bit"
    BIT_10 = "10bit"


class FFMpegCommandBuilder:
    def __init__(self) -> None:
        self._video_codec: Optional[str] = None
        self._audio_codec: Optional[str] = None
        self._resolution: Optional[str] = None
        self._input_path: Optional[str] = None
        self._output_path: Optional[str] = None
        self._segment_duration: Optional[int] = None
        self._color_depth: Optional[ColorDepth] = None
        self._quality_preset: Optional[str] = None
        self._start_timestamp: Optional[str] = None
        self._hls_start_number: Optional[int] = None

    def set_video_codec(self, codec: str) -> Self:
        """Set video codec (e.g., h264, hevc)"""
        if not codec:
            raise ValueError("Video codec cannot be empty")
        self._video_codec = codec
        return self

    def set_audio_codec(self, codec: str) -> Self:
        """Set audio codec (e.g., aac, mp3)"""
        if not codec:
            raise ValueError("Audio codec cannot be empty")
        self._audio_codec = codec
        return self

    def set_resolution(self, width: int, height: int) -> Self:
        """Set output resolution in WxH format"""
        self._resolution = f"{width}x{height}"
        return self

    def set_input_path(self, path: str) -> Self:
        """Set input file path"""
        if not path:
            raise ValueError("Input path cannot be empty")
        self._input_path = path
        return self

    def set_output_path(self, path: str) -> Self:
        """Set output segment path pattern.
        The path should be the base path without the segment pattern or extension.
        E.g., for 'media/segment' will generate 'media/segment_%03d.ts'
        """
        if not path:
            raise ValueError("Output path cannot be empty")
        self._output_path = path
        return self

    def set_segment_duration(self, seconds: int) -> Self:
        """Set segment duration in seconds"""
        self._segment_duration = seconds
        return self

    def set_color_depth(self, depth: ColorDepth) -> Self:
        """Set output color depth (8 or 10 bit)"""
        self._color_depth = depth
        return self

    def set_quality_preset(self, preset: str) -> Self:
        """Set quality preset (e.g., 'slow', 'medium', 'fast')"""
        self._quality_preset = preset
        return self

    def set_start_timestamp(self, timestamp: str) -> Self:
        """Set the timestamp where the stream should start.

        Args:
            timestamp (str): Timestamp in FFmpeg format (e.g., '00:01:30' for 1m30s)
                             Can also be in seconds (e.g., '90')

        Returns:
            Self: Builder instance for method chaining
        """
        if not timestamp:
            raise ValueError("Start timestamp cannot be empty")
        self._start_timestamp = timestamp
        return self

    def calculate_hls_start_number(self) -> int:
        """Calculate the HLS start segment number based on start timestamp and segment duration.
        
        Returns:
            int: The calculated segment number. Defaults to 1 if timestamp or segment duration not set.
        """
        if not self._start_timestamp or not self._segment_duration:
            return 1
            
        # Convert timestamp to seconds if it's in HH:MM:SS format
        timestamp_seconds = self._convert_timestamp_to_seconds(self._start_timestamp)
        
        # Calculate segment number as timestamp / segment_duration + 1
        segment_number = (timestamp_seconds // self._segment_duration) + 1
        return int(segment_number)
    
    def _convert_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp from HH:MM:SS or SS format to seconds.
        
        Args:
            timestamp (str): Timestamp in FFmpeg format (e.g., '00:01:30' or '90')
            
        Returns:
            float: Timestamp converted to seconds
        """
        # Check if timestamp is already in seconds
        if timestamp.isdigit():
            return float(timestamp)
            
        # Parse HH:MM:SS format
        parts = timestamp.split(':')
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        else:
            # Default case if format is unexpected
            return 0.0
    
    def set_hls_start_number(self, number: int) -> Self:
        """Manually set the HLS start segment number.
        
        Args:
            number (int): The segment number to start with
            
        Returns:
            Self: Builder instance for method chaining
        """
        self._hls_start_number = number
        return self

    def build(self) -> str:
        """Build and validate the FFmpeg command string
        
        Returns:
            str: Command as shell-compatible string
            
        Raises:
            InvalidCodecConfigurationError: If neither video nor audio codec is specified
            ValueError: If input or output paths are not specified
        """
        return self.build_string()
    
    def build_string(self) -> str:
        """Build the FFmpeg command as a shell-compatible string
        
        Returns:
            str: Command as shell-compatible string
            
        Raises:
            InvalidCodecConfigurationError: If neither video nor audio codec is specified
            ValueError: If input or output paths are not specified
        """
        command_list = self._build_command_list()
        return " ".join(command_list)
    
    def build_list(self) -> List[str]:
        """Build the FFmpeg command as a list of arguments (safer for subprocess)
        
        Returns:
            List[str]: List of command components
            
        Raises:
            InvalidCodecConfigurationError: If neither video nor audio codec is specified
            ValueError: If input or output paths are not specified
        """
        return self._build_command_list()
    
    def _build_command_list(self) -> List[str]:
        """Build the FFmpeg command as a list of arguments"""
        if not self._video_codec and not self._audio_codec:
            raise InvalidCodecConfigurationError(
                "At least one of video or audio codec must be specified"
            )

        if not self._input_path or not self._output_path:
            raise ValueError("Both input and output paths must be specified")

        cmd: List[str] = ["ffmpeg"]

        # Add start timestamp if specified
        if self._start_timestamp:
            cmd.extend(["-ss", self._start_timestamp])

        cmd.extend(["-i", self._input_path])

        # Video processing chain
        if self._video_codec:
            cmd.extend(["-c:v", self._video_codec])

            if self._resolution:
                cmd.extend(["-vf", f"scale={self._resolution}"])

            if self._color_depth == ColorDepth.BIT_10:
                cmd.extend(["-pix_fmt", "yuv420p10le"])
            elif self._color_depth == ColorDepth.BIT_8:
                cmd.extend(["-pix_fmt", "yuv420p"])

            if self._quality_preset:
                cmd.extend(["-preset", self._quality_preset])

        # Audio processing
        if self._audio_codec:
            cmd.extend(["-c:a", self._audio_codec])

        # Segment configuration
        cmd.extend(["-f", "segment"])
        if self._segment_duration:
            cmd.extend(["-segment_time", str(self._segment_duration)])

        # Add segment numbering parameters if timestamp is provided
        if self._start_timestamp:
            cmd.extend(["-segment_format", "mpegts"])
            
            # Use manually set start number if available, otherwise calculate it
            start_number = self._hls_start_number if self._hls_start_number is not None else self.calculate_hls_start_number()
            cmd.extend(["-segment_start_number", str(start_number)])

        # Add output pattern
        cmd.append(f"{self._output_path}_%03d.ts")

        return cmd

@dataclass
class M3U8Generator:
    """Generates a complete HLS manifest file."""
    
    duration_seconds: float
    segment_duration: int
    session_id: str
    
    def generate_manifest(self) -> str:
        """Generate a complete HLS manifest with all segments."""
        total_segments = math.ceil(self.duration_seconds / self.segment_duration)
        
        manifest = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            "#EXT-X-TARGETDURATION:" + str(self.segment_duration),
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:VOD"
        ]
        
        for i in range(total_segments):
            manifest.append(f"#EXTINF:{self.segment_duration},")
            # Use zero-padded segment numbers (000, 001, etc.)
            manifest.append(f"{self.session_id}_{i:03d}.ts")
        
        manifest.append("#EXT-X-ENDLIST")
        return "\n".join(manifest)
