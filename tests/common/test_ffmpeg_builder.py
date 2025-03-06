from pathlib import Path
from pytest import raises
from src.common.ffmpeg_builder import (
    FFMpegCommandBuilder,
    ColorDepth,
    InvalidCodecConfigurationError,
)


def test_minimal_valid_command() -> None:
    """Test minimal valid command with just video codec"""
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .build()
    )
    assert cmd == 'ffmpeg -i "input.mp4" -c:v libx264 -f segment "output_%03d.ts"'


def test_full_command_all_options() -> None:
    """Test builder with all options set"""
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx265")
        .set_audio_codec("aac")
        .set_resolution(1920, 1080)
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_segment_duration(4)
        .set_color_depth(ColorDepth.BIT_10)
        .set_quality_preset("slow")
        .build()
    )
    expected = (
        'ffmpeg -i "input.mp4" -c:v libx265 -vf scale=1920x1080 '
        "-pix_fmt yuv420p10le -preset slow -c:a aac -f segment "
        '-segment_time 4 "output_%03d.ts"'
    )
    assert cmd == expected


def test_audio_only_command() -> None:
    """Test command with only audio codec"""
    cmd = (
        FFMpegCommandBuilder()
        .set_audio_codec("aac")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .build()
    )
    assert cmd == 'ffmpeg -i "input.mp4" -c:a aac -f segment "output_%03d.ts"'


def test_color_depth_options() -> None:
    """Test different color depth settings"""
    # Test 8-bit
    cmd_8bit = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_color_depth(ColorDepth.BIT_8)
        .build()
    )
    assert "-pix_fmt yuv420p" in cmd_8bit

    # Test 10-bit
    cmd_10bit = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_color_depth(ColorDepth.BIT_10)
        .build()
    )
    assert "-pix_fmt yuv420p10le" in cmd_10bit

    # Test no color depth specified
    cmd_no_depth = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .build()
    )
    assert "pix_fmt" not in cmd_no_depth


def test_paths_with_spaces() -> None:
    """Test handling of paths containing spaces"""
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input file.mp4")
        .set_output_path("output folder/segment")
        .build()
    )
    assert 'ffmpeg -i "input file.mp4"' in cmd
    assert '"output folder/segment_%03d.ts"' in cmd


def test_resolution_settings() -> None:
    """Test different resolution settings"""
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_resolution(1280, 720)
        .build()
    )
    assert "-vf scale=1280x720" in cmd


def test_segment_duration() -> None:
    """Test segment duration settings"""
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_segment_duration(10)
        .build()
    )
    assert "-segment_time 10" in cmd


def test_quality_presets() -> None:
    """Test different quality presets"""
    presets = [
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ]

    for preset in presets:
        cmd = (
            FFMpegCommandBuilder()
            .set_video_codec("libx264")
            .set_input_path("input.mp4")
            .set_output_path("output")
            .set_quality_preset(preset)
            .build()
        )
        assert f"-preset {preset}" in cmd


# Error cases
def test_missing_input_path() -> None:
    """Test error when input path is missing"""
    builder = (
        FFMpegCommandBuilder().set_video_codec("libx264").set_output_path("output")
    )
    with raises(ValueError, match="Both input and output paths must be specified"):
        builder.build()


def test_missing_output_path() -> None:
    """Test error when output path is missing"""
    builder = (
        FFMpegCommandBuilder().set_video_codec("libx264").set_input_path("input.mp4")
    )
    with raises(ValueError, match="Both input and output paths must be specified"):
        builder.build()


def test_missing_codecs() -> None:
    """Test error when neither video nor audio codec is specified"""
    builder = (
        FFMpegCommandBuilder().set_input_path("input.mp4").set_output_path("output")
    )
    with raises(InvalidCodecConfigurationError):
        builder.build()


def test_builder_immutability() -> None:
    """Test that builder operations don't affect other instances"""
    builder1 = FFMpegCommandBuilder().set_video_codec("libx264")
    builder2 = FFMpegCommandBuilder().set_video_codec("libx265")

    assert "libx264" in builder1.set_input_path("in.mp4").set_output_path("out").build()
    assert "libx265" in builder2.set_input_path("in.mp4").set_output_path("out").build()


def test_path_normalization() -> None:
    """Test that paths with different separators work correctly"""
    paths = [
        ("folder/subfolder/file.mp4", "folder/subfolder/output"),
        (r"folder\subfolder\file.mp4", r"folder\subfolder\output"),
        (Path("folder/subfolder/file.mp4"), Path("folder/subfolder/output")),
    ]

    for input_path, output_path in paths:
        cmd = (
            FFMpegCommandBuilder()
            .set_video_codec("libx264")
            .set_input_path(str(input_path))
            .set_output_path(str(output_path))
            .build()
        )
        assert "-i" in cmd
        assert "_%03d.ts" in cmd


def test_empty_strings() -> None:
    """Test that empty strings are handled appropriately"""
    builder = FFMpegCommandBuilder()

    with raises(ValueError):
        builder.set_video_codec("")

    with raises(ValueError):
        builder.set_audio_codec("")

    with raises(ValueError):
        builder.set_input_path("")

    with raises(ValueError):
        builder.set_output_path("")


def test_start_timestamp() -> None:
    """Test setting start timestamp in different formats"""
    # Test with HH:MM:SS format
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_start_timestamp("01:30:45")
        .build()
    )
    assert "ffmpeg -ss 01:30:45 -i" in cmd

    # Test with seconds format
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_start_timestamp("90")
        .build()
    )
    assert "ffmpeg -ss 90 -i" in cmd

    # Test with MM:SS format
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx264")
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_start_timestamp("05:30")
        .build()
    )
    assert "ffmpeg -ss 05:30 -i" in cmd


def test_empty_timestamp() -> None:
    """Test that empty timestamp raises ValueError"""
    builder = FFMpegCommandBuilder()

    with raises(ValueError, match="Start timestamp cannot be empty"):
        builder.set_start_timestamp("")


def test_timestamp_with_full_command() -> None:
    """Test timestamp in a full command with all options"""
    cmd = (
        FFMpegCommandBuilder()
        .set_video_codec("libx265")
        .set_audio_codec("aac")
        .set_resolution(1920, 1080)
        .set_input_path("input.mp4")
        .set_output_path("output")
        .set_segment_duration(4)
        .set_color_depth(ColorDepth.BIT_10)
        .set_quality_preset("slow")
        .set_start_timestamp("00:05:00")
        .build()
    )
    expected = (
        'ffmpeg -ss 00:05:00 -i "input.mp4" -c:v libx265 -vf scale=1920x1080 '
        "-pix_fmt yuv420p10le -preset slow -c:a aac -f segment "
        '-segment_time 4 "output_%03d.ts"'
    )
    assert cmd == expected
