from __future__ import annotations

import enum
import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional, Type

from sqlalchemy import (
    CHAR,
    ForeignKey,
    UniqueConstraint,
    CheckConstraint,
    Index,
    String,
    Enum,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from src.common.system_types import (
    MediaType,
    EntityType,
    FileStatus,
    MetadataStatus,
    TranscodingSessionStatus,
    JobStatus,
    ServiceStatus,
    ServiceCommand,
    JobType,
    ServiceType,
)


class GUID(TypeDecorator[uuid.UUID]):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.
    """

    impl = CHAR

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(
        self, value: Optional[uuid.UUID], dialect: Any
    ) -> Optional[str]:
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return str(value)
        else:
            try:
                return "%.32x" % uuid.UUID(str(value)).int
            except (AttributeError, ValueError, TypeError):
                # Already a UUID
                return "%.32x" % value.int

    def process_result_value(
        self, value: Optional[str], dialect: Any
    ) -> Optional[uuid.UUID]:
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


class Base(DeclarativeBase):
    pass


# First, let's define a helper function to create enum columns consistently
def enum_column(enum_class: Type[enum.Enum], **kwargs: Any) -> Mapped[Any]:
    """Helper function to create enum columns with consistent behavior"""

    # Create a function that captures enum_class properly
    def get_values(x: Any) -> list[str]:
        return [e.value for e in enum_class]

    return mapped_column(
        Enum(
            enum_class,
            native_enum=False,
            values_callable=get_values,
        ),
        **kwargs,
    )


class File(Base):
    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    path: Mapped[str] = mapped_column(String, nullable=False)
    media_type: Mapped[MediaType] = mapped_column(
        Enum(MediaType, name="mediatype_enum"), nullable=False
    )
    indexed: Mapped[bool] = mapped_column(default=False)
    hash: Mapped[Optional[str]]
    status: Mapped[FileStatus] = enum_column(FileStatus, default=FileStatus.UNINDEXED)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    entities: Mapped[list[Entity]] = relationship(back_populates="file")
    technical_info: Mapped[Optional[MediaTechnicalInfo]] = relationship(
        back_populates="file", uselist=False
    )
    transcoding_sessions: Mapped[list[TranscodingSession]] = relationship(
        back_populates="file"
    )


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), type_=GUID
    )
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType, name="entitytype_enum", native_enum=False), nullable=False
    )
    movie_id: Mapped[Optional[str]] = mapped_column(ForeignKey("movies.id"))
    tv_episode_id: Mapped[Optional[str]] = mapped_column(ForeignKey("tv_episodes.id"))
    matched_data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    metadata_status: Mapped[MetadataStatus] = enum_column(
        MetadataStatus, default=MetadataStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    file: Mapped[File] = relationship(back_populates="entities")
    movie: Mapped[Optional[Movie]] = relationship(back_populates="entities")
    tv_episode: Mapped[Optional[TVEpisode]] = relationship(back_populates="entities")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            """
            (entity_type = 'MOVIE' AND movie_id IS NOT NULL AND tv_episode_id IS NULL) OR
            (entity_type = 'TV_EPISODE' AND movie_id IS NULL AND tv_episode_id IS NOT NULL)
            """,
            name="valid_entity_reference",
        ),
    )


class TVShow(Base):
    __tablename__ = "tv_shows"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tmdb_id: Mapped[Optional[int]] = mapped_column(unique=True)
    title: Mapped[str]
    overview: Mapped[Optional[str]]
    poster_path: Mapped[Optional[str]]
    backdrop_path: Mapped[Optional[str]]
    year: Mapped[Optional[int]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    seasons: Mapped[list[TVSeason]] = relationship(
        back_populates="show", cascade="all, delete-orphan"
    )


class TVSeason(Base):
    __tablename__ = "tv_seasons"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    show_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tv_shows.id", ondelete="CASCADE"), type_=GUID
    )
    season_number: Mapped[int]
    title: Mapped[Optional[str]]
    overview: Mapped[Optional[str]]
    poster_path: Mapped[Optional[str]]
    year: Mapped[Optional[int]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    show: Mapped[TVShow] = relationship(back_populates="seasons")
    episodes: Mapped[list[TVEpisode]] = relationship(
        back_populates="season", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("show_id", "season_number", name="uq_season_number_per_show"),
    )


class TVEpisode(Base):
    __tablename__ = "tv_episodes"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    season_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tv_seasons.id", ondelete="CASCADE"), type_=GUID
    )
    episode_number: Mapped[int]
    title: Mapped[Optional[str]]
    overview: Mapped[Optional[str]]
    still_path: Mapped[Optional[str]]
    air_date: Mapped[Optional[date]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    season: Mapped[TVSeason] = relationship(back_populates="episodes")
    entities: Mapped[list[Entity]] = relationship(back_populates="tv_episode")

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "season_id", "episode_number", name="uq_episode_number_per_season"
        ),
    )


class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    tmdb_id: Mapped[Optional[int]] = mapped_column(unique=True)
    title: Mapped[str]
    overview: Mapped[Optional[str]]
    poster_path: Mapped[Optional[str]]
    backdrop_path: Mapped[Optional[str]]
    release_date: Mapped[Optional[date]]
    year: Mapped[Optional[int]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    entities: Mapped[list[Entity]] = relationship(back_populates="movie")


class MediaTechnicalInfo(Base):
    __tablename__ = "media_technical_info"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))
    duration: Mapped[Optional[int]]  # Duration in seconds
    bitrate: Mapped[Optional[int]]  # Overall bitrate
    container_format: Mapped[Optional[str]]  # Container format (mp4, mkv, etc.)
    codec_data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    file: Mapped[File] = relationship(back_populates="technical_info")
    video_tracks: Mapped[list["VideoTrack"]] = relationship(
        back_populates="technical_info", cascade="all, delete-orphan"
    )
    audio_tracks: Mapped[list["AudioTrack"]] = relationship(
        back_populates="technical_info", cascade="all, delete-orphan"
    )


class VideoTrack(Base):
    __tablename__ = "video_tracks"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    technical_info_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("media_technical_info.id", ondelete="CASCADE"), type_=GUID
    )
    track_index: Mapped[int]  # Track number in the container
    width: Mapped[Optional[int]]  # Video width in pixels
    height: Mapped[Optional[int]]  # Video height in pixels
    codec: Mapped[Optional[str]]  # Video codec (h264, h265, etc.)
    frame_rate: Mapped[Optional[float]]  # Frames per second
    bit_depth: Mapped[Optional[int]]  # Color bit depth (8, 10, 12)
    color_space: Mapped[Optional[str]]  # Color space (BT.709, BT.2020, etc.)
    hdr_format: Mapped[Optional[str]]  # HDR format (HDR10, Dolby Vision, etc.)
    bitrate: Mapped[Optional[int]]  # Video bitrate
    is_default: Mapped[bool] = mapped_column(default=False)
    metadata_info: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    technical_info: Mapped[MediaTechnicalInfo] = relationship(
        back_populates="video_tracks"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "technical_info_id", "track_index", name="uq_video_track_per_media"
        ),
    )


class AudioTrack(Base):
    __tablename__ = "audio_tracks"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    technical_info_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("media_technical_info.id", ondelete="CASCADE"), type_=GUID
    )
    track_index: Mapped[int]  # Track number in the container
    codec: Mapped[Optional[str]]  # Audio codec (aac, ac3, etc.)
    language: Mapped[Optional[str]]  # ISO language code
    channels: Mapped[Optional[int]]  # Number of audio channels
    sample_rate: Mapped[Optional[int]]  # Sample rate in Hz
    bitrate: Mapped[Optional[int]]  # Audio bitrate
    title: Mapped[Optional[str]]  # Track title (e.g., "Director's Commentary")
    is_default: Mapped[bool] = mapped_column(default=False)
    metadata_info: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    technical_info: Mapped[MediaTechnicalInfo] = relationship(
        back_populates="audio_tracks"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "technical_info_id", "track_index", name="uq_audio_track_per_media"
        ),
    )


class TranscodingSession(Base):
    __tablename__ = "transcoding_sessions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("files.id"), type_=GUID)
    user_id: Mapped[uuid.UUID]  # Reference to user table (not shown)
    current_timestamp: Mapped[Optional[int]]  # Current playback position in seconds
    status: Mapped[TranscodingSessionStatus] = enum_column(
        TranscodingSessionStatus, default=TranscodingSessionStatus.ACTIVE
    )
    output_path: Mapped[Optional[str]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    file: Mapped[File] = relationship(back_populates="transcoding_sessions")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    job_type: Mapped[JobType] = enum_column(JobType)
    status: Mapped[JobStatus] = enum_column(JobStatus, default=JobStatus.OPEN)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON)
    priority: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    started_at: Mapped[Optional[datetime]]
    completed_at: Mapped[Optional[datetime]]
    error: Mapped[Optional[str]]  # Error message if failed
    retry_count: Mapped[int] = mapped_column(default=0)
    parent_job_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("jobs.id"), type_=GUID
    )

    # Relationships
    child_jobs: Mapped[list[Job]] = relationship(
        "Job", backref="parent_job", remote_side=[id]
    )

    # Indexes
    __table_args__ = (
        Index(
            "idx_jobs_status_priority_created",
            "status",
            "priority",
            "created_at",
            postgresql_ops={"priority": "DESC"},
        ),
        Index("idx_jobs_type_status", "job_type", "status"),
    )


class Service(Base):
    __tablename__ = "services"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    service_type: Mapped[ServiceType] = enum_column(ServiceType)
    status: Mapped[ServiceStatus] = enum_column(
        ServiceStatus, default=ServiceStatus.INACTIVE
    )
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON)
    command: Mapped[ServiceCommand] = enum_column(
        ServiceCommand, default=ServiceCommand.NONE
    )
    command_issued_at: Mapped[Optional[datetime]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    started_at: Mapped[Optional[datetime]]
    last_heartbeat_at: Mapped[Optional[datetime]]
    error: Mapped[Optional[str]]
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    # Indexes
    __table_args__ = (
        Index("idx_services_type_status", "service_type", "status"),
        Index("idx_services_command", "command", postgresql_where="command != 'NONE'"),
        Index(
            "idx_services_heartbeat",
            "last_heartbeat_at",
            postgresql_where="status = 'ACTIVE'",
        ),
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    playback_sessions: Mapped[list["PlaybackSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class ClientProfile(Base):
    __tablename__ = "client_profiles"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    client_hash: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    device_type: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_used_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    codec_supports: Mapped[list["ProfileCodecSupport"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )
    playback_sessions: Mapped[list["PlaybackSession"]] = relationship(
        back_populates="profile"
    )


class CodecFormat(Base):
    __tablename__ = "codec_formats"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    video_codec: Mapped[str] = mapped_column(String, nullable=False)
    video_profile: Mapped[Optional[str]] = mapped_column(String)
    bit_depth: Mapped[Optional[int]]
    color_space: Mapped[Optional[str]] = mapped_column(String)
    max_resolution: Mapped[Optional[str]] = mapped_column(String)
    audio_codec: Mapped[str] = mapped_column(String, nullable=False)
    container_format: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    profile_supports: Mapped[list["ProfileCodecSupport"]] = relationship(
        back_populates="codec_format", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "video_codec", 
            "video_profile", 
            "bit_depth", 
            "audio_codec", 
            "container_format",
            name="uq_codec_format"
        ),
    )


class ProfileCodecSupport(Base):
    __tablename__ = "profile_codec_support"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("client_profiles.id", ondelete="CASCADE"), type_=GUID
    )
    codec_format_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("codec_formats.id", ondelete="CASCADE"), type_=GUID
    )
    is_supported: Mapped[bool] = mapped_column(nullable=False)
    confidence: Mapped[int] = mapped_column(default=100)
    notes: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    profile: Mapped[ClientProfile] = relationship(back_populates="codec_supports")
    codec_format: Mapped[CodecFormat] = relationship(back_populates="profile_supports")

    # Constraints
    __table_args__ = (
        UniqueConstraint("profile_id", "codec_format_id", name="uq_profile_codec"),
    )


class PlaybackSession(Base):
    __tablename__ = "playback_sessions"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), type_=GUID
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), type_=GUID
    )
    profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("client_profiles.id", ondelete="CASCADE"), type_=GUID
    )
    transcoding_session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("transcoding_sessions.id", ondelete="SET NULL"), type_=GUID
    )
    current_position: Mapped[int] = mapped_column(default=0)
    duration: Mapped[Optional[int]]
    watched_percentage: Mapped[float] = mapped_column(default=0.0)
    is_completed: Mapped[bool] = mapped_column(default=False)
    selected_audio_track: Mapped[Optional[int]]
    selected_subtitle_track: Mapped[Optional[int]]
    started_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    last_updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]]

    # Relationships
    user: Mapped[User] = relationship(back_populates="playback_sessions")
    file: Mapped[File] = relationship()
    profile: Mapped[ClientProfile] = relationship(back_populates="playback_sessions")
    transcoding_session: Mapped[Optional[TranscodingSession]] = relationship()
