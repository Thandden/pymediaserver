from pathlib import Path
from pydantic import BaseModel, Field, UUID4
from typing import Optional, Any, TypeAlias
import uuid
from datetime import datetime, date
from src.common.system_types import (
    MediaType,
    EntityType,
    JobType,
    JobStatus,
    FileStatus,
    TranscodingSessionStatus,
    ServiceStatus,
    ServiceCommand,
    MetadataStatus,
    ServiceType,
)


class MatchedData(BaseModel):
    title: str
    media_type: MediaType
    year: Optional[int] = None
    season_number: Optional[int] = None
    episode_number: Optional[int] = None


def validate_uuid(v: str) -> str:
    try:
        uuid.UUID(v)
    except ValueError:
        raise ValueError("Must be a valid UUID")
    return v


class JobParams(BaseModel):
    pass


class DTO(BaseModel):
    pass


class MediaScannerParams(JobParams):
    dir_path: str
    media_type: MediaType
    file_extensions: list[str]
    create_watchdog: bool = False


class WatchDogParams(JobParams):
    dir_path: Path
    media_type: MediaType
    file_extensions: list[str]


class FileMatcherParams(JobParams):
    path: str
    media_type: MediaType
    file_id: UUID4


class MetadataMatcherParams(JobParams):
    matched_data: MatchedData
    file_id: UUID4


class MovieMatcherParams(JobParams):
    tmdb_id: int
    file_id: UUID4


class TvMatcherParams(JobParams):
    tmdb_id: int
    file_id: UUID4
    season_number: int
    episode_number: int

class FFProbeParams(JobParams):
    file_id: UUID4
    path: str

class JobDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    params: Optional[JobParams] = None
    job_type: JobType
    status: JobStatus = JobStatus.OPEN
    priority: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    parent_job_id: Optional[UUID4] = None


class FileDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    path: str
    media_type: MediaType
    indexed: bool = False
    hash: Optional[str] = None
    status: FileStatus = FileStatus.UNINDEXED
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class EntityDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    file_id: UUID4
    entity_type: EntityType
    movie_id: Optional[UUID4] = None
    tv_episode_id: Optional[UUID4] = None
    matched_data: Optional[dict[str, Any]] = None
    metadata_status: MetadataStatus = MetadataStatus.PENDING
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TVShowDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    tmdb_id: Optional[int] = None
    title: str
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    year: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TVSeasonDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    show_id: UUID4
    season_number: int
    title: Optional[str] = None
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    year: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TVEpisodeDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    season_id: UUID4
    episode_number: int
    title: Optional[str] = None
    overview: Optional[str] = None
    still_path: Optional[str] = None
    air_date: Optional[date] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MovieDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    tmdb_id: Optional[int] = None
    title: str
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    release_date: Optional[date] = None
    year: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class VideoTrackDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    technical_info_id: UUID4
    track_index: int
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    frame_rate: Optional[float] = None
    bit_depth: Optional[int] = None
    color_space: Optional[str] = None
    hdr_format: Optional[str] = None
    bitrate: Optional[int] = None
    is_default: bool = False
    metadata: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AudioTrackDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    technical_info_id: UUID4
    track_index: int
    codec: Optional[str] = None
    language: Optional[str] = None
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    bitrate: Optional[int] = None
    title: Optional[str] = None
    is_default: bool = False
    metadata: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MediaTechnicalInfoDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    file_id: UUID4
    duration: Optional[int] = None
    bitrate: Optional[int] = None
    container_format: Optional[str] = None
    codec_data: Optional[dict[str, Any]] = None
    video_tracks: list[VideoTrackDTO] = Field(default_factory=list)
    audio_tracks: list[AudioTrackDTO] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TranscodingSessionDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    file_id: UUID4
    user_id: UUID4
    current_timestamp: Optional[int] = None
    status: TranscodingSessionStatus = TranscodingSessionStatus.ACTIVE
    output_path: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ServiceDTO(DTO):
    id: Optional[UUID4] = Field(default_factory=lambda: uuid.uuid4())
    service_type: ServiceType
    status: ServiceStatus = ServiceStatus.INACTIVE
    parameters: dict[str, Any]
    command: ServiceCommand = ServiceCommand.NONE
    command_issued_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = None
    error: Optional[str] = None
    config: Optional[dict[str, Any]] = None


ChildJobRequest: TypeAlias = JobDTO
NoChildJob: TypeAlias = None
