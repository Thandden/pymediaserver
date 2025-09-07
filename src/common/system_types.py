from enum import Enum


class MediaType(Enum):
    MOVIE = "MOVIE"
    TV = "TV"
    MUSIC = "MUSIC"


class EntityType(Enum):
    MOVIE = "MOVIE"
    TV_EPISODE = "TV_EPISODE"


class JobType(Enum):
    """Types of jobs that can be processed"""

    MEDIA_SCAN = "MEDIA_SCAN"
    WATCH_DOG = "WATCH_DOG"
    FILE_MATCHER = "FILE_MATCHER"
    METADATA_MATCHER = "METADATA_MATCHER"
    MOVIE_MATCHER = "MOVIE_MATCHER"
    TV_MATCHER = "TV_METADATA_MATCHER"
    FFPROBE = "FFPROBE"
    TRANSCODER = "TRANSCODER"
    CLEAN_UP = "CLEAN_UP"


# Add FileStatus enum to match the model
class FileStatus(Enum):
    INDEXED = "INDEXED"
    UNINDEXED = "UNINDEXED"


class MetadataStatus(Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class JobStatus(Enum):
    OPEN = "OPEN"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ServiceType(Enum):
    WATCH_DOG = "WATCH_DOG"
    METRICS_COLLECTOR = "METRICS_COLLECTOR"
    EVENT_RELAY = "EVENT_RELAY"


class ServiceStatus(Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    SHUTTING_DOWN = "SHUTTING_DOWN"


class ServiceCommand(Enum):
    NONE = "NONE"
    START = "START"
    STOP = "STOP"
    RESTART = "RESTART"


class TranscodingSessionStatus(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
