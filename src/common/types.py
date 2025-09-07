from enum import Enum


class MediaType(Enum):
    MOVIE = "MOVIE"
    TV = "TV"
    MUSIC = "MUSIC"


class EntityType(Enum):
    MOVIE = "MOVIE"
    TV_SHOW = "TV_SHOW"
    TV_EPISODE = "TV_EPISODE"


class JobType(Enum):
    MEDIA_SCAN = "MEDIA_SCAN"
    WATCH_DOG = "WATCH_DOG"
    FILE_MATCHER = "FILE_MATCHER"
    METADATA_MATCHER = "METADATA_MATCHER"
    TV_METADATA_MATCHER = "TV_METADATA_MATCHER"
    FFPROBE = "FFPROBE"
    TRANSCODER = "TRANSCODER"
    CLEAN_UP = "CLEAN_UP"


class FileStatus(Enum):
    INDEXED = "INDEXED"
    UNINDEXED = "UNINDEXED"


class MetadataStatus(Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
