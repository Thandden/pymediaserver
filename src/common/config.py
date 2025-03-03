"""Configuration module for loading and accessing environment variables."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic, cast, Type, get_type_hints

from dotenv import load_dotenv

T = TypeVar("T")


class ConfigValue(Generic[T]):
    """A descriptor for accessing typed environment variables."""

    def __init__(
        self,
        env_name: str,
        default: Optional[T] = None,
        value_type: Optional[Type[T]] = None,
        required: bool = False,
    ) -> None:
        """
        Initialize the config value.

        Args:
            env_name: Environment variable name
            default: Default value if env variable is not set
            value_type: Type to convert the value to
            required: Whether the value is required
        """
        self.env_name = env_name
        self.default = default
        self.value_type = value_type
        self.required = required

    def __get__(self, obj: Any, objtype: Any = None) -> T:
        """
        Get the environment variable value with proper type conversion.

        Args:
            obj: Instance of the owner class
            objtype: Type of the owner class

        Returns:
            The typed environment variable value

        Raises:
            ValueError: If the variable is required but not set
        """
        value = os.environ.get(self.env_name)
        
        if value is None:
            if self.required and self.default is None:
                raise ValueError(f"Required environment variable {self.env_name} is not set")
            return cast(T, self.default)
        
        if self.value_type is bool and not isinstance(self.default, bool):
            return cast(T, value.lower() in ("true", "1", "yes", "y", "t"))
        elif self.value_type is not None:
            try:
                return cast(T, self.value_type(value))
            except ValueError:
                raise ValueError(f"Cannot convert {self.env_name}={value} to {self.value_type.__name__}")
        
        return cast(T, value)


class Config:
    """Configuration class that automatically loads environment variables."""

    # App settings
    APP_NAME: str = ConfigValue("APP_NAME", "MediaApp")
    DEBUG: bool = ConfigValue("DEBUG", False, bool)
    ENV: str = ConfigValue("ENV", "development")
    
    # Database settings
    DATABASE_URL: str = ConfigValue("DATABASE_URL", "sqlite:///media.db")
    DB_POOL_SIZE: int = ConfigValue("DB_POOL_SIZE", 5, int)
    
    # API settings
    TMDB_API_KEY: str = ConfigValue("TMDB_API_KEY", required=True)
    TMDB_API_URL: str = ConfigValue("TMDB_API_URL", "https://api.themoviedb.org/3")
    
    # Service settings
    SERVICE_HEARTBEAT_INTERVAL: int = ConfigValue("SERVICE_HEARTBEAT_INTERVAL", 30, int)
    JOB_POLL_INTERVAL: int = ConfigValue("JOB_POLL_INTERVAL", 5, int)
    
    # File settings
    MEDIA_DIRECTORY: str = ConfigValue("MEDIA_DIRECTORY", str(Path.home() / "media"))
    
    # Logging
    LOG_LEVEL: str = ConfigValue("LOG_LEVEL", "INFO")
    LOG_TO_FILE: bool = ConfigValue("LOG_TO_FILE", False, bool)
    LOG_FILE_PATH: str = ConfigValue("LOG_FILE_PATH", "logs/app.log")

    def __init__(self, env_file: Optional[str] = None) -> None:
        """
        Initialize the configuration by loading environment variables.
        
        Args:
            env_file: Optional path to .env file
        """
        # Load from .env file if provided
        if env_file:
            load_dotenv(env_file)
        else:
            # Look for .env file in standard locations
            load_dotenv()  # Default looks in current directory and parent
        
    def as_dict(self) -> Dict[str, Any]:
        """
        Get all configuration values as a dictionary.
        
        Returns:
            Dictionary of configuration values
        """
        result: Dict[str, Any] = {}
        for key, _ in get_type_hints(self.__class__).items():
            if not key.startswith("_") and hasattr(self, key):
                result[key] = getattr(self, key)
        return result


# Create a singleton instance for easy importing
config = Config() 