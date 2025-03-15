from typing import Optional, TypeVar, Generic, Any
from abc import ABC, abstractmethod
from src.common.db import AsyncDatabaseSession
from src.common.dto import ServiceParams
from src.common.logger import Logger


T_ServiceParams = TypeVar("T_ServiceParams")


class Service(Generic[T_ServiceParams], ABC):
    """Base class for all services."""

    def __init__(self, db_session: Any, logger: Optional[Any] = None) -> None:
        """
        Initialize the service.

        Args:
            db_session: Database session
            logger: Optional logger
        """
        self.db_session = db_session
        self.logger = logger

    @abstractmethod
    async def start(self, parameters: Optional[T_ServiceParams] = None) -> None:
        """
        Start the service.

        Args:
            parameters: Service parameters
        """
        pass

    @abstractmethod
    async def process_iteration(
        self, parameters: Optional[T_ServiceParams] = None
    ) -> None:
        """
        Process a single iteration of the service.

        Args:
            parameters: Service parameters
        """
        pass

    async def stop(self) -> None:
        """Stop the service. Override this method if cleanup is needed."""
        pass
