from abc import ABC, abstractmethod
from typing import Optional, TypeVar
from src.common.db import AsyncDatabaseSession
from src.common.dto import JobParams
from src.common.logger import Logger
from src.common.dto import ChildJobRequest, NoChildJob

T_JobParams = TypeVar("T_JobParams", bound=JobParams)


class Worker(ABC):
    """Base class for all job workers"""

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        self.db_session = db_session
        self.logger = logger

    @abstractmethod
    async def execute(
        self, parameters: Optional[T_JobParams] = None
    ) -> list[ChildJobRequest] | NoChildJob:
        """
        Execute the job logic

        Args:
            parameters: Job parameters (optional)

        Returns:
            JobDTO: Create a new child JobDTO.
        """
        pass
