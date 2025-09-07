"""Service management and processing system."""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Type, TypeVar
from uuid import UUID
from datetime import datetime, timezone
import asyncio

from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.db import AsyncDatabaseSession
from src.common.dto import ServiceParams, WatchDogParams, CleanupParams
from src.common.logger import Logger
from src.common.models import Service as ServiceModel
from src.common.system_types import ServiceStatus, ServiceType
from src.common.base_service import Service
from src.services.cleanup import Cleanup
from src.services.watchdog import WatchDog

T_Params = TypeVar("T_Params", bound=ServiceParams)


class ServiceContext:
    """Async context manager for service processing lifecycle."""

    # Map service types to their service classes
    SERVICE_MAP: Dict[ServiceType, Type[Any]] = {
        ServiceType.WATCH_DOG: WatchDog,
        ServiceType.CLEANUP: Cleanup,
        # Add other service types and their implementations here
    }

    def __init__(
        self,
        db_session: AsyncDatabaseSession,
        service_id: UUID,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initialize the service context.

        Args:
            db_session: Database session for database operations
            service_id: ID of the service to process
            logger: Optional logger instance
        """
        self.db_session = db_session
        self.service_id = service_id
        self.logger = logger
        self.service_model: Optional[ServiceModel] = None
        self.service: Optional[Service[ServiceParams]] = None
        self.session: Optional[AsyncSession] = None
        self.service_type: Optional[ServiceType] = None
        self.params: Optional[ServiceParams] = None

    async def __aenter__(self) -> "ServiceContext":
        """Enter the service context and mark service as active."""
        async for session in self.db_session.get_session():
            self.session = session
            # Get and lock the service
            result = await session.execute(
                select(ServiceModel)
                .where(ServiceModel.id == self.service_id)
                .with_for_update()
            )
            self.service_model = result.scalar_one_or_none()

            if not self.service_model:
                raise ValueError(f"Service {self.service_id} not found")

            # Check if the service is in a valid state to start
            if self.service_model.status not in [
                ServiceStatus.INACTIVE,
                ServiceStatus.FAILED,
            ]:
                raise ValueError(
                    f"Service {self.service_id} is not in a startable state: {self.service_model.status}"
                )

            # Update service status to active
            self.service_model.status = ServiceStatus.ACTIVE
            self.service_model.started_at = datetime.now(timezone.utc)
            self.service_model.last_heartbeat_at = datetime.now(timezone.utc)
            await session.commit()

            # Create the appropriate service for this service type
            self._create_service(self.service_model.service_type)

            # Store service type and parameters as instance attributes
            self.service_type = self.service_model.service_type

            # Parse parameters based on service type
            if self.service_model.parameters:
                if self.service_type == ServiceType.WATCH_DOG:
                    params_adapter = TypeAdapter(WatchDogParams)
                    self.params = params_adapter.validate_python(
                        self.service_model.parameters
                    )
                elif self.service_type == ServiceType.CLEANUP:
                    params_adapter = TypeAdapter(CleanupParams)
                    self.params = params_adapter.validate_python(
                        self.service_model.parameters
                    )
                # Add other service types here as needed

            return self

        # This should never be reached if the database session is working correctly
        raise RuntimeError("Failed to acquire database session")

    def _create_service(self, service_type: ServiceType) -> None:
        """
        Create the appropriate service for the service type.

        Args:
            service_type: The type of service to create

        Raises:
            ValueError: If no service is registered for the service type
        """
        if service_type not in self.SERVICE_MAP:
            raise ValueError(f"No service registered for service type {service_type}")

        service_class = self.SERVICE_MAP[service_type]
        self.service = service_class(self.db_session, self.logger)

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit the service context and update final status."""
        if not self.service_model or not self.session:
            return

        async for session in self.db_session.get_session():
            # Get a fresh copy of the service
            result = await session.execute(
                select(ServiceModel).where(ServiceModel.id == self.service_id)
            )
            service_model = result.scalar_one_or_none()

            if not service_model:
                if self.logger:
                    self.logger.error(
                        f"Service {self.service_id} not found during exit"
                    )
                return

            if exc_val:
                # Handle error case
                service_model.status = ServiceStatus.FAILED
                service_model.error = str(exc_val)
                if self.logger:
                    self.logger.error(f"Service failed: {exc_val}")
            else:
                # Handle normal shutdown case
                service_model.status = ServiceStatus.INACTIVE
                if self.logger:
                    self.logger.info(f"Service shut down gracefully: {self.service_id}")

            await session.commit()

    async def execute_service(self) -> None:
        """
        Execute the service using the appropriate implementation.

        Raises:
            ValueError: If no service has been created for this context
        """
        if not self.service:
            raise ValueError("No service available for execution")

        if self.logger:
            self.logger.info(
                f"Executing service {self.service_id} with implementation {type(self.service).__name__}"
            )

        # Start the service
        await self.service.start(self.params)

        # Run the service in a loop until cancelled
        while True:
            try:
                # Process the next iteration
                await self.service.process_iteration(self.params)

                # For Cleanup service, we need to sleep between iterations
                if self.service_type == ServiceType.CLEANUP:
                    if self.params and hasattr(self.params, "cleanup_interval"):
                        sleep_interval = getattr(self.params, "cleanup_interval")
                        if self.logger:
                            self.logger.debug(
                                f"Sleeping for {sleep_interval} seconds until next cleanup cycle"
                            )
                        await asyncio.sleep(sleep_interval)
                    else:
                        # Default sleep for cleanup
                        await asyncio.sleep(3600)  # 1 hour default

                # WatchDog service handles its own timing in process_iteration

            except asyncio.CancelledError:
                # Handle service shutdown
                if hasattr(self.service, "stop") and callable(self.service.stop):
                    await self.service.stop()
                if self.logger:
                    self.logger.info(
                        f"Service {self.service_id} received cancellation signal"
                    )
                raise

    async def update_heartbeat(self) -> None:
        """Update the service heartbeat timestamp."""
        if not self.service_model or not self.session:
            return

        async for session in self.db_session.get_session():
            # Get a fresh copy of the service
            result = await session.execute(
                select(ServiceModel).where(ServiceModel.id == self.service_id)
            )
            service_model = result.scalar_one_or_none()

            if not service_model:
                if self.logger:
                    self.logger.error(
                        f"Service {self.service_id} not found during heartbeat update"
                    )
                return

            # Update the heartbeat timestamp
            service_model.last_heartbeat_at = datetime.now(timezone.utc)
            await session.commit()

            if self.logger:
                self.logger.debug(f"Updated heartbeat for service {self.service_id}")


@asynccontextmanager
async def service_manager(
    db_session: AsyncDatabaseSession,
    service_id: UUID,
    logger: Optional[Logger] = None,
) -> AsyncGenerator[ServiceContext, None]:
    """
    Create a service processing context.

    Args:
        db_session: Database session for database operations
        service_id: ID of the service to process
        logger: Optional logger instance

    Yields:
        ServiceContext: Context manager for service processing
    """
    context = ServiceContext(db_session, service_id, logger)
    try:
        await context.__aenter__()
        yield context
    except Exception as e:
        if logger:
            logger.error(f"Error in service context: {str(e)}")
        await context.__aexit__(type(e), e, e.__traceback__)
        raise
    else:
        await context.__aexit__(None, None, None)
