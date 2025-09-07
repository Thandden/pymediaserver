"""Service dispatcher for managing long-running services."""

import asyncio
import signal
from typing import Optional, Dict, Set, Any
from uuid import UUID

from sqlalchemy import select, or_
from datetime import datetime, timezone, timedelta

from src.common.db import AsyncDatabaseSession
from src.common.dto import WatchDogParams
from src.common.logger import Logger, LogLevel
from src.common.models import Service as ServiceModel
from src.common.system_types import ServiceStatus, ServiceCommand, ServiceType
from src.service_manager import service_manager, ServiceContext


class ServiceDispatcher:
    """Dispatcher that manages and monitors long-running services."""

    # Map service types to their parameter classes
    PARAMS_MAP = {
        ServiceType.WATCH_DOG: WatchDogParams,
        # Add other service types and their parameter classes here
    }

    def __init__(
        self,
        db_session: AsyncDatabaseSession,
        heartbeat_interval: int = 30,
        max_concurrent_services: int = 10,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the service dispatcher.

        Args:
            db_session: Database session for database operations
            heartbeat_interval: Interval in seconds between service heartbeat checks
            max_concurrent_services: Maximum number of services to run concurrently
            logger: Optional logger instance
        """
        self.db_session = db_session
        self.heartbeat_interval = heartbeat_interval
        self.max_concurrent_services = max_concurrent_services
        self.logger = logger or Logger("service_dispatcher", LogLevel.INFO)
        self.running = False
        self.active_services: Dict[UUID, asyncio.Task[Any]] = {}
        self.service_heartbeats: Dict[UUID, datetime] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_services)

    async def start(self) -> None:
        """Start the service dispatcher loop."""
        self.running = True
        self.logger.info(
            f"Starting service dispatcher with heartbeat interval {self.heartbeat_interval}s"
        )

        # Start the command monitoring task
        command_task = asyncio.create_task(self._monitor_commands())

        # Start the heartbeat monitoring task
        heartbeat_task = asyncio.create_task(self._monitor_heartbeats())

        try:
            # Run until stopped
            while self.running:
                try:
                    # Check for services that need to be started
                    await self._check_services_to_start()

                    # Check for services that need to be stopped
                    await self._check_services_to_stop()

                    # Wait for the next cycle
                    await asyncio.sleep(self.heartbeat_interval)

                except asyncio.CancelledError:
                    self.logger.info("Service dispatcher received cancellation")
                    self.running = False
                    break
                except Exception as e:
                    self.logger.error(f"Error in service dispatcher loop: {str(e)}")
                    await asyncio.sleep(self.heartbeat_interval)
        finally:
            # Cancel the monitoring tasks
            command_task.cancel()
            heartbeat_task.cancel()

            # Stop all running services
            await self._stop_all_services()

    async def stop(self) -> None:
        """Stop the service dispatcher."""
        self.logger.info("Stopping service dispatcher")
        self.running = False
        await self._stop_all_services()

    async def _check_services_to_start(self) -> None:
        """Check for services that need to be started."""
        async for session in self.db_session.get_session():
            # Calculate how many more services we can run
            available_slots = self.max_concurrent_services - len(self.active_services)

            if available_slots <= 0:
                return

            # Query for services with START command
            stmt = (
                select(ServiceModel)
                .where(
                    ServiceModel.command == ServiceCommand.START,
                    ServiceModel.status.in_(
                        [ServiceStatus.INACTIVE, ServiceStatus.FAILED]
                    ),
                )
                .limit(available_slots)
            )

            self.logger.debug(f"Checking for services to start with query: {str(stmt)}")
            result = await session.execute(stmt)
            services_to_start = list(result.scalars().all())

            self.logger.debug(f"Found {len(services_to_start)} services to start")

            for service in services_to_start:
                self.logger.debug(
                    f"Service to start: {service.id} ({service.service_type.name}), status: {service.status}, command: {service.command}"
                )
                if service.id in self.active_services:
                    self.logger.warning(
                        f"Service {service.id} is already active but was returned in query"
                    )
                    continue

                # Start the service
                self.logger.info(
                    f"Starting service {service.id} of type {service.service_type}"
                )
                service_task = asyncio.create_task(self._run_service(service.id))
                self.active_services[service.id] = service_task

                # Reset the command
                service.command = ServiceCommand.NONE
                service.command_issued_at = None
                await session.commit()

    async def _check_services_to_stop(self) -> None:
        """Check for services that need to be stopped."""
        async for session in self.db_session.get_session():
            # Query for services with STOP command
            stmt = select(ServiceModel).where(
                ServiceModel.command == ServiceCommand.STOP,
                ServiceModel.status == ServiceStatus.ACTIVE,
            )

            result = await session.execute(stmt)
            services_to_stop = list(result.scalars().all())

            for service in services_to_stop:
                if service.id not in self.active_services:
                    self.logger.warning(
                        f"Service {service.id} is not active but has STOP command"
                    )
                    # Reset the command anyway
                    service.command = ServiceCommand.NONE
                    service.command_issued_at = None
                    service.status = ServiceStatus.INACTIVE
                    await session.commit()
                    continue

                # Stop the service
                self.logger.info(f"Stopping service {service.id}")
                service_task = self.active_services[service.id]
                service_task.cancel()

                # Reset the command
                service.command = ServiceCommand.NONE
                service.command_issued_at = None
                service.status = ServiceStatus.SHUTTING_DOWN
                await session.commit()

    async def _monitor_commands(self) -> None:
        """Monitor for service commands like restart."""
        while self.running:
            try:
                async for session in self.db_session.get_session():
                    # Query for services with RESTART command
                    stmt = select(ServiceModel).where(
                        ServiceModel.command == ServiceCommand.RESTART
                    )

                    result = await session.execute(stmt)
                    services_to_restart = list(result.scalars().all())

                    for service in services_to_restart:
                        self.logger.info(f"Restarting service {service.id}")

                        # If the service is active, stop it first
                        if service.id in self.active_services:
                            service_task = self.active_services[service.id]
                            service_task.cancel()
                            # Wait for the task to be cancelled
                            try:
                                await asyncio.wait_for(service_task, timeout=5.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                self.logger.warning(
                                    f"Service {service.id} did not stop gracefully"
                                )

                            # Remove from active services
                            self.active_services.pop(service.id, None)

                        # Update service status to inactive
                        service.status = ServiceStatus.INACTIVE

                        # Start the service again
                        service.command = ServiceCommand.START
                        service.command_issued_at = datetime.now(timezone.utc)
                        await session.commit()

                # Sleep for a short time before checking again
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in command monitor: {str(e)}")
                await asyncio.sleep(5)

    async def _monitor_heartbeats(self) -> None:
        """Monitor service heartbeats and detect stalled services."""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                stalled_services: Set[UUID] = set()

                async for session in self.db_session.get_session():
                    # Query for active services with old heartbeats
                    heartbeat_threshold = now - timedelta(
                        seconds=self.heartbeat_interval * 3
                    )
                    stmt = select(ServiceModel).where(
                        ServiceModel.status == ServiceStatus.ACTIVE,
                        or_(
                            ServiceModel.last_heartbeat_at < heartbeat_threshold,
                            ServiceModel.last_heartbeat_at.is_(None),
                        ),
                    )

                    result = await session.execute(stmt)
                    stalled_services_models = list(result.scalars().all())

                    for service in stalled_services_models:
                        self.logger.warning(
                            f"Service {service.id} has stalled (last heartbeat: {service.last_heartbeat_at})"
                        )
                        stalled_services.add(service.id)

                        # Mark the service as failed
                        service.status = ServiceStatus.FAILED
                        service.error = "Service heartbeat timeout"
                        await session.commit()

                # Cancel tasks for stalled services
                for service_id in stalled_services:
                    if service_id in self.active_services:
                        self.logger.info(
                            f"Cancelling stalled service task {service_id}"
                        )
                        self.active_services[service_id].cancel()
                        self.active_services.pop(service_id, None)

                # Sleep for a short time before checking again
                await asyncio.sleep(self.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {str(e)}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _run_service(self, service_id: UUID) -> None:
        """Run a service in a separate task.

        Args:
            service_id: ID of the service to run
        """
        service_logger = self.logger.bind(service_id=str(service_id))

        try:
            async with self.semaphore:
                service_logger.info(f"Starting service {service_id}")

                try:
                    async with service_manager(
                        self.db_session, service_id, service_logger
                    ) as context:
                        # Start a heartbeat task
                        heartbeat_task = asyncio.create_task(
                            self._service_heartbeat(service_id, context)
                        )

                        try:
                            # Execute the service (this will run until cancelled or completed)
                            await context.execute_service()
                            service_logger.info(
                                f"Service {service_id} completed successfully"
                            )
                        finally:
                            # Cancel the heartbeat task
                            heartbeat_task.cancel()
                            try:
                                await heartbeat_task
                            except asyncio.CancelledError:
                                pass

                except asyncio.CancelledError:
                    service_logger.info(f"Service {service_id} was cancelled")
                    raise
                except Exception as e:
                    service_logger.error(f"Error in service context: {str(e)}")
                    raise
        except Exception as e:
            service_logger.error(f"Error running service {service_id}: {str(e)}")
        finally:
            # Remove from active services
            self.active_services.pop(service_id, None)
            service_logger.info(f"Service {service_id} task ended")

    async def _service_heartbeat(
        self, service_id: UUID, context: ServiceContext
    ) -> None:
        """Send periodic heartbeats for a service.

        Args:
            service_id: ID of the service
            context: Service context for updating heartbeat
        """
        while True:
            try:
                await context.update_heartbeat()
                self.service_heartbeats[service_id] = datetime.now(timezone.utc)
                await asyncio.sleep(
                    self.heartbeat_interval / 2
                )  # Update at half the check interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error updating heartbeat for service {service_id}: {str(e)}"
                )
                await asyncio.sleep(5)  # Retry after a short delay

    async def _stop_all_services(self) -> None:
        """Stop all running services."""
        if not self.active_services:
            return

        self.logger.info(f"Stopping {len(self.active_services)} active services")

        # Cancel all service tasks
        for service_id, task in list(self.active_services.items()):
            self.logger.info(f"Cancelling service {service_id}")
            task.cancel()

        # Wait for all tasks to complete
        if self.active_services:
            pending_tasks = list(self.active_services.values())
            self.logger.info(f"Waiting for {len(pending_tasks)} services to stop")

            try:
                # Wait with a timeout
                _, pending = await asyncio.wait(
                    pending_tasks, timeout=10.0, return_when=asyncio.ALL_COMPLETED
                )

                if pending:
                    self.logger.warning(
                        f"{len(pending)} services did not stop gracefully"
                    )
            except Exception as e:
                self.logger.error(f"Error waiting for services to stop: {str(e)}")

        # Clear the active services
        self.active_services.clear()

        # Update the database to mark all active services as inactive
        async for session in self.db_session.get_session():
            stmt = select(ServiceModel).where(
                ServiceModel.status.in_(
                    [ServiceStatus.ACTIVE, ServiceStatus.SHUTTING_DOWN]
                )
            )

            result = await session.execute(stmt)
            active_services = list(result.scalars().all())

            for service in active_services:
                service.status = ServiceStatus.INACTIVE
                service.command = ServiceCommand.NONE

            await session.commit()


async def run_service_dispatcher(
    db_session: AsyncDatabaseSession,
    heartbeat_interval: int = 30,
    max_concurrent_services: int = 10,
    logger: Optional[Logger] = None,
) -> None:
    """Run the service dispatcher as a standalone process.

    Args:
        db_session: Database session for database operations
        heartbeat_interval: Interval in seconds between service heartbeat checks
        max_concurrent_services: Maximum number of services to run concurrently
        logger: Optional logger instance
    """
    dispatcher = ServiceDispatcher(
        db_session=db_session,
        heartbeat_interval=heartbeat_interval,
        max_concurrent_services=max_concurrent_services,
        logger=logger,
    )

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    async def shutdown(signal_: Optional[signal.Signals] = None) -> None:
        if signal_ and logger:
            logger.info(f"Received exit signal {signal_.name}")
        await dispatcher.stop()

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    try:
        await dispatcher.start()
    except asyncio.CancelledError:
        await dispatcher.stop()
