import asyncio
import signal
import uuid
from typing import Optional, cast

from sqlalchemy import select

from src.job_dispatcher import JobDispatcher
from src.service_dispatcher import ServiceDispatcher
from src.common.logger import Logger, LogLevel
from src.common.db import AsyncDatabaseSession
from src.common.models import Service as ServiceModel
from src.common.system_types import ServiceType, ServiceStatus, ServiceCommand
from src.common.dto import CleanupParams
from src.common.config import config


async def create_default_services(
    db_session: AsyncDatabaseSession, logger: Logger
) -> None:
    """
    Create default services if they don't exist.
    Args:
        db_session: Database session
        logger: Logger instance
    """
    logger.info("Checking for default services")

    # Define default services
    default_services = [
        {
            "service_type": ServiceType.CLEANUP,
            "parameters": CleanupParams(
                cleanup_interval=300
            ).model_dump(),  # Run every 5 minutes for testing
            "status": ServiceStatus.INACTIVE,
            "command": ServiceCommand.START,
        },
    ]

    async for session in db_session.get_session():
        for service_def in default_services:
            # Check if service of this type already exists
            service_type = cast(ServiceType, service_def["service_type"])
            stmt = select(ServiceModel).where(ServiceModel.service_type == service_type)
            result = await session.execute(stmt)
            existing_service = result.scalar_one_or_none()

            if not existing_service:
                # Create new service
                new_service = ServiceModel(
                    id=uuid.uuid4(),
                    service_type=service_type,
                    parameters=service_def["parameters"],
                    status=service_def["status"],
                    command=service_def["command"],
                )
                session.add(new_service)
                logger.info(f"Created default service: {service_type.name}")

        await session.commit()


async def start_default_services(
    db_session: AsyncDatabaseSession, logger: Logger
) -> None:
    """
    Set default services to start.

    Args:
        db_session: Database session
        logger: Logger instance
    """
    logger.info("Starting default services")

    async for session in db_session.get_session():
        # Find all inactive services
        stmt = select(ServiceModel).where(ServiceModel.status == ServiceStatus.INACTIVE)
        result = await session.execute(stmt)
        inactive_services = result.scalars().all()

        for service in inactive_services:
            # Set command to START
            service.command = ServiceCommand.START
            logger.info(
                f"Setting service {service.id} ({service.service_type.name}) to start"
            )

        await session.commit()


async def run_job_dispatcher(db_session: AsyncDatabaseSession, logger: Logger) -> None:
    """
    Run the job dispatcher.

    Args:
        db_session: Database session
        logger: Logger instance
    """
    # Get poll interval from config
    poll_interval = config.JOB_POLL_INTERVAL

    # Create job dispatcher
    dispatcher = JobDispatcher(
        db_session=db_session,
        poll_interval=poll_interval,
        max_concurrent_jobs=5,
        logger=logger,
    )

    try:
        # Start the job dispatcher
        logger.info("Starting job dispatcher")
        await dispatcher.start()
    except asyncio.CancelledError:
        logger.info("Job dispatcher received cancellation")
        await dispatcher.stop()
        raise
    except Exception as e:
        logger.error(f"Error in job dispatcher: {str(e)}")
        await dispatcher.stop()
        raise


async def run_service_dispatcher(
    db_session: AsyncDatabaseSession, logger: Logger
) -> None:
    """
    Run the service dispatcher.

    Args:
        db_session: Database session
        logger: Logger instance
    """
    # Get heartbeat interval from config
    heartbeat_interval = config.SERVICE_HEARTBEAT_INTERVAL

    # Create service dispatcher
    dispatcher = ServiceDispatcher(
        db_session=db_session,
        heartbeat_interval=heartbeat_interval,
        max_concurrent_services=5,
        logger=logger,
    )

    try:
        # Start the service dispatcher
        logger.info("Starting service dispatcher")
        await dispatcher.start()
    except asyncio.CancelledError:
        logger.info("Service dispatcher received cancellation")
        await dispatcher.stop()
        raise
    except Exception as e:
        logger.error(f"Error in service dispatcher: {str(e)}")
        await dispatcher.stop()
        raise


async def main() -> None:
    """Main entry point for the application."""
    # Set up logger with log level from config
    log_level_str = config.LOG_LEVEL
    log_level = getattr(LogLevel, log_level_str, LogLevel.INFO)
    logger = Logger(name="main", level=log_level)
    logger.info("Starting application")

    # Initialize database connection with async driver
    # Use the same database for both jobs and services
    db_url = "sqlite+aiosqlite:///db.sqlite"
    db_session = AsyncDatabaseSession(db_url=db_url)

    # Create database tables if they don't exist
    logger.info("Creating database tables if they don't exist")
    await db_session.create_all()

    # Create default services if they don't exist
    await create_default_services(db_session, logger)

    # Start default services
    await start_default_services(db_session, logger)

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    # Create tasks for job and service dispatchers
    job_task = asyncio.create_task(run_job_dispatcher(db_session, logger))
    service_task = asyncio.create_task(run_service_dispatcher(db_session, logger))

    # Define signal handler
    async def shutdown(signal: Optional[signal.Signals] = None) -> None:
        """Handle shutdown gracefully."""
        if signal:
            logger.info(f"Received exit signal {signal.name}")

        logger.info("Cancelling dispatcher tasks")
        job_task.cancel()
        service_task.cancel()

        try:
            # Wait for tasks to complete with timeout
            await asyncio.wait([job_task, service_task], timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not complete in time")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

        logger.info("Closing database connection")
        await db_session.close()

        logger.info("Shutdown complete")

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    try:
        # Wait for both tasks to complete (they should run indefinitely)
        await asyncio.gather(job_task, service_task)
    except asyncio.CancelledError:
        logger.info("Main task received cancellation")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
