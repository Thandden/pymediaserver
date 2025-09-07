#!/usr/bin/env python3
"""Script to run the service dispatcher."""

import asyncio
import argparse
import sys

from src.common.db import AsyncDatabaseSession
from src.common.logger import Logger, LogLevel
from src.service_dispatcher import run_service_dispatcher
from src.common.config import config


async def main(
    heartbeat_interval: int = 30,
    max_concurrent_services: int = 10,
    log_level: str = "INFO",
) -> int:
    """
    Run the service dispatcher.

    Args:
        heartbeat_interval: Interval in seconds between service heartbeat checks
        max_concurrent_services: Maximum number of services to run concurrently
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Set up logger
    log_level_enum = getattr(LogLevel, log_level.upper(), LogLevel.INFO)
    logger = Logger("service_runner", log_level_enum)
    logger.info("Starting service runner")

    # Set up database session
    db_session = AsyncDatabaseSession(db_url=config.DATABASE_URL)

    try:
        # Run the service dispatcher
        await run_service_dispatcher(
            db_session=db_session,
            heartbeat_interval=heartbeat_interval,
            max_concurrent_services=max_concurrent_services,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.info("Service runner interrupted by user")
    except Exception as e:
        logger.error(f"Error in service runner: {str(e)}")
        return 1
    finally:
        logger.info("Service runner shutting down")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the service dispatcher")
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Interval in seconds between service heartbeat checks",
    )
    parser.add_argument(
        "--max-concurrent-services",
        type=int,
        default=10,
        help="Maximum number of services to run concurrently",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    result = asyncio.run(
        main(
            heartbeat_interval=args.heartbeat_interval,
            max_concurrent_services=args.max_concurrent_services,
            log_level=args.log_level,
        )
    )

    sys.exit(result)
