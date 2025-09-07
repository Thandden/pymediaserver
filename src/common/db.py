from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)

from src.common.models import Base


class AsyncDatabaseSession:
    """Async database session manager for SQLAlchemy.

    Provides async context manager for database sessions with automatic
    commit/rollback handling.
    """

    _engine: AsyncEngine
    _session_factory: async_sessionmaker[AsyncSession]

    def __init__(self, db_url: str) -> None:
        """Initialize database session manager.

        Args:
            db_url: Database connection URL
        """
        self._engine = create_async_engine(
            db_url,
            echo=False,  # Set to True for SQL query logging
            future=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    async def create_all(self) -> None:
        """Create all database tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        """Drop all database tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session.

        Yields:
            AsyncSession: Database session

        Example:
            async with AsyncDatabaseSession("sqlite+aiosqlite:///db.sqlite3").get_session() as session:
                result = await session.execute(select(User))
        """
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        """Close database engine."""
        await self._engine.dispose()
