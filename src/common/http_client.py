import aiohttp
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional, AsyncIterator, TypeVar, Type

# T = TypeVar("T")


class AsyncHttpClient:
    """
    A class to handle asynchronous HTTP requests with retries and response management.
    """

    def __init__(self, retries: int = 3, delay: int = 2):
        """
        Initialize the HTTP client.
        :param retries: Number of retries in case of failure (default: 3)
        :param delay: Delay between retries in seconds (default: 2)
        """
        self.retries = retries
        self.delay = delay
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Initialize client session when entering context."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Cleanup client session when exiting context."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        """Ensure a session exists, creating one if necessary."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_data(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Optional[aiohttp.ClientResponse]:
        """
        Perform an asynchronous GET request with retry mechanism.
        :param endpoint: The API endpoint to request
        :param params: Optional parameters for the request
        :return: ClientResponse object or None if all retries fail
        """
        session = await self._ensure_session()

        for attempt in range(self.retries):
            response = None
            try:
                response = await session.get(endpoint, params=params)
                if response.status == 200:
                    return response
                print(f"Attempt {attempt + 1} failed with status {response.status}")
            except aiohttp.ClientError as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
            finally:
                # Only attempt to close if response is not None
                if response is not None and response.status != 200:
                    try:
                        await response.close()
                    except (TypeError, AttributeError):
                        pass

            if attempt < self.retries - 1:
                await asyncio.sleep(self.delay)

        return None

    @asynccontextmanager
    async def handle_response(
        self, response: Optional[aiohttp.ClientResponse]
    ) -> AsyncIterator[Optional[aiohttp.ClientResponse]]:
        """
        Context manager to handle response lifecycle.
        :param response: The ClientResponse object to manage
        :yield: The response object or None
        """
        if response is None:
            yield None
            return

        try:
            yield response
        finally:
            if not response.closed:
                try:
                    # Ensure close() method exists and is callable
                    close_method = getattr(response, "close", None)
                    if close_method is not None and callable(close_method):
                        await close_method()
                except (TypeError, AttributeError):
                    pass

    async def fetch_json(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Convenience method to fetch and parse JSON data.
        :param endpoint: The API endpoint to request
        :param params: Optional parameters for the request
        :return: Parsed JSON data or None if request fails
        """
        async with self.handle_response(
            await self.fetch_data(endpoint, params)
        ) as response:
            if response is not None:
                try:
                    return await response.json()
                except (TypeError, AttributeError):
                    return None
            return None
