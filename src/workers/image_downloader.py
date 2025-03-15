import os
import aiofiles
from pathlib import Path
from typing import Optional

from src.common.base import Worker, T_JobParams, NoChildJob
from src.common.db import AsyncDatabaseSession
from src.common.dto import ImageDownloaderParams
from src.common.http_client import AsyncHttpClient
from src.common.logger import Logger
from src.common.config import config


class ImageDownloader(Worker):
    """Worker for downloading images from TMDB."""

    TMDB_IMAGE_BASE_URL: str = "https://image.tmdb.org/t/p/original"

    def __init__(
        self, db_session: AsyncDatabaseSession, logger: Optional[Logger] = None
    ) -> None:
        """
        Initialize the image downloader worker.

        Args:
            db_session: Database session for database operations
            logger: Optional logger instance
        """
        super().__init__(db_session, logger)
        self.http_client = AsyncHttpClient(
            retries=3,
            delay=2,
        )

    async def execute(self, parameters: Optional[T_JobParams] = None) -> NoChildJob:
        """
        Execute the image download process.

        Args:
            parameters: Job parameters containing image URL and entity ID

        Returns:
            None: No child jobs are created

        Raises:
            ValueError: If parameters are not of the correct type or are invalid
        """
        if parameters is None:
            raise ValueError("Parameters cannot be None")

        if not isinstance(parameters, ImageDownloaderParams):
            raise ValueError("Parameters must be of type ImageDownloaderParams")

        image_url = parameters.image_url
        entity_id = parameters.entity_id

        if not image_url:
            if self.logger:
                self.logger.error("Image URL cannot be empty")
            return None

        # Construct the full TMDB image URL
        full_image_url: str
        if not image_url.startswith(("http://", "https://")):
            full_image_url = f"{self.TMDB_IMAGE_BASE_URL}/{image_url.lstrip('/')}"
        else:
            full_image_url = image_url

        # Construct the local path where the image will be saved
        entity_dir = Path(config.IMAGE_DIRECTORY) / str(entity_id)
        image_filename = os.path.basename(image_url)
        image_path = entity_dir / image_filename

        # Create the directory if it doesn't exist
        os.makedirs(entity_dir, exist_ok=True)

        if self.logger:
            self.logger.info(f"Downloading image from {full_image_url} to {image_path}")

        # Download and save the image
        success = await self._download_image(full_image_url, image_path)

        if success:
            if self.logger:
                self.logger.info(f"Successfully downloaded image to {image_path}")
        else:
            if self.logger:
                self.logger.error(f"Failed to download image from {full_image_url}")

        return None

    async def _download_image(self, url: str, path: Path) -> bool:
        """
        Download an image from a URL and save it to the specified path.

        Args:
            url: URL of the image to download
            path: Path where the image will be saved

        Returns:
            bool: True if download was successful, False otherwise
        """
        async with self.http_client:
            response = await self.http_client.fetch_data(url)

            if response is None:
                if self.logger:
                    self.logger.error(f"Failed to download image from {url}")
                return False

            try:
                # Read the image data
                image_data = await response.read()

                if not image_data:
                    if self.logger:
                        self.logger.error(f"Empty image data received from {url}")
                    return False

                # Save the image to the specified path
                async with aiofiles.open(path, "wb") as f:
                    await f.write(image_data)

                return True
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error saving image: {e}")
                return False
            finally:
                if not response.closed:
                    await response.release()
