# PyMediaServer

A Python-based media server that automatically processes, matches, and organizes media files with metadata enrichment and transcoding capabilities.

## Features

- **Automated Media Processing**: Scans directories for media files and matches them with metadata
- **Movie & TV Show Matching**: Automatically identifies and organizes movies and TV episodes
- **Transcoding**: Converts media files to different formats using FFmpeg
- **Image Management**: Downloads and manages artwork and thumbnails
- **File Monitoring**: Watches directories for new media files
- **Cleanup Services**: Automatically maintains file organization
- **Async Architecture**: Built with asyncio for high performance

## Requirements

- Python 3.10+
- FFmpeg (for transcoding)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pymediaserver
```

2. Install dependencies:
```bash
uv sync
```

## Usage

Run the media server:
```bash
uv run python main.py
```

The server will:
- Create a SQLite database (`db.sqlite`) 
- Start job and service dispatchers
- Begin monitoring for media files
- Process files automatically

## Configuration

Configuration is handled through environment variables. Create a `.env` file or set:

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `JOB_POLL_INTERVAL`: How often to check for new jobs
- `SERVICE_HEARTBEAT_INTERVAL`: Service health check interval

## Testing

Run tests with:
```bash
uv run pytest
```

## Project Structure

```
src/
├── common/          # Shared utilities and models
├── services/        # Background services (cleanup, watchdog)
├── workers/         # Media processing workers
└── tests/          # Test suite
```
