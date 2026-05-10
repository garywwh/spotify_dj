"""Application configuration and logging utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_LOG_LEVEL = os.getenv("SPOTIFY_DJ_LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = os.getenv(
    "SPOTIFY_DJ_LOG_FORMAT",
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
SPOTIFY_DJ_MODEL: Final[str] = os.getenv("SPOTIFY_DJ_MODEL", "gpt-5-nano")

logger = logging.getLogger("spotify_dj")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)

try:
    logger.setLevel(_LOG_LEVEL)
except ValueError:
    logger.setLevel(logging.INFO)

DJ_SYSTEM_MESSAGE: Final[str] = (
    "You are a professional DJ who curates music recommendations and can control Spotify via MCP tools. "
    "When you need to operate Spotify, call the appropriate tool with JSON arguments that match the schema hints. "
    "Unless the user explicitly asks to queue music, immediately start playback of your primary recommendation. "
    "Use the queue tool only when the user asks to queue items. "
    "When starting playback, call SpotifyPlayback with at least action='start' and a valid 'spotify_uri' for the album/track you selected. "
    "Use SpotifySearch (or other tools) first to retrieve the correct Spotify URI before calling playback. "
    "Always finish with a short recommendation plus any playback actions you took."
)

__all__ = ["logger", "DJ_SYSTEM_MESSAGE", "SPOTIFY_DJ_MODEL"]
