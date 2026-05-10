# Spotify DJ

FastAPI service that turns a music prompt into a Spotify recommendation and playback action. It uses OpenAI for planning, optionally uses Brave Search for fresh context, and calls the Spotify MCP server for Spotify search, item info, queue, and playback tools.

## What It Does

- Accepts a natural-language topic such as "play new UK garage for late night coding".
- Discovers available tools from the configured Spotify MCP server.
- Uses an OpenAI chat model to choose a track, album, artist, or playlist.
- Optionally uses Brave Search when the prompt needs fresh or time-sensitive music context.
- Resolves Spotify URIs through MCP tools before playback.
- Starts playback by default unless the request explicitly asks to queue music.

## Project Layout

- `src/spotify_dj/app.py`: FastAPI application and `/recommend` endpoint.
- `src/spotify_dj/config.py`: dotenv loading, logging, and DJ system prompt.
- `src/spotify_dj/models.py`: dataclasses shared across services.
- `src/spotify_dj/services/mcp_client.py`: MCP streamable HTTP client helpers.
- `src/spotify_dj/services/planner.py`: OpenAI planning and playback selection.
- `src/spotify_dj/services/web_search.py`: optional Brave Search integration.

## Requirements

- Python 3.13 or newer, as declared in `pyproject.toml`.
- A running Spotify MCP server.
- `OPENAI_API_KEY`.
- `BRAVE_API_KEY` if you want fresh web-search context.

## Environment

Create a `.env` file in this project root:

```env
OPENAI_API_KEY=your_openai_api_key
SPOTIFY_DJ_MODEL=gpt-5-nano
MCP_SERVER_URL=http://127.0.0.1:8080
BRAVE_API_KEY=your_brave_api_key
SPOTIFY_DJ_LOG_LEVEL=INFO
```

`BRAVE_API_KEY` is optional. If it is missing, Spotify DJ still runs and skips web search.

`SPOTIFY_DJ_MODEL` defaults to `gpt-5-nano`.

`SPOTIFY_DJ_LOG_FORMAT` can be set to override the default log format.

## Install

Using `uv`:

```bash
uv sync
```

Or with pip in a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

Start the Spotify MCP server first and set `MCP_SERVER_URL` to match its port. The MCP server's `python -m src.spotify_mcp_server.server` command starts on port 8080; Spotify DJ's code fallback is 9082 if `MCP_SERVER_URL` is unset.

Then from this project root:

```bash
uvicorn src.spotify_dj.app:app --host 0.0.0.0 --port 9090
```

If you use `uv`:

```bash
uv run uvicorn src.spotify_dj.app:app --host 0.0.0.0 --port 9090
```

The app loads `.env` from this project root during startup.

## API

### `POST /recommend`

Request:

```json
{
  "topic": "play something energetic for a rainy London evening"
}
```

Response includes the selected plan, Spotify item details where available, playback status, and a user-facing recommendation message.

## Troubleshooting

- `OPENAI_API_KEY not set; skipping LLM initialization`: check that `.env` is in the `spotify_app` project root and contains `OPENAI_API_KEY`.
- `LLM not initialized`: the OpenAI client did not initialize; check the key and installed dependencies.
- MCP discovery failures: check `MCP_SERVER_URL` and confirm it matches the port where the Spotify MCP server is running.
- Playback does not start: Spotify playback control requires an active Spotify device and a Spotify Premium account.
