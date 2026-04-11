"""FastAPI application entrypoint for Spotify DJ."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import logger
from .models import PlaybackSelection, SearchSnippet, SelectionPlan
from .services.mcp_client import (
    call_mcp_tool,
    discover_tools,
    ensure_streamable_http_url,
    extract_text_content,
    fetch_item_info,
)
from .services.planner import resolve_playback_selection, select_music_plan
from .services.web_search import brave_web_search, needs_fresh_data


class RecommendationRequest(BaseModel):
    topic: str


class StopStrippingChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that ignores unsupported stop parameters."""

    async def _agenerate(self, messages, stop=None, **kwargs):  # type: ignore[override]
        kwargs.pop("stop", None)
        return await super()._agenerate(messages, stop=None, **kwargs)

    def _generate(self, messages, stop=None, **kwargs):  # type: ignore[override]
        kwargs.pop("stop", None)
        return super()._generate(messages, stop=None, **kwargs)


def _format_artists(artists: list[str]) -> str:
    return ", ".join(artists) if artists else "Unknown artist"


def _extract_error_text(tool_result: Dict[str, Any]) -> Optional[str]:
    for text in extract_text_content(tool_result):
        return text
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        return json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def _format_duration_ms(duration_ms: int) -> str:
    minutes = duration_ms // 60000
    seconds = (duration_ms % 60000) // 1000
    return f"{minutes}:{seconds:02d}"


def _coerce_artist_names(artists: Any) -> list[str]:
    if isinstance(artists, dict):
        name = artists.get("name")
        return [name] if isinstance(name, str) and name else []
    if isinstance(artists, list):
        names: list[str] = []
        for artist in artists:
            if isinstance(artist, str) and artist:
                names.append(artist)
            elif isinstance(artist, dict):
                name = artist.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
        return names
    if isinstance(artists, str) and artists:
        return [artists]
    return []


def _derive_additional_info(
    selection: Optional[PlaybackSelection],
    item_info: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    if not selection or not item_info:
        return {}

    additional_info: Dict[str, str] = {}
    item_type = selection.type

    if item_type == "track":
        song_bits: list[str] = []
        duration_ms = item_info.get("duration_ms")
        if isinstance(duration_ms, int) and duration_ms > 0:
            song_bits.append(f"Duration {_format_duration_ms(duration_ms)}")
        track_number = item_info.get("track_number")
        if isinstance(track_number, int) and track_number > 0:
            song_bits.append(f"Track {track_number}")
        explicit = item_info.get("explicit")
        if explicit is True:
            song_bits.append("Explicit")
        if song_bits:
            additional_info["song"] = ", ".join(song_bits)

        album_info = item_info.get("album")
        if isinstance(album_info, dict):
            album_name = album_info.get("name")
            if isinstance(album_name, str) and album_name:
                additional_info["album"] = f"From the album {album_name}."
    elif item_type == "album":
        album_bits: list[str] = []
        release = item_info.get("release_date") or item_info.get("releaseDate")
        if isinstance(release, str) and release:
            album_bits.append(f"Released {release}")
        total_tracks = item_info.get("total_tracks")
        if isinstance(total_tracks, int) and total_tracks > 0:
            album_bits.append(f"{total_tracks} tracks")
        genres = item_info.get("genres")
        if isinstance(genres, list):
            genre_list = [genre for genre in genres if isinstance(genre, str) and genre]
            if genre_list:
                album_bits.append(f"Genres: {', '.join(genre_list)}")
        if album_bits:
            additional_info["album"] = ". ".join(album_bits) + "."

    artists = _coerce_artist_names(item_info.get("artists") or item_info.get("artist"))
    if artists:
        additional_info["artist"] = f"Artists: {', '.join(artists)}."

    return additional_info


async def _generate_fun_fact_and_info(
    llm: ChatOpenAI,
    topic: str,
    selection: Optional[PlaybackSelection],
    item_info: Optional[Dict[str, Any]],
    snippets: list[SearchSnippet],
) -> tuple[Optional[str], Dict[str, str]]:
    if not selection:
        return None, {}

    fallback_info = _derive_additional_info(selection, item_info)
    fallback_fact = None
    if item_info:
        release = item_info.get("release_date") or item_info.get("releaseDate")
        if isinstance(release, str) and release:
            fallback_fact = f"It first landed in {release}."

    snippet_text = "\n".join(
        f"- {item.title}: {item.snippet} (source: {item.url})"
        for item in snippets
        if item.snippet
    )

    selection_summary = {
        "type": selection.type,
        "name": selection.name,
        "artists": selection.artists,
        "uri": selection.uri,
    }
    info_summary = item_info or {}

    prompt = (
        "You are a music expert. Use only the provided data to write:\n"
        "- one fun fact (1 short sentence)\n"
        "- additional info about the album, song, and artist (each 1 short sentence if possible)\n"
        "Return ONLY JSON with keys: fun_fact, album, song, artist.\n"
        "If you cannot support a field with the data, use an empty string.\n"
        f"Listener request: {topic}\n"
        f"Selection: {selection_summary}\n"
        f"Item info: {info_summary}\n"
    )
    if snippet_text:
        prompt += "Web snippets:\n" + snippet_text + "\n"

    try:
        response = (await llm.apredict(prompt)).strip()
        data = _extract_json_object(response)
    except Exception as exc:
        logger.warning("Fun fact generation failed: %s", exc)
        data = None

    fun_fact = None
    additional_info: Dict[str, str] = dict(fallback_info)

    if isinstance(data, dict):
        fun_fact_value = data.get("fun_fact")
        if isinstance(fun_fact_value, str) and fun_fact_value.strip():
            fun_fact = fun_fact_value.strip()

        for key in ("album", "song", "artist"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                additional_info[key] = value.strip()

    if not fun_fact:
        fun_fact = fallback_fact

    return fun_fact, additional_info


def _compose_recommendation_message(
    topic: str,
    plan: Optional[SelectionPlan],
    selection: Optional[PlaybackSelection],
    item_info: Optional[Dict[str, Any]],
    playback_status: str,
    playback_error: Optional[str],
    fun_fact: Optional[str],
) -> str:
    if not selection:
        return (
            "Couldn’t lock onto an exact match, so here’s a jazz staple you can’t miss: "
            "Miles Davis – Kind of Blue. Modal magic, endlessly replayable. "
            "Cue it up when you’re ready and let the cool tones flow."
        )

    artists = _format_artists(selection.artists)
    title = selection.name or "Unknown selection"
    lines: list[str] = []

    if plan and plan.announcement:
        lines.append(plan.announcement.strip())
    else:
        lines.append(f"Spinning {title} by {artists} — let's ride the groove.")

    if plan and plan.notes:
        lines.append(plan.notes.strip())

    if item_info:
        release = item_info.get("release_date") or item_info.get("releaseDate")
        genres = item_info.get("genres")
        if release:
            lines.append(f"Released in {release}, it still sounds timeless.")
        if isinstance(genres, list) and genres:
            lines.append(f"Expect shades of {', '.join(genres)}.")

    if playback_status == "started":
        lines.append("Playback is live — enjoy the vibe!")
    elif playback_status == "failed":
        lines.append(f"Tried to start playback but hit a snag: {playback_error}.")
    else:
        lines.append("Couldn’t autoplay, but queue it up when you can.")

    if fun_fact:
        lines.append(f"Fun fact: {fun_fact}")

    return " ".join(line for line in lines if line)


# -- Application lifecycle ---------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tools = []
    app.state.mcp_url = None
    app.state.llm = None

    async def _initialize() -> None:
        try:
            tools, resolved_url = await discover_tools(os.getenv("MCP_SERVER_URL"))
            app.state.tools = tools
            app.state.mcp_url = resolved_url
            logger.debug("Discovered %d MCP tools via %s", len(tools), resolved_url)
        except Exception as exc:
            logger.exception("Tool discovery failed: %s", exc)
            app.state.mcp_url = ensure_streamable_http_url(
                os.getenv("MCP_SERVER_URL", "http://127.0.0.1:9082")
            )

        if os.getenv("OPENAI_API_KEY"):
            try:
                app.state.llm = StopStrippingChatOpenAI(model="gpt-4.1-nano")
            except Exception as exc:
                logger.exception("Failed to initialize OpenAI client: %s", exc)
        else:
            logger.info("OPENAI_API_KEY not set; skipping LLM initialization")

    init_task = asyncio.create_task(_initialize())

    try:
        yield
    finally:
        if not init_task.done():
            init_task.cancel()
            try:
                await init_task
            except asyncio.CancelledError:
                pass


app = FastAPI(lifespan=lifespan)


@app.post("/recommend")
async def recommend_music(payload: RecommendationRequest):
    llm: Optional[ChatOpenAI] = getattr(app.state, "llm", None)
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")

    topic = payload.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must not be empty")

    mcp_url = app.state.mcp_url or ensure_streamable_http_url(
        os.getenv("MCP_SERVER_URL", "http://127.0.0.1:9082")
    )

    snippets: list[SearchSnippet] = []
    if needs_fresh_data(topic):
        snippets = await brave_web_search(topic, limit=5)
        print(f"*** {snippets = }")

    plan = await select_music_plan(llm, topic, snippets=snippets)
    print(f"*** {plan = }")
    selection = await resolve_playback_selection(topic, mcp_url, plan)
    if not selection:
        selection = await resolve_playback_selection(topic, mcp_url, None)
    print(f"*** {selection = }")

    item_info = await fetch_item_info(selection.uri, mcp_url) if selection else None
    print(f"*** {item_info = }")

    fun_fact: Optional[str] = None
    additional_info: Dict[str, str] = {}
    if selection and llm:
        fun_fact, additional_info = await _generate_fun_fact_and_info(
            llm=llm,
            topic=topic,
            selection=selection,
            item_info=item_info,
            snippets=snippets,
        )

    playback_result: Optional[dict[str, Any]] = None
    playback_status = "skipped"
    playback_error: Optional[str] = None

    if selection:
        try:
            playback_result = await call_mcp_tool(
                "SpotifyPlayback",
                {"action": "start", "spotify_uri": selection.uri},
                mcp_url,
            )
            if playback_result.get("isError"):
                playback_status = "failed"
                playback_error = _extract_error_text(playback_result)
            else:
                playback_status = "started"
        except Exception as exc:
            playback_status = "failed"
            playback_error = str(exc)

    recommendation = _compose_recommendation_message(
        topic=topic,
        plan=plan,
        selection=selection,
        item_info=item_info,
        playback_status=playback_status,
        playback_error=playback_error,
        fun_fact=fun_fact,
    )

    response: Dict[str, Any] = {
        "topic": topic,
        "recommendation": recommendation,
        "playback_status": playback_status,
    }

    if plan:
        response["llm_plan"] = asdict(plan)
    if snippets:
        response["web_search"] = [asdict(snippet) for snippet in snippets]
    if selection:
        response["selection"] = asdict(selection)
    if item_info:
        response["item_info"] = item_info
    if fun_fact:
        response["fun_fact"] = fun_fact
    if additional_info:
        response["additional_info"] = additional_info
    if playback_result:
        response["playback"] = playback_result
    if playback_error:
        response["playback_error"] = playback_error

    return response
