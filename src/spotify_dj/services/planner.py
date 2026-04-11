"""LLM planning and selection helpers."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from langchain_openai import ChatOpenAI

from ..config import logger
from ..models import PlaybackSelection, SearchSnippet, SelectionPlan
from .mcp_client import call_mcp_tool, parse_first_json_block


def _select_album(data: dict[str, object]) -> Optional[PlaybackSelection]:
    results = data.get("results") or data
    albums = results.get("albums") if isinstance(results, dict) else None
    if not isinstance(albums, list) or not albums:
        return None

    album = albums[0]
    if not isinstance(album, dict):
        return None

    album_id = album.get("id")
    if not isinstance(album_id, str):
        return None

    artists_raw = album.get("artists") or ([album.get("artist")] if album.get("artist") else [])
    artists = [artist for artist in artists_raw if isinstance(artist, str) and artist]

    return PlaybackSelection(
        type="album",
        name=str(album.get("name", "")),
        artists=artists,
        uri=f"spotify:album:{album_id}",
        id=album_id,
    )


def _select_track(data: dict[str, object]) -> Optional[PlaybackSelection]:
    results = data.get("results") or data
    tracks = results.get("tracks") if isinstance(results, dict) else None
    if not isinstance(tracks, list) or not tracks:
        return None

    track = tracks[0]
    if not isinstance(track, dict):
        return None

    track_id = track.get("id")
    if not isinstance(track_id, str):
        return None

    artists_raw = track.get("artists") or ([track.get("artist")] if track.get("artist") else [])
    artists = [artist for artist in artists_raw if isinstance(artist, str) and artist]

    return PlaybackSelection(
        type="track",
        name=str(track.get("name", "")),
        artists=artists,
        uri=f"spotify:track:{track_id}",
        id=track_id,
    )


async def select_music_plan(
    llm: ChatOpenAI,
    topic: str,
    snippets: Optional[List[SearchSnippet]] = None,
) -> Optional[SelectionPlan]:
    prompt = (
        "You are a knowledgeable DJ planning the next spin. "
        "Return ONLY a JSON object with the shape: \n"
        "{\n"
        "  \"selection\": { \"title\": string, \"artists\": [strings], \"type\": 'album'|'track'|'playlist', \"notes\": string },\n"
        "  \"announcement\": string (a short DJ-style intro announcing immediate playback)\n"
        "}\n"
        "No text outside the JSON.\n"
        f"Listener request: {topic}\n"
    )

    if snippets:
        snippet_text = "\n".join(
            f"- {item.title}: {item.snippet} (source: {item.url})"
            for item in snippets
            if item.snippet
        )
        if snippet_text:
            prompt += "Recent web findings you can rely on:\n" + snippet_text + "\n"

    response = (await llm.apredict(prompt)).strip()
    data = _extract_json_object(response)
    if not data or not isinstance(data, dict):
        logger.warning("LLM plan did not return valid JSON for topic '%s'", topic)
        return None

    selection = data.get("selection") if isinstance(data.get("selection"), dict) else None
    if not selection:
        logger.warning("LLM plan missing 'selection' field for topic '%s'", topic)
        return None

    title = selection.get("title") or selection.get("name")
    if not isinstance(title, str) or not title.strip():
        logger.warning("LLM plan missing title for topic '%s'", topic)
        return None

    artists_field = selection.get("artists") or selection.get("artist")
    if isinstance(artists_field, str):
        artists = [artists_field]
    elif isinstance(artists_field, list):
        artists = [a for a in artists_field if isinstance(a, str) and a.strip()]
    else:
        artists = []

    item_type = selection.get("type") or selection.get("category") or "album"
    if not isinstance(item_type, str):
        item_type = "album"
    item_type = item_type.lower()
    if item_type not in {"track", "album", "playlist"}:
        item_type = "album"

    return SelectionPlan(
        title=title.strip(),
        artists=artists,
        type=item_type,
        notes=selection.get("notes") or selection.get("reason"),
        announcement=data.get("announcement"),
        raw_response=response,
    )


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        return json.loads(cleaned[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


async def resolve_playback_selection(
    topic: str,
    mcp_url: str,
    plan: Optional[SelectionPlan],
) -> Optional[PlaybackSelection]:
    search_variants: List[dict[str, Any]] = []

    if plan:
        title = plan.title
        artists = plan.artists or []
        item_type = plan.type or "album"
        artist_fragment = " ".join(artists) if artists else ""
        if title:
            query = f"{title} {artist_fragment}".strip()
            preferred_qtype = item_type if item_type in {"album", "track", "playlist"} else "album"
            search_variants.append({"query": query, "qtype": preferred_qtype, "limit": 5})
            if preferred_qtype != "track":
                search_variants.append({"query": query, "qtype": "track", "limit": 5})

    search_variants.extend(
        [
            {"query": topic, "qtype": "album", "limit": 5},
            {"query": f"{topic} album", "qtype": "album", "limit": 5},
            {"query": topic, "qtype": "track", "limit": 5},
        ]
    )

    for params in search_variants:
        try:
            result = await call_mcp_tool("SpotifySearch", params, mcp_url)
        except Exception:
            continue
        data = parse_first_json_block(result)
        if not data:
            continue
        album = _select_album(data)
        if album:
            return album
        track = _select_track(data)
        if track:
            return track
    return None


__all__ = ["select_music_plan", "resolve_playback_selection"]
