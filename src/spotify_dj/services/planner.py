"""LLM planning and selection helpers."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from typing import Any, Awaitable, Callable, List, Optional

from langchain_openai import ChatOpenAI

from ..config import logger
from ..models import PlaybackSelection, RecommendationPlan, SearchSnippet, SelectionPlan
from .mcp_client import call_mcp_tool, parse_first_json_block


_MEANINGFUL_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_QUOTED_TEXT_RE = re.compile(r"['\"“”‘’][^'\"“”‘’]+['\"“”‘’]")
_SPECIFIC_REQUEST_RE = re.compile(
    r"\b(play|listen to|spin|cue)\b.+\bby\b",
    re.IGNORECASE,
)
_GENERIC_REQUEST_RE = re.compile(
    r"\b("
    r"recommend|recommendation|suggest|latest|newest|recent|"
    r"new album|new release|new song|new single|new music|"
    r"something|anything|album from|track from|song from|"
    r"album by|track by|song by|released in|released on"
    r")\b",
    re.IGNORECASE,
)
_TRACK_REQUEST_RE = re.compile(r"\b(song|track|tune|single)\b", re.IGNORECASE)
_ALBUM_REQUEST_RE = re.compile(r"\balbum\b", re.IGNORECASE)
_ARTIST_FROM_RE = re.compile(r"\bfrom\s+(.+)$", re.IGNORECASE)
_ARTIST_BY_RE = re.compile(r"\b(?:album|song|track|tune|single)\s+by\s+(.+)$", re.IGNORECASE)
_ARTIST_BEFORE_MEDIA_RE = re.compile(
    r"^(?:play|recommend|suggest|spin|cue|listen to)?\s*(?:an?|some|the)?\s*(.+?)\s+"
    r"(?:album|song|track|tune|single)$",
    re.IGNORECASE,
)
_ARTIST_ONLY_MEDIA_REQUEST_RE = re.compile(
    r"(?:\b(?:an?|some|the)?\s*(?:album|song|track|tune|single)\s+by\s+.+$|"
    r"^(?:play|recommend|suggest|spin|cue|listen to)?\s*(?:an?|some|the)?\s*.+?\s+"
    r"(?:album|song|track|tune|single)$)",
    re.IGNORECASE,
)


async def llm_text_response(llm: ChatOpenAI, prompt: str) -> str:
    """Return text from current and older LangChain chat model APIs."""
    if hasattr(llm, "ainvoke"):
        response = await llm.ainvoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    response = await llm.apredict(prompt)  # type: ignore[attr-defined]
    return str(response)


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value.casefold())
    return " ".join(_MEANINGFUL_TOKEN_RE.findall(normalized))


def _title_matches(expected: str, actual: str) -> bool:
    expected_normalized = _normalize_text(expected)
    actual_normalized = _normalize_text(actual)
    if not expected_normalized or not actual_normalized:
        return False
    if expected_normalized == actual_normalized:
        return True

    expected_tokens = set(expected_normalized.split())
    actual_tokens = set(actual_normalized.split())
    if len(expected_tokens) < 2:
        return False
    return expected_tokens.issubset(actual_tokens)


def _artist_matches(expected_artists: list[str], actual_artists: list[str]) -> bool:
    if not expected_artists:
        return True

    expected = [_normalize_text(artist) for artist in expected_artists]
    actual = [_normalize_text(artist) for artist in actual_artists]
    for expected_artist in expected:
        if not expected_artist:
            continue
        for actual_artist in actual:
            if expected_artist == actual_artist:
                return True
    return False


def _matches_plan(selection: PlaybackSelection, plan: Optional[SelectionPlan]) -> bool:
    if not plan:
        return True
    return _title_matches(plan.title, selection.name) and _artist_matches(plan.artists, selection.artists)


def _infer_request_intent(topic: str, llm_intent: str) -> str:
    """Keep generic recommendation prompts on the multi-candidate path."""
    if _ARTIST_ONLY_MEDIA_REQUEST_RE.search(topic):
        return "generic"
    if _QUOTED_TEXT_RE.search(topic) or _SPECIFIC_REQUEST_RE.search(topic):
        return "specific"
    if _GENERIC_REQUEST_RE.search(topic):
        return "generic"
    return llm_intent


def _requires_current_release(topic: str) -> bool:
    lowered = topic.lower()
    return any(cue in lowered for cue in ("latest", "newest", "new album", "new release", "recent"))


def _topic_prefers_track(topic: str) -> bool:
    return bool(_TRACK_REQUEST_RE.search(topic))


def _topic_prefers_album(topic: str) -> bool:
    return bool(_ALBUM_REQUEST_RE.search(topic))


def _topic_search_queries(topic: str) -> list[str]:
    queries = [topic]
    for pattern in (_ARTIST_FROM_RE, _ARTIST_BY_RE, _ARTIST_BEFORE_MEDIA_RE):
        match = pattern.search(topic)
        if not match:
            continue
        artist = match.group(1).strip(" .?!")
        if artist:
            queries.insert(0, artist)
            break
    return list(dict.fromkeys(queries))


def _select_album(
    data: dict[str, object],
    plan: Optional[SelectionPlan] = None,
) -> Optional[PlaybackSelection]:
    results = data.get("results") or data
    albums = results.get("albums") if isinstance(results, dict) else None
    if not isinstance(albums, list) or not albums:
        return None

    for album in albums:
        if not isinstance(album, dict):
            continue

        album_id = album.get("id")
        if not isinstance(album_id, str):
            continue

        artists_raw = album.get("artists") or ([album.get("artist")] if album.get("artist") else [])
        artists = [artist for artist in artists_raw if isinstance(artist, str) and artist]

        selection = PlaybackSelection(
            type="album",
            name=str(album.get("name", "")),
            artists=artists,
            uri=f"spotify:album:{album_id}",
            id=album_id,
        )
        if _matches_plan(selection, plan):
            return selection
    return None


def _select_track(
    data: dict[str, object],
    plan: Optional[SelectionPlan] = None,
) -> Optional[PlaybackSelection]:
    results = data.get("results") or data
    tracks = results.get("tracks") if isinstance(results, dict) else None
    if not isinstance(tracks, list) or not tracks:
        return None

    for track in tracks:
        if not isinstance(track, dict):
            continue

        track_id = track.get("id")
        if not isinstance(track_id, str):
            continue

        artists_raw = track.get("artists") or ([track.get("artist")] if track.get("artist") else [])
        artists = [artist for artist in artists_raw if isinstance(artist, str) and artist]

        selection = PlaybackSelection(
            type="track",
            name=str(track.get("name", "")),
            artists=artists,
            uri=f"spotify:track:{track_id}",
            id=track_id,
        )
        if _matches_plan(selection, plan):
            return selection
    return None


async def select_music_plan(
    llm: ChatOpenAI,
    topic: str,
    snippets: Optional[List[SearchSnippet]] = None,
) -> Optional[SelectionPlan]:
    recommendation_plan = await select_recommendation_plan(llm, topic, snippets=snippets)
    if not recommendation_plan or not recommendation_plan.candidates:
        return None
    return recommendation_plan.candidates[0]


async def select_recommendation_plan(
    llm: ChatOpenAI,
    topic: str,
    snippets: Optional[List[SearchSnippet]] = None,
) -> Optional[RecommendationPlan]:
    today = date.today()
    prompt = (
        "You are a knowledgeable DJ planning the next spin. "
        "Return ONLY a JSON object with the shape: \n"
        "{\n"
        "  \"intent\": \"specific\" | \"generic\",\n"
        "  \"candidates\": [\n"
        "    { \"title\": string, \"artists\": [strings], \"type\": 'album'|'track', \"notes\": string, \"announcement\": string }\n"
        "  ]\n"
        "}\n"
        "Use intent=specific when the listener names a particular album, track, or artist/title target. "
        "Use intent=generic when the listener asks for a recommendation matching constraints. "
        "For generic requests, return exactly 5 distinct ranked candidates whenever possible; "
        "do not stop after the first plausible answer. "
        "For specific requests, return only the requested target. "
        "No text outside the JSON.\n"
        f"Today's date is {today.isoformat()}.\n"
        f"Listener request: {topic}\n"
    )
    if _requires_current_release(topic):
        prompt += (
            f"The listener is asking for a current/latest release. Treat this as a {today.year} release constraint "
            "unless the listener names another year. Use only real album/song titles and real artist names supported "
            "by the web findings or reliable music knowledge. Do not return older releases, best-of classics, "
            "or candidates with unknown artists.\n"
        )

    if snippets:
        snippet_text = "\n".join(
            f"- {item.title}: {item.snippet} (source: {item.url})"
            for item in snippets
            if item.snippet
        )
        if snippet_text:
            prompt += "Recent web findings you can rely on:\n" + snippet_text + "\n"

    response = (await llm_text_response(llm, prompt)).strip()
    parsed = _parse_recommendation_response(topic, response)
    if not parsed:
        logger.warning("LLM plan missing usable candidates for topic '%s'", topic)
        return None
    intent, candidates = parsed

    if intent == "generic" and len(candidates) < 2:
        retry_prompt = (
            prompt
            + "\nYour previous response had too few candidates for a generic recommendation. "
            + "Return ONLY JSON with intent=\"generic\" and exactly 5 distinct candidates. "
            + "Do not repeat candidates that may be unavailable on Spotify.\n"
        )
        retry_response = (await llm_text_response(llm, retry_prompt)).strip()
        retry_parsed = _parse_recommendation_response(topic, retry_response)
        if retry_parsed:
            retry_intent, retry_candidates = retry_parsed
            if retry_intent == "generic" and len(retry_candidates) >= 2:
                response = retry_response
                intent = retry_intent
                candidates = retry_candidates

    return RecommendationPlan(
        intent=intent,
        candidates=candidates,
        raw_response=response,
    )


def _parse_recommendation_response(
    topic: str,
    response: str,
) -> Optional[tuple[str, list[SelectionPlan]]]:
    data = _extract_json_object(response)
    if not data or not isinstance(data, dict):
        logger.warning("LLM plan did not return valid JSON for topic '%s'", topic)
        return None

    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list):
        selection = data.get("selection") if isinstance(data.get("selection"), dict) else None
        raw_candidates = [selection] if selection else []

    candidates: list[SelectionPlan] = []
    for raw_candidate in raw_candidates[:5]:
        if not isinstance(raw_candidate, dict):
            continue

        candidate = _parse_selection_plan(raw_candidate, response, data.get("announcement"))
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return None

    intent = data.get("intent") or data.get("specificity") or "generic"
    if not isinstance(intent, str):
        intent = "generic"
    intent = intent.lower()
    if intent not in {"specific", "generic"}:
        intent = "generic"
    return _infer_request_intent(topic, intent), candidates


def _parse_selection_plan(
    selection: dict[str, Any],
    raw_response: str,
    fallback_announcement: object = None,
) -> Optional[SelectionPlan]:
    title = selection.get("title") or selection.get("name")
    if not isinstance(title, str) or not title.strip():
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
    if item_type not in {"track", "album"}:
        item_type = "album"

    announcement = selection.get("announcement") or fallback_announcement
    if not isinstance(announcement, str):
        announcement = None

    return SelectionPlan(
        title=title.strip(),
        artists=artists,
        type=item_type,
        notes=selection.get("notes") or selection.get("reason"),
        announcement=announcement,
        raw_response=raw_response,
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
            preferred_qtype = item_type if item_type in {"album", "track"} else "album"
            search_variants.append({"query": query, "qtype": preferred_qtype, "limit": 5})
            if preferred_qtype != "track":
                search_variants.append({"query": query, "qtype": "track", "limit": 5})
    else:
        if _topic_prefers_track(topic) and not _topic_prefers_album(topic):
            qtypes = ["track", "album"]
        else:
            qtypes = ["album", "track"]
        for query in _topic_search_queries(topic):
            for qtype in qtypes:
                search_variants.append({"query": query, "qtype": qtype, "limit": 5})
            if "album" in qtypes:
                search_variants.append({"query": f"{query} album", "qtype": "album", "limit": 5})

    for params in search_variants:
        try:
            result = await call_mcp_tool("SpotifySearch", params, mcp_url)
        except Exception:
            continue
        data = parse_first_json_block(result)
        if not data:
            continue
        album = _select_album(data, plan=plan)
        if album:
            return album
        track = _select_track(data, plan=plan)
        if track:
            return track
    return None


async def resolve_recommendation_plan(
    topic: str,
    mcp_url: str,
    recommendation_plan: Optional[RecommendationPlan],
    validator: Optional[Callable[[PlaybackSelection, SelectionPlan], Awaitable[bool]]] = None,
) -> tuple[Optional[PlaybackSelection], Optional[SelectionPlan], list[SelectionPlan]]:
    if not recommendation_plan:
        selection = await resolve_playback_selection(topic, mcp_url, None)
        return selection, None, []

    attempted: list[SelectionPlan] = []
    for candidate in recommendation_plan.candidates[:5]:
        attempted.append(candidate)
        selection = await resolve_playback_selection(topic, mcp_url, candidate)
        if selection:
            if validator and not await validator(selection, candidate):
                if recommendation_plan.intent == "specific":
                    break
                continue
            return selection, candidate, attempted

        if recommendation_plan.intent == "specific":
            break

    if recommendation_plan.intent == "generic":
        selection = await resolve_playback_selection(topic, mcp_url, None)
        if selection:
            fallback_plan = recommendation_plan.candidates[0] if recommendation_plan.candidates else None
            if validator and fallback_plan and not await validator(selection, fallback_plan):
                return None, fallback_plan, attempted
            return selection, None, attempted

    return None, recommendation_plan.candidates[0] if recommendation_plan.candidates else None, attempted


__all__ = [
    "select_music_plan",
    "select_recommendation_plan",
    "resolve_playback_selection",
    "resolve_recommendation_plan",
]
