"""Real-time web search helpers."""

from __future__ import annotations

import os
import re
from html import unescape
from datetime import datetime, timezone
from typing import List

import httpx

from ..config import logger
from ..models import SearchSnippet


def needs_fresh_data(topic: str) -> bool:
    """Return True when the query likely requires up-to-date information."""
    lowered = topic.lower()
    current_year = datetime.now(timezone.utc).year
    mentioned_years = {int(year) for year in re.findall(r"\b((?:19|20)\d{2})\b", lowered)}
    freshness_cues = {
        "latest",
        "new release",
        "new album",
        "new song",
        "new single",
        "new music",
        "released this year",
        "just released",
        "current",
        "recent",
        "lately",
        "newly", 
        "today",
        "this week",
        "this month",
        "this year",
    }
    recent_years = {current_year, current_year - 1, current_year + 1}
    return any(cue in lowered for cue in freshness_cues) or bool(mentioned_years & recent_years)


def fresh_search_queries(topic: str) -> list[str]:
    """Build focused web queries for time-sensitive music requests."""
    current_year = datetime.now(timezone.utc).year
    lowered = topic.lower()
    queries = [topic]
    if needs_fresh_data(topic):
        if "album" in lowered:
            queries.append(f"{current_year} blues album releases United States")
            queries.append(f"{current_year} latest blues albums release date Spotify")
        else:
            queries.append(f"{topic} {current_year} release date")
    return list(dict.fromkeys(queries))


async def brave_web_search(query: str, limit: int = 5) -> List[SearchSnippet]:
    """Lookup recent information via Brave search API."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        logger.info("BRAVE_API_KEY not set; skipping web search")
        return []

    url = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": limit}
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Brave search failed for query '%s': %s", query, exc)
        return []

    snippets: List[SearchSnippet] = []
    for item in data.get("web", {}).get("results", [])[:limit]:
        title = item.get("title")
        result_url = item.get("url")
        snippet = item.get("description") or item.get("snippet")
        if not title or not result_url:
            continue
        snippets.append(SearchSnippet(title=title, url=result_url, snippet=snippet or ""))

    logger.debug("Brave search returned %d snippets for '%s'", len(snippets), query)
    return snippets


def _html_to_text(value: str) -> str:
    without_scripts = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", value, flags=re.I | re.S)
    without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def _candidate_page_lines(html: str, current_year: int) -> list[str]:
    """Extract compact music-release hints from result pages."""
    candidates: list[str] = []
    seen: set[str] = set()
    text = _html_to_text(html)

    chart_re = re.compile(
        r"(?:\b\d+\s+[▲▼]?\s*LW\s+\d+\s+)?(.{2,120}?)\s+by:\s+(.{2,80}?)\s+Label:",
        flags=re.I,
    )
    for match in chart_re.finditer(text):
        title = re.sub(r"\s+", " ", match.group(1)).strip(" -|")
        artist = re.sub(r"\s+", " ", match.group(2)).strip(" -|")
        line = f"{title} by {artist} ({current_year} blues album chart)"
        normalized = line.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(line)
        if len(candidates) >= 12:
            return candidates

    attr_re = re.compile(r"\b(?:alt|title)=([\"'])(.*?)\1", flags=re.I | re.S)
    for match in attr_re.finditer(html):
        text = _html_to_text(match.group(2))
        lowered = text.lower()
        if not text or len(text) > 180:
            continue
        if str(current_year) not in text and not any(word in lowered for word in ("blues", "album", "release")):
            continue
        normalized = text.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(text)
        if len(candidates) >= 12:
            break

    if len(candidates) >= 5:
        return candidates

    year_match_re = re.compile(rf"(.{{0,90}}{current_year}.{{0,120}})", flags=re.I)
    for match in year_match_re.finditer(text):
        line = re.sub(r"\s+", " ", match.group(1)).strip(" -|")
        lowered = line.lower()
        if len(line) < 20 or len(line) > 220:
            continue
        if not any(word in lowered for word in ("blues", "album", "release")):
            continue
        normalized = line.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(line)
        if len(candidates) >= 12:
            break
    return candidates


async def expand_fresh_snippets(snippets: list[SearchSnippet], limit: int = 3) -> list[SearchSnippet]:
    """Fetch top result pages and add concise release hints for current-data prompts."""
    current_year = datetime.now(timezone.utc).year
    expanded: list[SearchSnippet] = []
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        for snippet in snippets[:limit]:
            try:
                resp = await client.get(snippet.url)
                resp.raise_for_status()
            except Exception as exc:
                logger.debug("Skipping page expansion for %s: %s", snippet.url, exc)
                continue

            lines = _candidate_page_lines(resp.text, current_year)
            if not lines:
                continue
            expanded.append(
                SearchSnippet(
                    title=f"Page details from {snippet.title}",
                    url=snippet.url,
                    snippet="; ".join(lines[:8]),
                )
            )
    return expanded


__all__ = ["needs_fresh_data", "fresh_search_queries", "brave_web_search", "expand_fresh_snippets"]
