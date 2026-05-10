"""Real-time web search helpers."""

from __future__ import annotations

import os
import re
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


__all__ = ["needs_fresh_data", "brave_web_search"]
