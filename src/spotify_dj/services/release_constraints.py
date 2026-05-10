"""Release-date parsing and validation for recommendation constraints."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def parse_release_date(value: Any) -> Optional[date]:
    if not isinstance(value, str) or not value:
        return None
    parts = value.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (TypeError, ValueError):
        return None


def requested_release_month(topic: str) -> Optional[tuple[int, int]]:
    lowered = topic.lower()
    for month_name, month_number in _MONTHS.items():
        match = re.search(rf"\b{month_name}\b\s+((?:19|20)\d{{2}})\b", lowered)
        if match:
            return int(match.group(1)), month_number
    return None


def requested_release_year(topic: str) -> Optional[int]:
    match = re.search(r"\b((?:19|20)\d{2})\b", topic)
    return int(match.group(1)) if match else None


def requires_current_release(topic: str) -> bool:
    lowered = topic.lower()
    cues = ("latest", "newest", "new album", "new release", "recent")
    return any(cue in lowered for cue in cues)


def has_release_constraints(topic: str) -> bool:
    return (
        requested_release_month(topic) is not None
        or requested_release_year(topic) is not None
        or requires_current_release(topic)
    )


def passes_release_constraints(topic: str, item_info: Optional[dict[str, Any]]) -> bool:
    if not has_release_constraints(topic):
        return True
    if not item_info:
        return False

    release_date = parse_release_date(item_info.get("release_date") or item_info.get("releaseDate"))
    if not release_date:
        return False

    today = date.today()
    if release_date > today:
        return False

    requested_month = requested_release_month(topic)
    if requested_month:
        year, month = requested_month
        return release_date.year == year and release_date.month == month

    requested_year = requested_release_year(topic)
    if requested_year:
        return release_date.year == requested_year

    if requires_current_release(topic):
        return release_date.year == today.year

    return True


def humanize_release_date(value: Any) -> Optional[str]:
    release_date = parse_release_date(value)
    if not release_date:
        return value if isinstance(value, str) and value else None

    month = release_date.strftime("%B")
    if isinstance(value, str) and len(value.split("-")) == 1:
        return str(release_date.year)
    if isinstance(value, str) and len(value.split("-")) == 2:
        return f"{month} {release_date.year}"
    return f"{month} {release_date.day}, {release_date.year}"


__all__ = [
    "has_release_constraints",
    "humanize_release_date",
    "parse_release_date",
    "passes_release_constraints",
    "requested_release_month",
    "requested_release_year",
    "requires_current_release",
]
