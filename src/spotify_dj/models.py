"""Dataclasses used across Spotify DJ services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchSnippet:
    title: str
    url: str
    snippet: str


@dataclass
class SelectionPlan:
    title: str
    artists: List[str]
    type: str
    notes: Optional[str] = None
    announcement: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class RecommendationPlan:
    intent: str
    candidates: List[SelectionPlan]
    raw_response: Optional[str] = None


@dataclass
class PlaybackSelection:
    name: str
    uri: str
    artists: List[str]
    type: str
    id: Optional[str] = None
