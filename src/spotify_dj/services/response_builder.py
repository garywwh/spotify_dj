"""User-facing recommendation response helpers."""

from __future__ import annotations

from typing import Any, Optional

from ..models import PlaybackSelection, RecommendationPlan, SelectionPlan
from .release_constraints import humanize_release_date


def format_artists(artists: list[str]) -> str:
    return ", ".join(artists) if artists else "Unknown artist"


def format_duration_ms(duration_ms: int) -> str:
    minutes = duration_ms // 60000
    seconds = (duration_ms % 60000) // 1000
    return f"{minutes}:{seconds:02d}"


def clean_info_sentence(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."


def coerce_artist_names(artists: Any) -> list[str]:
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


def derive_additional_info(
    selection: Optional[PlaybackSelection],
    item_info: Optional[dict[str, Any]],
) -> dict[str, str]:
    if not selection or not item_info:
        return {}

    additional_info: dict[str, str] = {}
    item_type = selection.type

    if item_type == "track":
        song_bits: list[str] = []
        duration_ms = item_info.get("duration_ms")
        if isinstance(duration_ms, int) and duration_ms > 0:
            song_bits.append(f"Duration {format_duration_ms(duration_ms)}")
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

    artists = coerce_artist_names(item_info.get("artists") or item_info.get("artist"))
    if artists:
        additional_info["artist"] = f"Artists: {', '.join(artists)}."

    return additional_info


def compose_recommendation_message(
    topic: str,
    recommendation_plan: Optional[RecommendationPlan],
    plan: Optional[SelectionPlan],
    selection: Optional[PlaybackSelection],
    item_info: Optional[dict[str, Any]],
    playback_status: str,
    playback_error: Optional[str],
    fun_fact: Optional[str],
    additional_info: Optional[dict[str, str]] = None,
    attempted_plans: Optional[list[SelectionPlan]] = None,
) -> str:
    if not selection:
        if plan:
            artists = format_artists(plan.artists)
            if recommendation_plan and recommendation_plan.intent == "generic":
                attempted_count = len(attempted_plans or recommendation_plan.candidates)
                return (
                    f"I checked {attempted_count} likely picks for that request, starting with "
                    f"{plan.title} by {artists}, but none resolved cleanly on Spotify. "
                    "I left playback stopped rather than putting on the wrong record."
                )
            return (
                f"I found {plan.title} by {artists}, but couldn't resolve that exact item on Spotify. "
                "I left playback stopped rather than swapping in a lookalike."
            )
        return (
            "I couldn't lock onto a clean Spotify match for that request. "
            "I left playback stopped so the next record is one we can stand behind."
        )

    artists = format_artists(selection.artists)
    title = selection.name or "Unknown selection"
    details = additional_info or {}
    lines: list[str] = [f"Weekend dial-in: I’m putting on {title} by {artists}."]

    album_context: list[str] = []
    album_info = clean_info_sentence(details.get("album"))
    artist_info = clean_info_sentence(details.get("artist"))
    song_info = clean_info_sentence(details.get("song"))
    notes = clean_info_sentence(plan.notes if plan else None)

    if notes:
        album_context.append(notes)
    if album_info and album_info not in album_context:
        album_context.append(album_info)

    if item_info:
        release = item_info.get("release_date") or item_info.get("releaseDate")
        release_text = humanize_release_date(release)
        total_tracks = item_info.get("total_tracks")
        genres = item_info.get("genres")
        metadata_bits: list[str] = []
        if release_text:
            metadata_bits.append(f"released {release_text}")
        if isinstance(total_tracks, int) and total_tracks > 0:
            metadata_bits.append(f"{total_tracks} tracks")
        if isinstance(genres, list) and genres:
            metadata_bits.append(", ".join(genres[:3]))
        if metadata_bits:
            album_context.append("On Spotify it’s listed as " + "; ".join(metadata_bits) + ".")

    if album_context:
        lines.append(" ".join(album_context[:3]))

    artist_context = artist_info or song_info
    if artist_context:
        lines.append(artist_context)

    if playback_status == "started":
        lines.append("Playback is rolling now; let it ease into the room.")
    elif playback_status == "failed":
        lines.append(f"I had the record picked, but Spotify playback hit a snag: {playback_error}.")
    else:
        lines.append("I found the record, but left autoplay alone.")

    if fun_fact:
        lines.append(f"One for the liner notes: {fun_fact}")

    return " ".join(line for line in lines if line)


__all__ = [
    "coerce_artist_names",
    "compose_recommendation_message",
    "derive_additional_info",
    "format_artists",
    "format_duration_ms",
]
