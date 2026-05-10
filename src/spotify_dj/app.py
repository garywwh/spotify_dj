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
from .services.planner import resolve_recommendation_plan, select_recommendation_plan
from .services.release_constraints import passes_release_constraints
from .services.response_builder import compose_recommendation_message, derive_additional_info
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


async def _generate_fun_fact_and_info(
    llm: ChatOpenAI,
    topic: str,
    selection: Optional[PlaybackSelection],
    item_info: Optional[Dict[str, Any]],
    snippets: list[SearchSnippet],
) -> tuple[Optional[str], Dict[str, str]]:
    if not selection:
        return None, {}

    fallback_info = derive_additional_info(selection, item_info)
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
        "- warm, specific additional info about the album, song, and artist (each 1 short sentence if possible)\n"
        "Tone: easy, informed radio DJ on a weekend morning. Avoid hype, cliches, and unsupported claims.\n"
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
        logger.debug("Web snippets for topic '%s': %s", topic, snippets)

    recommendation_plan = await select_recommendation_plan(llm, topic, snippets=snippets)
    logger.debug("Recommendation plan for topic '%s': %s", topic, recommendation_plan)
    item_info_cache: dict[str, Dict[str, Any]] = {}

    async def _validate_candidate(
        candidate_selection: PlaybackSelection,
        candidate_plan: SelectionPlan,
    ) -> bool:
        item_info = await fetch_item_info(candidate_selection.uri, mcp_url)
        if item_info:
            item_info_cache[candidate_selection.uri] = item_info
        if passes_release_constraints(topic, item_info):
            return True

        release = item_info.get("release_date") or item_info.get("releaseDate") if item_info else None
        logger.info(
            "Skipping candidate that failed release constraints",
            extra={
                "topic": topic,
                "candidate": candidate_plan.title,
                "spotify_uri": candidate_selection.uri,
                "release_date": release,
            },
        )
        return False

    selection, plan, attempted_plans = await resolve_recommendation_plan(
        topic,
        mcp_url,
        recommendation_plan,
        validator=_validate_candidate,
    )
    logger.debug("Matched plan for topic '%s': %s", topic, plan)
    logger.debug("Playback selection for topic '%s': %s", topic, selection)

    item_info = item_info_cache.get(selection.uri) if selection else None
    if selection and not item_info:
        item_info = await fetch_item_info(selection.uri, mcp_url)
    logger.debug("Item info for topic '%s': %s", topic, item_info)

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

    recommendation = compose_recommendation_message(
        topic=topic,
        recommendation_plan=recommendation_plan,
        plan=plan,
        selection=selection,
        item_info=item_info,
        playback_status=playback_status,
        playback_error=playback_error,
        fun_fact=fun_fact,
        additional_info=additional_info,
        attempted_plans=attempted_plans,
    )

    response: Dict[str, Any] = {
        "topic": topic,
        "recommendation": recommendation,
        "playback_status": playback_status,
    }

    if plan:
        response["llm_plan"] = asdict(plan)
    if recommendation_plan:
        response["recommendation_plan"] = asdict(recommendation_plan)
    if attempted_plans:
        response["attempted_plans"] = [asdict(attempted_plan) for attempted_plan in attempted_plans]
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
