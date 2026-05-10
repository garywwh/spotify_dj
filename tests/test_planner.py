import json
import unittest
from unittest.mock import patch

from spotify_dj.models import RecommendationPlan, SelectionPlan
from spotify_dj.services import planner


def _spotify_search_result(results):
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps({"status": "ok", "results": results}),
            }
        ]
    }


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def apredict(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class ResolvePlaybackSelectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_generic_prompt_overrides_specific_llm_intent(self):
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "intent": "specific",
                        "candidates": [
                            {
                                "title": "Wildflowers: The New York Loft Jazz Sessions",
                                "artists": ["Various Artists"],
                                "type": "album",
                                "notes": "Loft jazz anthology.",
                            },
                            {
                                "title": "Second May Jazz Album",
                                "artists": ["Second Artist"],
                                "type": "album",
                            },
                        ],
                    }
                )
            ]
        )

        plan = await planner.select_recommendation_plan(
            llm,
            "Recommend a jazz album from new york released in May 2026",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.intent, "generic")
        self.assertEqual(len(plan.candidates), 2)

    async def test_quoted_specific_prompt_wins_over_generic_cues(self):
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "intent": "generic",
                        "candidates": [
                            {
                                "title": "New York State of Mind",
                                "artists": ["Billy Joel"],
                                "type": "track",
                            }
                        ],
                    }
                )
            ]
        )

        plan = await planner.select_recommendation_plan(
            llm,
            'Play "New York State of Mind" by Billy Joel',
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.intent, "specific")

    async def test_playlist_candidate_is_normalized_to_album(self):
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "intent": "generic",
                        "candidates": [
                            {
                                "title": "Morning Jazz",
                                "artists": ["Various Artists"],
                                "type": "playlist",
                            },
                            {
                                "title": "Second Album",
                                "artists": ["Second Artist"],
                                "type": "album",
                            },
                        ],
                    }
                )
            ]
        )

        plan = await planner.select_recommendation_plan(
            llm,
            "Recommend morning jazz",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.candidates[0].type, "album")

    async def test_generic_prompt_retries_when_llm_returns_one_candidate(self):
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "intent": "generic",
                        "candidates": [
                            {
                                "title": "From Japan With Love",
                                "artists": ["Minyo Crusaders"],
                                "type": "album",
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "intent": "generic",
                        "candidates": [
                            {
                                "title": "From Japan With Love",
                                "artists": ["Minyo Crusaders"],
                                "type": "album",
                            },
                            {
                                "title": "Second Japanese Album",
                                "artists": ["Second Artist"],
                                "type": "album",
                            },
                        ],
                    }
                ),
            ]
        )

        plan = await planner.select_recommendation_plan(
            llm,
            "what is the latest album from japan",
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.intent, "generic")
        self.assertEqual([candidate.title for candidate in plan.candidates], ["From Japan With Love", "Second Japanese Album"])
        self.assertEqual(len(llm.prompts), 2)

    async def test_planned_album_does_not_fall_back_to_unrelated_broad_result(self):
        plan = SelectionPlan(
            title="From Japan With Love",
            artists=["Minyo Crusaders"],
            type="album",
        )
        calls = []

        async def fake_call_mcp_tool(tool_name, arguments, mcp_url):
            calls.append(arguments)
            return _spotify_search_result(
                {
                    "albums": [
                        {
                            "name": "midnight cruisin'",
                            "id": "30lgWjklkY1TOx7EdiGYlq",
                            "artist": "Kingo Hamada",
                        }
                    ]
                }
            )

        with patch.object(planner, "call_mcp_tool", side_effect=fake_call_mcp_tool):
            selection = await planner.resolve_playback_selection(
                "what is the latest album from japan",
                "http://mcp.test/mcp",
                plan,
            )

        self.assertIsNone(selection)
        self.assertEqual(
            calls,
            [
                {
                    "query": "From Japan With Love Minyo Crusaders",
                    "qtype": "album",
                    "limit": 5,
                },
                {
                    "query": "From Japan With Love Minyo Crusaders",
                    "qtype": "track",
                    "limit": 5,
                },
            ],
        )

    async def test_planned_album_accepts_matching_spotify_result(self):
        plan = SelectionPlan(
            title="From Japan With Love",
            artists=["Minyo Crusaders"],
            type="album",
        )

        async def fake_call_mcp_tool(tool_name, arguments, mcp_url):
            return _spotify_search_result(
                {
                    "albums": [
                        {
                            "name": "From Japan With Love",
                            "id": "album123",
                            "artist": "Minyo Crusaders",
                        }
                    ]
                }
            )

        with patch.object(planner, "call_mcp_tool", side_effect=fake_call_mcp_tool):
            selection = await planner.resolve_playback_selection(
                "what is the latest album from japan",
                "http://mcp.test/mcp",
                plan,
            )

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.name, "From Japan With Love")
        self.assertEqual(selection.artists, ["Minyo Crusaders"])
        self.assertEqual(selection.uri, "spotify:album:album123")

    async def test_generic_recommendation_plan_tries_next_candidate(self):
        recommendation_plan = RecommendationPlan(
            intent="generic",
            candidates=[
                SelectionPlan(
                    title="Unavailable May Jazz",
                    artists=["First Artist"],
                    type="album",
                ),
                SelectionPlan(
                    title="Playable May Jazz",
                    artists=["Second Artist"],
                    type="album",
                ),
            ],
        )
        calls = []

        async def fake_call_mcp_tool(tool_name, arguments, mcp_url):
            calls.append(arguments)
            if arguments["query"] == "Playable May Jazz Second Artist":
                return _spotify_search_result(
                    {
                        "albums": [
                            {
                                "name": "Playable May Jazz",
                                "id": "playable123",
                                "artist": "Second Artist",
                            }
                        ]
                    }
                )
            return _spotify_search_result(
                {
                    "albums": [
                        {
                            "name": "Different Album",
                            "id": "different123",
                            "artist": "Different Artist",
                        }
                    ]
                }
            )

        with patch.object(planner, "call_mcp_tool", side_effect=fake_call_mcp_tool):
            selection, matched_plan, attempted = await planner.resolve_recommendation_plan(
                "Recommend a jazz album from new york released in May 2026",
                "http://mcp.test/mcp",
                recommendation_plan,
            )

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.uri, "spotify:album:playable123")
        self.assertIsNotNone(matched_plan)
        assert matched_plan is not None
        self.assertEqual(matched_plan.title, "Playable May Jazz")
        self.assertEqual([candidate.title for candidate in attempted], ["Unavailable May Jazz", "Playable May Jazz"])
        self.assertEqual(
            [call["query"] for call in calls],
            [
                "Unavailable May Jazz First Artist",
                "Unavailable May Jazz First Artist",
                "Playable May Jazz Second Artist",
            ],
        )

    async def test_specific_recommendation_plan_stops_after_first_candidate(self):
        recommendation_plan = RecommendationPlan(
            intent="specific",
            candidates=[
                SelectionPlan(
                    title="Unavailable Specific Album",
                    artists=["Named Artist"],
                    type="album",
                ),
                SelectionPlan(
                    title="Playable Alternative",
                    artists=["Other Artist"],
                    type="album",
                ),
            ],
        )
        calls = []

        async def fake_call_mcp_tool(tool_name, arguments, mcp_url):
            calls.append(arguments)
            return _spotify_search_result(
                {
                    "albums": [
                        {
                            "name": "Playable Alternative",
                            "id": "playable456",
                            "artist": "Other Artist",
                        }
                    ]
                }
            )

        with patch.object(planner, "call_mcp_tool", side_effect=fake_call_mcp_tool):
            selection, matched_plan, attempted = await planner.resolve_recommendation_plan(
                "Play Unavailable Specific Album by Named Artist",
                "http://mcp.test/mcp",
                recommendation_plan,
            )

        self.assertIsNone(selection)
        self.assertIsNotNone(matched_plan)
        assert matched_plan is not None
        self.assertEqual(matched_plan.title, "Unavailable Specific Album")
        self.assertEqual([candidate.title for candidate in attempted], ["Unavailable Specific Album"])
        self.assertEqual(
            [call["query"] for call in calls],
            [
                "Unavailable Specific Album Named Artist",
                "Unavailable Specific Album Named Artist",
            ],
        )

    async def test_generic_recommendation_plan_skips_validator_failure(self):
        recommendation_plan = RecommendationPlan(
            intent="generic",
            candidates=[
                SelectionPlan(
                    title="Stale Latest Album",
                    artists=["First Artist"],
                    type="album",
                ),
                SelectionPlan(
                    title="Fresh Latest Album",
                    artists=["Second Artist"],
                    type="album",
                ),
            ],
        )
        validated = []

        async def fake_call_mcp_tool(tool_name, arguments, mcp_url):
            if arguments["query"] == "Fresh Latest Album Second Artist":
                return _spotify_search_result(
                    {
                        "albums": [
                            {
                                "name": "Fresh Latest Album",
                                "id": "fresh123",
                                "artist": "Second Artist",
                            }
                        ]
                    }
                )
            return _spotify_search_result(
                {
                    "albums": [
                        {
                            "name": "Stale Latest Album",
                            "id": "stale123",
                            "artist": "First Artist",
                        }
                    ]
                }
            )

        async def validator(selection, candidate):
            validated.append(selection.uri)
            return selection.uri != "spotify:album:stale123"

        with patch.object(planner, "call_mcp_tool", side_effect=fake_call_mcp_tool):
            selection, matched_plan, attempted = await planner.resolve_recommendation_plan(
                "what is the latest album from japan",
                "http://mcp.test/mcp",
                recommendation_plan,
                validator=validator,
            )

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.uri, "spotify:album:fresh123")
        self.assertIsNotNone(matched_plan)
        assert matched_plan is not None
        self.assertEqual(matched_plan.title, "Fresh Latest Album")
        self.assertEqual([candidate.title for candidate in attempted], ["Stale Latest Album", "Fresh Latest Album"])
        self.assertEqual(validated, ["spotify:album:stale123", "spotify:album:fresh123"])


if __name__ == "__main__":
    unittest.main()
