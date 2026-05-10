import unittest

from spotify_dj.models import PlaybackSelection, RecommendationPlan, SelectionPlan
from spotify_dj.services import response_builder


class RecommendationMessageTest(unittest.TestCase):
    def test_started_playback_message_reads_like_radio_recommendation(self):
        message = response_builder.compose_recommendation_message(
            topic="Recommend a jazz album from new york released in May 2026",
            recommendation_plan=RecommendationPlan(intent="generic", candidates=[]),
            plan=SelectionPlan(
                title="Test Session",
                artists=["A. Musician"],
                type="album",
                notes="A small-room jazz date with patient ensemble writing.",
            ),
            selection=PlaybackSelection(
                name="Test Session",
                uri="spotify:album:test",
                artists=["A. Musician"],
                type="album",
            ),
            item_info={
                "release_date": "2026-05-01",
                "total_tracks": 8,
                "genres": ["jazz"],
            },
            playback_status="started",
            playback_error=None,
            fun_fact="The session was recorded in New York.",
            additional_info={
                "artist": "A. Musician is known for spacious, lyrical horn lines.",
                "album": "The album keeps the arrangements close and conversational.",
            },
        )

        self.assertIn("Weekend dial-in:", message)
        self.assertIn("Test Session by A. Musician", message)
        self.assertIn("released May 1, 2026", message)
        self.assertIn("Playback is rolling now", message)
        self.assertIn("One for the liner notes:", message)
        self.assertNotIn("it still sounds timeless", message)
        self.assertNotIn("Playback is live", message)

    def test_generic_no_match_message_is_honest_and_less_mechanical(self):
        message = response_builder.compose_recommendation_message(
            topic="what is the latest album from japan",
            recommendation_plan=RecommendationPlan(
                intent="generic",
                candidates=[
                    SelectionPlan(
                        title="First Candidate",
                        artists=["First Artist"],
                        type="album",
                    )
                ],
            ),
            plan=SelectionPlan(
                title="First Candidate",
                artists=["First Artist"],
                type="album",
            ),
            selection=None,
            item_info=None,
            playback_status="skipped",
            playback_error=None,
            fun_fact=None,
            attempted_plans=[
                SelectionPlan(
                    title="First Candidate",
                    artists=["First Artist"],
                    type="album",
                )
            ],
        )

        self.assertIn("I checked 1 likely picks", message)
        self.assertIn("left playback stopped", message)
        self.assertNotIn("couldn't resolve any exact Spotify matches, so I did not start playback", message)


if __name__ == "__main__":
    unittest.main()
