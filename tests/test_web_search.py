import unittest
from datetime import datetime, timezone

from spotify_dj.services.web_search import needs_fresh_data


class NeedsFreshDataTest(unittest.TestCase):
    def test_current_year_needs_fresh_data(self):
        current_year = datetime.now(timezone.utc).year
        self.assertTrue(needs_fresh_data(f"Recommend a jazz album released in {current_year}"))

    def test_stale_year_alone_does_not_force_fresh_data(self):
        self.assertFalse(needs_fresh_data("Recommend a jazz album released in 2023"))

    def test_latest_needs_fresh_data(self):
        self.assertTrue(needs_fresh_data("what is the latest album from japan"))


if __name__ == "__main__":
    unittest.main()
