import unittest
from datetime import date

from spotify_dj.services import release_constraints


class ReleaseConstraintTest(unittest.TestCase):
    def test_latest_request_rejects_previous_year_release(self):
        previous_year = date.today().year - 1
        self.assertFalse(
            release_constraints.passes_release_constraints(
                "what is the latest album from japan",
                {"release_date": f"{previous_year}-07-02"},
            )
        )

    def test_latest_request_accepts_current_year_release(self):
        current_year = date.today().year
        self.assertTrue(
            release_constraints.passes_release_constraints(
                "what is the latest album from japan",
                {"release_date": f"{current_year}-01-01"},
            )
        )

    def test_month_request_rejects_wrong_month(self):
        self.assertFalse(
            release_constraints.passes_release_constraints(
                "Recommend a jazz album from new york released in May 2026",
                {"release_date": "2026-02-27"},
            )
        )

    def test_month_request_accepts_matching_month(self):
        self.assertTrue(
            release_constraints.passes_release_constraints(
                "Recommend a jazz album from new york released in May 2026",
                {"release_date": "2026-05-01"},
            )
        )

    def test_date_constrained_request_rejects_missing_metadata(self):
        self.assertFalse(
            release_constraints.passes_release_constraints(
                "Recommend a jazz album from new york released in May 2026",
                None,
            )
        )

    def test_undated_request_allows_missing_metadata(self):
        self.assertTrue(
            release_constraints.passes_release_constraints(
                "Recommend a mellow jazz album",
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()
