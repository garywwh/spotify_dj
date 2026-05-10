import unittest
from datetime import datetime, timezone

from spotify_dj.services.web_search import (
    _candidate_page_lines,
    fresh_search_queries,
    needs_fresh_data,
)


class NeedsFreshDataTest(unittest.TestCase):
    def test_current_year_needs_fresh_data(self):
        current_year = datetime.now(timezone.utc).year
        self.assertTrue(needs_fresh_data(f"Recommend a jazz album released in {current_year}"))

    def test_stale_year_alone_does_not_force_fresh_data(self):
        self.assertFalse(needs_fresh_data("Recommend a jazz album released in 2023"))

    def test_latest_needs_fresh_data(self):
        self.assertTrue(needs_fresh_data("what is the latest album from japan"))

    def test_latest_album_gets_current_year_release_queries(self):
        current_year = datetime.now(timezone.utc).year
        queries = fresh_search_queries("recommend a latest blues album from the US")

        self.assertIn("recommend a latest blues album from the US", queries)
        self.assertIn(f"{current_year} blues album releases United States", queries)
        self.assertIn(f"{current_year} latest blues albums release date Spotify", queries)

    def test_candidate_page_lines_extract_album_title_attributes(self):
        current_year = datetime.now(timezone.utc).year
        html = (
            f'<img title="Seth James - Motormouth | New Texas Blues & Roots Album {current_year}" />'
            '<img title="Generic Site Logo" />'
        )

        lines = _candidate_page_lines(html, current_year)

        self.assertEqual(lines, [f"Seth James - Motormouth | New Texas Blues & Roots Album {current_year}"])

    def test_candidate_page_lines_extract_chart_rows(self):
        current_year = datetime.now(timezone.utc).year
        html = (
            "1 LW 13 Motormouth by: Seth James Label: Qualified "
            "Sub Genre: Contemporary Blues"
        )

        lines = _candidate_page_lines(html, current_year)

        self.assertEqual(lines, [f"Motormouth by Seth James ({current_year} blues album chart)"])


if __name__ == "__main__":
    unittest.main()
