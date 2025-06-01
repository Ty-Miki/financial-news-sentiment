import unittest
import pandas as pd
from scripts.eda_utils.headline_stats import analyze_headline_lengths

class TestAnalyzeHeadlineLengths(unittest.TestCase):

    def setUp(self):
        self.valid_df = pd.DataFrame({
            "headline": [
                "Apple stock hits record high",
                "Tesla sees surge in vehicle deliveries",
                "Amazon reports strong Q3 earnings",
                "Meta unveils new VR headset"
            ]
        })

        self.empty_df = pd.DataFrame()
        self.missing_column_df = pd.DataFrame({
            "title": ["Sample headline"]
        })

    def test_valid_headlines(self):
        stats = analyze_headline_lengths(self.valid_df)
        self.assertIsInstance(stats, dict)
        self.assertIn("mean", stats)
        self.assertGreater(stats["count"], 0)

    def test_missing_column(self):
        stats = analyze_headline_lengths(self.missing_column_df)
        self.assertEqual(stats, {})  # Should return empty dict on error

    def test_empty_dataframe(self):
        stats = analyze_headline_lengths(self.empty_df)
        self.assertEqual(stats, {})  # Should return empty dict on error

if __name__ == '__main__':
    unittest.main()
