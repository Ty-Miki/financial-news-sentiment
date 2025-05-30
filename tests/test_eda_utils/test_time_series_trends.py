import pytest
import pandas as pd
from scripts.eda_utils.time_series_trends import TimeSeriesTrendsAnalyzer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'published_date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'headline': [f"Headline {i}" for i in range(10)]
    })

def test_get_publication_counts_daily(sample_data):
    analyzer = TimeSeriesTrendsAnalyzer()
    result = analyzer.get_publication_counts(sample_data, 'published_date', freq='D')
    assert not result.empty
    assert 'article_count' in result.columns