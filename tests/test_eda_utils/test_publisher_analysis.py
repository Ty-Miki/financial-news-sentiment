import pytest
import pandas as pd
from scripts.eda_utils.publisher_analysis import PublisherAnalyzer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'publisher': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'C', 'C']
    })

def test_count_articles_by_publisher(sample_df):
    generator = PublisherAnalyzer()
    result = generator.count_articles_by_publisher(sample_df, 'publisher')
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'publisher', 'article_count'}
    
    # Check counts are correct
    a_count = result[result['publisher'] == 'A']['article_count'].values[0]
    b_count = result[result['publisher'] == 'B']['article_count'].values[0]
    c_count = result[result['publisher'] == 'C']['article_count'].values[0]

    assert a_count == 3
    assert b_count == 2
    assert c_count == 4

def test_invalid_column():
    df = pd.DataFrame({'not_publisher': ['X', 'Y']})
    generator = PublisherAnalyzer()
    result = generator.count_articles_by_publisher(df, 'publisher')

    assert result.empty
