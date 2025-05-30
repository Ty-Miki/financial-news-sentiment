import pytest
import pandas as pd
from scripts.eda_utils.extract_publisher_emails import PublisherEmailExtractor

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'publisher': [
            'nytimes.com', 
            'news@finance.com', 
            'reuters.com', 
            'editor@marketwatch.com', 
            'cnbc.com', 
            'info@investingnews.org',
            'news@finance.com'
        ]
    })

def test_extract_email_publishers(sample_df):
    extractor = PublisherEmailExtractor()
    result = extractor.extract_publisher_emails(sample_df, 'publisher')
    
    assert isinstance(result, pd.DataFrame)
    assert 'publisher_email' in result.columns
    assert 'count' in result.columns
    assert len(result) == 3
    assert result.loc[result['publisher_email'] == 'news@finance.com', 'count'].values[0] == 2
