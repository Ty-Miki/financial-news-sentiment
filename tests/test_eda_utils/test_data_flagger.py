import pytest
import pandas as pd
import numpy as np
from scripts.eda_utils.data_flagger import DataFlagger

@pytest.fixture
def test_df():
    return pd.DataFrame({
        "headline": ["news one", "news two", "news three", "news four"],
        "headline_length": [10, 12, 15, 8]
    }, index=[100, 101, 102, 103])  # custom index for testing

def test_flag_rows_valid(test_df):
    flagger = DataFlagger()
    modified_df = flagger.flag_rows(test_df.copy(), [100, 103], "is_outlier")

    assert "is_outlier" in modified_df.columns
    assert modified_df.loc[100, "is_outlier"] is np.True_
    assert modified_df.loc[103, "is_outlier"] is np.True_
    assert modified_df.loc[101, "is_outlier"] is np.False_

def test_flag_rows_empty_indices(test_df):
    flagger = DataFlagger()
    modified_df = flagger.flag_rows(test_df.copy(), [], "is_duplicate")

    assert "is_duplicate" in modified_df.columns
    assert modified_df["is_duplicate"].sum() == 0  # No rows flagged

def test_flag_rows_invalid_index(test_df):
    flagger = DataFlagger()
    # Intentionally pass a non-existent index
    modified_df = flagger.flag_rows(test_df.copy(), [999], "is_outlier")

    # Should raise no error, but won't mark anything
    assert modified_df["is_outlier"].sum() == 0
