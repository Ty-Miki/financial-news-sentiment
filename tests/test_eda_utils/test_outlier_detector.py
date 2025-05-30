import pytest
import pandas as pd
from scripts.eda_utils.identify_text_outliers import TextOutlierDetector

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "headline": ["short", "a bit longer headline", "very long headline text here",
                     "tiny", "huge massive enormous headline string"],
        "headline_length": [5, 22, 29, 4, 38]
    })

def test_detect_length_outliers(sample_data):
    detector = TextOutlierDetector()
    outliers = detector.detect_length_outliers(sample_data, "headline_length", z_thresh=1.0)

    assert isinstance(outliers, pd.DataFrame)
    assert not outliers.empty
    assert "z_score" in outliers.columns
    assert all(abs(outliers["z_score"]) > 1.0)

def test_detect_length_outliers_invalid_column(sample_data):
    detector = TextOutlierDetector()
    outliers = detector.detect_length_outliers(sample_data, "invalid_column")
    assert outliers.empty

def test_detect_duplicate_texts():
    data = pd.DataFrame({
        "headline": ["Breaking news", "Breaking news", "Other news", "Different", "Different"],
        "headline_length": [13, 13, 10, 9, 9]
    })

    detector = TextOutlierDetector()
    duplicates = detector.detect_duplicate_texts(data, "headline")

    assert isinstance(duplicates, pd.DataFrame)
    assert not duplicates.empty
    assert len(duplicates) == 4

def test_detect_duplicate_texts_invalid_column(sample_data):
    detector = TextOutlierDetector()
    duplicates = detector.detect_duplicate_texts(sample_data, "missing_column")
    assert duplicates.empty
