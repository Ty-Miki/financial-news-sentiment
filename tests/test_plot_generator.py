import pytest
from unittest.mock import patch
import pandas as pd
from scripts.plot_generator import PlotGenerator
import matplotlib.pyplot as plt

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "headline": ["Meta launches new AI", "Apple stock rises", "Tesla earnings report"],
        "headline_length": [21, 18, 22]
    })

def test_plot_histogram_valid_column(monkeypatch, sample_df):
    plotter = PlotGenerator()

    # Monkeypatch plt.show to avoid opening a window during tests
    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        plotter.plot_histogram(
            df=sample_df,
            column="headline_length",
            bins=5,
            title="Test Histogram",
            xlabel="Length"
        )
    except Exception as e:
        pytest.fail(f"plot_histogram raised an exception: {e}")

@patch("matplotlib.pyplot.show")
def test_plot_histogram_invalid_column(monkeypatch, sample_df, caplog):
    plotter = PlotGenerator()

    # Monkeypatch plt.show
    monkeypatch.setattr(plt, "show", lambda: None)

    plotter.plot_histogram(
        df=sample_df,
        column="nonexistent_column",
        bins=5
    )

    assert "Column 'nonexistent_column' not found in DataFrame." in caplog.text

@pytest.fixture
def publisher_counts_df():
    return pd.DataFrame({
        'publisher': ['A', 'B', 'C', 'D'],
        'article_count': [100, 150, 120, 90]
    })

@patch("matplotlib.pyplot.show")
def test_plot_ranked_bar_chart(publisher_counts_df):
    plotter = PlotGenerator()
    
    try:
        plotter.plot_ranked_bar_chart(
            df=publisher_counts_df,
            x_col='publisher',
            y_col='article_count',
            title='Test Plot',
            xlabel='Publisher',
            ylabel='Articles',
            top_n=3
        )
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")

@pytest.fixture
def ts_data():
    return pd.DataFrame({
        'published_date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'article_count': [5, 7, 6, 4, 8, 10, 3, 6, 7, 9]
    })

@patch("matplotlib.pyplot.show")
def test_plot_time_series(ts_data):
    plotter = PlotGenerator()

    try:
        plotter.plot_time_series(ts_data, 'published_date', 'article_count', title="Test Time Series")
    except Exception as e:
        pytest.fail(f"plot_time_series raised an exception: {e}")
