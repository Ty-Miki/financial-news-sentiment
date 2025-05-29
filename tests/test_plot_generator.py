import pytest
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
