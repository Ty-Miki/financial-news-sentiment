import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlotGenerator:
    def __init__(self):
        logging.info("PlotGenerator initialized.")

    def plot_histogram(self, df: pd.DataFrame, column: str, bins: int = 20, title: str | None = None, xlabel: str | None = None, ylabel: str ="Frequency") -> None:
        """
        Displays a histogram of the specified column in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): Column name to plot histogram for.
            bins (int): Number of histogram bins.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
        """
        try:
            if column not in df.columns:
                logging.error(f"Column '{column}' not found in DataFrame.")
                return

            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], bins=bins, kde=True)
            plt.title(title or f"Histogram of {column}")
            plt.xlabel(xlabel or column)
            plt.ylabel(ylabel)

            logging.info(f"Displaying histogram for column '{column}'")
            plt.show()

        except Exception as e:
            logging.error(f"Failed to display histogram for column '{column}': {e}")

    def plot_ranked_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, top_n: int = 20):
        """
        Plots a ranked bar chart.

        Args:
            df (pd.DataFrame): DataFrame with counts.
            x_col (str): X-axis column (e.g., publisher).
            y_col (str): Y-axis column (e.g., count).
            title (str): Title of the chart.
            xlabel (str): Label for x-axis.
            ylabel (str): Label for y-axis.
            top_n (int): Number of top rows to display.
        """
        try:
            top_df = df.nlargest(top_n, y_col)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_df, x=x_col, y=y_col)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            logging.info(f"Plotted ranked bar chart for top {top_n} items by {y_col}")
        except Exception as e:
            logging.error(f"Error plotting ranked bar chart: {e}")
