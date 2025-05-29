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
