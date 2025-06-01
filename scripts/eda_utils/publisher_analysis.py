import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PublisherAnalyzer:

    def __init__(self):
        logging.info("PublisherStatsGenerator initialized.")

    def count_articles_by_publisher(self, df: pd.DataFrame, publisher_col: str) -> pd.DataFrame:
        try:
            if publisher_col not in df.columns:
                raise ValueError(f"Column '{publisher_col}' not found in DataFrame")

            counts = df[publisher_col].value_counts().reset_index()
            counts.columns = [publisher_col, 'article_count']

            logging.info(f"Counted {len(counts)} unique publishers.")
            return counts

        except Exception as e:
            logging.error(f"Error counting articles by publisher: {e}")
            return pd.DataFrame()
