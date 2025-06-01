import logging
import pandas as pd
from typing import List
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_lda_topic_modeling(texts: List[str], num_topics: int = 5) -> pd.DataFrame:
    """
    Apply LDA topic modeling to a list of texts using scikit-learn.

    Args:
        texts (List[str]): List of documents.
        num_topics (int): Number of topics to extract.

    Returns:
        pd.DataFrame: DataFrame with the dominant topic for each document.
    """
    try:
        logging.info("Applying LDA topic modeling with scikit-learn")

        # Vectorize the text
        vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
        doc_term_matrix = vectorizer.fit_transform(texts)

        # Fit LDA model
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        doc_topic_dist = lda_model.fit_transform(doc_term_matrix)

        # Get dominant topic for each document
        dominant_topics = doc_topic_dist.argmax(axis=1)

        return pd.DataFrame({"dominant_topic": dominant_topics})

    except Exception as e:
        logging.error(f"Error in apply_lda_topic_modeling: {e}")
        return pd.DataFrame()
