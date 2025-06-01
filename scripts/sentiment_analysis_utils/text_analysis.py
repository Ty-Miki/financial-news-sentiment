import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nlp = spacy.load("en_core_web_sm")

def extract_keywords(docs: list, top_n: int = 10) -> pd.DataFrame:
    try:
        logging.info("Extracting keywords using TF-IDF")
        vectorizer = TfidfVectorizer(max_features=top_n)
        X = vectorizer.fit_transform(docs)
        keywords = vectorizer.get_feature_names_out()
        return pd.DataFrame({"keyword": keywords})
    except Exception as e:
        logging.error(f"Error in extract_keywords: {e}")
        return pd.DataFrame()

def extract_named_entities(texts: list) -> pd.DataFrame:
    try:
        logging.info("Extracting named entities")
        all_entities = []
        for doc in nlp.pipe(texts, disable=["parser", "tagger"]):
            for ent in doc.ents:
                all_entities.append((ent.text, ent.label_))
        return pd.DataFrame(all_entities, columns=["entity", "label"])
    except Exception as e:
        logging.error(f"Error in extract_named_entities: {e}")
        return pd.DataFrame()