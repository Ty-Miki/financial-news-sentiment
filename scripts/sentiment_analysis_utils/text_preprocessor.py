import logging
import re
import spacy

import nltk
nltk.download('stopwords', quiet=True)  # Ensure stopwords are downloaded

from nltk.corpus import stopwords
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> List[str]:
    try:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words and not token.is_punct and not token.is_space]
        return tokens
    except Exception as e:
        return []
    
