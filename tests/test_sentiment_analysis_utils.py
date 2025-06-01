from scripts.sentiment_analysis_utils.text_preprocessor import preprocess_text
from scripts.sentiment_analysis_utils.text_analysis import extract_keywords, extract_named_entities
from scripts.sentiment_analysis_utils.text_modeler import apply_lda_topic_modeling

def test_preprocess_text():
    result = preprocess_text("Apple is looking at buying U.K. startup for $1 billion!")
    assert isinstance(result, list)
    assert all(isinstance(token, str) for token in result)


def test_extract_keywords():
    docs = ["stock market crash leads to massive losses", "tech companies show strong earnings"]
    result = extract_keywords(docs, top_n=5)
    assert not result.empty


def test_extract_named_entities():
    texts = ["Apple Inc. is planning to buy a startup in the UK"]
    result = extract_named_entities(texts)
    assert not result.empty
    assert "entity" in result.columns


def test_apply_lda_topic_modeling_runs():
    texts = [
        "Stock prices fell sharply after earnings report",
        "Interest rates increased by the central bank",
        "Earnings report leads to stock rally",
        "Central bank cuts interest rates",
        "New financial regulations impact the markets"
    ]
    result = apply_lda_topic_modeling(texts, num_topics=3)
    assert isinstance(result, object)
    assert "dominant_topic" in result.columns
    assert len(result) == len(texts)