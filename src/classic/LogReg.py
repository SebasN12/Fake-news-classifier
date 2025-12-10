from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42

def get_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 1),
            max_df=0.9,
            min_df=5,
            max_features=20000
        )),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
    ])

def get_param_grid():
    return {
        "clf__C": [0.1, 1.0]
    }
