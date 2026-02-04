import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
try:
    from .config import DATA_PATH, DATASET_DIR
except ImportError:
    from config import DATA_PATH, DATASET_DIR

TITLE_COL = "title"
TEXT_COL = "text"
LABEL_COL = "is_fake"

# Stemmer used for lightweight normalization in counting models.
stemmer = PorterStemmer()

# Ensure required NLTK resources are available without hard-failing.
def _ensure_nltk_resource(resource: str) -> bool:
    try:
        if resource == "punkt":
            word_tokenize("test")
        elif resource == "stopwords":
            stopwords.words("english")
        return True
    except LookupError:
        try:
            import nltk
            nltk.download(resource, quiet=True)
            return True
        except Exception:
            return False

_ensure_nltk_resource("punkt")
if _ensure_nltk_resource("stopwords"):
    stopwords_en = set(stopwords.words("english"))
else:
    stopwords_en = set(ENGLISH_STOP_WORDS)

# Load dataset and build a combined text body (title + text).
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[LABEL_COL] = df[LABEL_COL].astype(bool)
    df["body"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    return df

# Tokenize, normalize, and count words for the counting baselines.
def count_words(text: str) -> Counter:
    text = text.lower()
    try:
        words = word_tokenize(text)
    except LookupError:
        words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stopwords_en]
    words = [w.strip(punctuation) for w in words if len(w) >= 2]
    return Counter(words)

# Cache word-counts to avoid reprocessing on repeated runs.
def load_or_create_word_counts(df: pd.DataFrame):
    counts_path = os.path.join(DATASET_DIR, "body_counts.pkl")

    if os.path.exists(counts_path):
        with open(counts_path, "rb") as f:
            return pickle.load(f)

    body_counts = [count_words(t) for t in df["body"].astype(str)]
    with open(counts_path, "wb") as f:
        pickle.dump(body_counts, f)
    return body_counts

# Extract model-ready text features and string labels.
def get_features_and_labels(df: pd.DataFrame) -> Tuple[pd.Series, list]:
    X = df["body"].astype(str).values
    y = df[LABEL_COL].map({False: "real", True: "fake"}).values
    return X, y

# Optional downsampling and class balancing for faster experiments.
def sample_dataset(X, y, max_samples=None, random_seed=42, balance_classes=False):
    X = np.asarray(X)
    y = np.asarray(y)

    total = len(y)
    sample_size = None
    if max_samples is not None:
        if isinstance(max_samples, (float, np.floating)) and 0 < max_samples <= 1:
            sample_size = int(total * max_samples)
        else:
            sample_size = int(max_samples)
        sample_size = min(sample_size, total)

    if sample_size is None and not balance_classes:
        return X, y

    rng = np.random.RandomState(random_seed)

    if balance_classes:
        labels = np.unique(y)
        class_indices = [np.where(y == label)[0] for label in labels]
        min_class = min(len(idx) for idx in class_indices)
        if sample_size is None:
            per_class = min_class
        else:
            per_class = min(sample_size // len(labels), min_class)
        if per_class <= 0:
            return X[:0], y[:0]
        idx = np.concatenate([
            rng.choice(idx, size=per_class, replace=False) for idx in class_indices
        ])
        rng.shuffle(idx)
        return X[idx], y[idx]

    if sample_size is None:
        return X, y

    idx = rng.choice(total, size=sample_size, replace=False)
    return X[idx], y[idx]
