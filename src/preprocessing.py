import os
import pickle
import pandas as pd
from collections import Counter
from typing import Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation

from config import DATA_PATH, DATASET_DIR

TITLE_COL = "title"
TEXT_COL = "text"
LABEL_COL = "is_fake"

stemmer = PorterStemmer()
stopwords_en = set(stopwords.words('english'))

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[LABEL_COL] = df[LABEL_COL].astype(bool)
    df["body"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    return df

def count_words(text: str) -> Counter:
    text = text.lower()
    words = word_tokenize(text)
    words = [stemmer.stem(w) for w in words if w not in stopwords_en]
    words = [w.strip(punctuation) for w in words if len(w) >= 2]
    return Counter(words)

def load_or_create_word_counts(df: pd.DataFrame):
    counts_path = os.path.join(DATASET_DIR, "body_counts.pkl")

    if os.path.exists(counts_path):
        with open(counts_path, "rb") as f:
            return pickle.load(f)

    body_counts = [count_words(t) for t in df["body"].astype(str)]
    with open(counts_path, "wb") as f:
        pickle.dump(body_counts, f)
    return body_counts

def get_features_and_labels(df: pd.DataFrame) -> Tuple[pd.Series, list]:
    X = df["body"].astype(str).values
    y = df[LABEL_COL].map({False: "real", True: "fake"}).values
    return X, y
