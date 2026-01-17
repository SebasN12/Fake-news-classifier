from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

stemmer = PorterStemmer()
stopwords_en = set(stopwords.words('english'))

def is_fake1_from_counts(counter: Counter, top_fake: str, top_real: str) -> str:
    return "fake" if counter[top_fake] > counter[top_real] else "real"

def is_fake2_from_counts(counter: Counter,
                         fake_train_counts: Counter,
                         real_train_counts: Counter) -> str:
    return "fake" if (counter & fake_train_counts).total() > (counter & real_train_counts).total() else "real"

def stemmed_words(text):
    return (stemmer.stem(w) for w in word_tokenize(text) if w not in stopwords_en)

def get_vectorizer():
    return CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        tokenizer=stemmed_words
    )

def get_linear_regression_model():
    return LinearRegression()

def classify_linear_regression(y_pred, threshold=0.5):
    return (y_pred >= threshold).astype(int)

class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model_ = None
        self._return_labels = False
        self._pos_label = None
        self._neg_label = None

    def fit(self, X, y):
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in {"U", "S", "O"}:
            self._pos_label = "fake"
            self._neg_label = "real"
            y_num = (y_arr == self._pos_label).astype(int)
            self._return_labels = True
        else:
            y_num = y_arr.astype(float)
            self._return_labels = False

        self.model_ = LinearRegression()
        self.model_.fit(X, y_num)
        return self

    def predict(self, X):
        scores = self.model_.predict(X)
        preds = (scores >= self.threshold).astype(int)
        if self._return_labels:
            return np.where(preds == 1, self._pos_label, self._neg_label)
        return preds
