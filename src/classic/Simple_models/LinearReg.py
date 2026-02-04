from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# Shared TF-IDF vectorizer for the linear regression baseline.
def get_vectorizer():
    return TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        lowercase=True
    )

# Base regression model used for the thresholded classifier.
def get_linear_regression_model():
    return LinearRegression()

# Convert regression scores into hard labels.
def classify_linear_regression(y_pred, threshold=0.5):
    return (y_pred >= threshold).astype(int)

# Wrapper to use LinearRegression as a sklearn-style classifier.
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
