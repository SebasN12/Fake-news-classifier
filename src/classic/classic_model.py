from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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
