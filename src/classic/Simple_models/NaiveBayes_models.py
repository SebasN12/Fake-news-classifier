from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# Binary count features for Bernoulli Naive Bayes.
def get_bnb_vectorizer():
    return CountVectorizer(
        lowercase=True,
        stop_words="english",
        binary=True
    )

# Bernoulli NB classifier (alpha can be tuned externally if needed).
def get_bnb_model(alpha=0.5):
    return BernoulliNB(alpha=alpha)

# TF-IDF features for Multinomial Naive Bayes.
def get_mnb_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

# Multinomial NB classifier (alpha can be tuned externally if needed).
def get_mnb_model(alpha=0.5):
    return MultinomialNB(alpha=alpha)
