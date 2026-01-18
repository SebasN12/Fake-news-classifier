import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from scipy.sparse import hstack
import pickle

nltk.download('punkt')
nltk.download('stopwords')

RANDOM_SEED = 42

# Split dataset into training and testing sets with stratification

df = pd.read_csv('dataset\\all_news.csv')

# Alternatively, if using the combined dataset from Kaggle: Fake News Detection Datasets. 
# Remember to run the combine_dataset.py first for this alternative.
# df = pd.read_csv('dataset/combined_fake_news.csv')

X_text = (df['title'] + ' ' + df['text']).fillna('')
y = df['is_fake']

X_subject = df[['subject']].fillna('Unknown')

X_full = pd.concat([X_text.rename('text'), X_subject], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)


train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('dataset\\train.csv', index=False)
test_df.to_csv('dataset\\test.csv', index=False)

# -------------------------------
# Text preprocessing functions
# -------------------------------

stemmer = PorterStemmer()
stopwords_en = set(stopwords.words('english'))

def stemmed_words(doc):
    return [stemmer.stem(w.lower()) for w in word_tokenize(doc) if w.lower() not in stopwords_en]

vectorizer = CountVectorizer(lowercase=True, strip_accents='unicode', tokenizer=stemmed_words)

print("Preprocessing text...")

X_train_text_vec = vectorizer.fit_transform(X_train['text'])
X_test_text_vec = vectorizer.transform(X_test['text'])

# AI for suggestion: OneHotEncoder + sparse matrix handling
subject_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_train_subject = subject_encoder.fit_transform(X_train[['subject']])
X_test_subject = subject_encoder.transform(X_test[['subject']])

X_train_final = hstack([X_train_text_vec, X_train_subject])
X_test_final = hstack([X_test_text_vec, X_test_subject])

# -------------------------------
# Save matrices and transformers with pickle
# -------------------------------
# AI for using pickle for saving

with open('dataset/X_train_final.pkl', 'wb') as file:
    pickle.dump(X_train_final, file)

with open('dataset/X_test_final.pkl', 'wb') as file:
    pickle.dump(X_test_final, file)

with open('dataset/y_train.pkl', 'wb') as file:
    pickle.dump(y_train, file)

with open('dataset/y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)

with open('dataset/vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('dataset/subject_encoder.pkl', 'wb') as file:
    pickle.dump(subject_encoder, file)
