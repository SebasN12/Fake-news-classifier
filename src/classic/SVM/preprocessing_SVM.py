import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from scipy.sparse import hstack
import pickle

RANDOM_SEED = 42

# Change these variables if using other dataset
isOtherDataset = False
datasetPath = 'dataset/all_news.csv'



stemmer = PorterStemmer()
stopwords_en = set(stopwords.words('english'))

def stemmed_words(doc):
        return [stemmer.stem(w.lower()) for w in word_tokenize(doc) if w.lower() not in stopwords_en]


# --------------------------------
# Preprocessing function
# --------------------------------

def run_preprocessing():
    nltk.download('punkt')
    nltk.download('stopwords')

    df = pd.read_csv(datasetPath)

    X_text = (df['title'] + ' ' + df['text']).fillna('')
    y = df['is_fake']

    X_full = pd.concat([X_text.rename('text')], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)


    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    if isOtherDataset:
        train_df.to_csv('dataset\\other_train.csv', index=False)
        test_df.to_csv('dataset\\other_test.csv', index=False)
    else:
        train_df.to_csv('dataset\\train.csv', index=False)
        test_df.to_csv('dataset\\test.csv', index=False)

    # -------------------------------
    # Text preprocessing functions
    # -------------------------------

    vectorizer = CountVectorizer(lowercase=True, strip_accents='unicode', tokenizer=stemmed_words)

    print("Preprocessing text...")

    X_train_text_vec = vectorizer.fit_transform(X_train['text'])
    X_test_text_vec = vectorizer.transform(X_test['text'])

    X_train_final = hstack([X_train_text_vec])
    X_test_final = hstack([X_test_text_vec])

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

    print("Preprocessing complete. Data saved.")

if __name__ == "__main__":
    run_preprocessing()