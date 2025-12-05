from pathlib import Path
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

df_train = pd.read_csv('dataset\\train.csv').fillna('')
df_test = pd.read_csv('dataset\\test.csv').fillna('')

X_train = df_train['text']
y_train = df_train['is_fake']
X_test = df_test['text']
y_test = df_test['is_fake']

stemmer = PorterStemmer()
stopwords_en = set(stopwords.words('english'))

def stemmed_words(doc):
    return [stemmer.stem(w.lower()) for w in word_tokenize(doc) if w.lower() not in stopwords_en]

vectorizer = CountVectorizer(lowercase=True, strip_accents='unicode', tokenizer=stemmed_words)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm = SVC()
svm.fit(X_train_vec, y_train)

y_pred = svm.predict(X_test_vec)

# Evaluation outputs

Path('metrics').mkdir(exist_ok=True)

labels = ['Real', 'Fake']
report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)

with open('metrics\\svm_classification_report.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"MCC: {matthews_corrcoef(y_test, y_pred)}\n\n")
    f.write(report)


cm = confusion_matrix(y_test, y_pred)
s = sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', xticklabels=labels, yticklabels=labels)
s.set_xlabel('Predicted')
s.set_ylabel('Actual')
s.set_title('Confusion matrix')
s.figure.tight_layout()
s.figure.savefig('metrics\\svm_confusion_matrix.png', dpi=300)