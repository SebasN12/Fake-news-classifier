import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('dataset\\all_news.csv')

X_text = (df['title'] + ' ' + df['text']).fillna('')
y = df['is_fake']

X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)


train_df = pd.concat([X_train.rename('text'), y_train], axis=1)
test_df = pd.concat([X_test.rename('text'), y_test], axis=1)

train_df.to_csv('dataset\\train.csv', index=False)
test_df.to_csv('dataset\\test.csv', index=False)