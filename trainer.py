import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib

df = pd.read_csv('news.csv')
labels = df.label

articleText = df['text']

# text feature extraction with bag of words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(articleText)
featureNames = vectorizer.get_feature_names_out()

# reduce dimensions
svd = TruncatedSVD(n_components = 44)
svd.fit(X)
reduced_X = svd.transform(X)

# split into training and testing
x_train,x_test,y_train,y_test=train_test_split(reduced_X, labels, test_size=0.2, random_state=7)

# logistic regression
model = LogisticRegression(max_iter=15000)
model.fit(reduced_X, labels)

# save model
joblib.dump(model, 'detector.joblib')

# evaluate model
# accuracy = accuracy_score(y_true=y_test, y_pred=model.predict(x_test))
# print(accuracy)