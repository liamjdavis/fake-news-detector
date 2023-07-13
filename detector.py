import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

model = joblib.load('detector.joblib')

# load article as csv
df = pd.read_csv('fake news story.csv')

# transform article
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df)

# test article
predictions = model.predict(X)
print(predictions)