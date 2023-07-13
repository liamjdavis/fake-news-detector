import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

model = joblib.load('detector.joblib')

# load article as csv
df = pd.read_csv('fake news story.csv')

# transform article
vectorizer = CountVectorizer()
text = vectorizer.fit_transform(df)

# test article
predictions = model.predict(text)
print(predictions)