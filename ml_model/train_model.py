import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'news_dataset.csv')

print("[1] Loading dataset...")
df = pd.read_csv(DATASET_PATH).dropna()

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[2] Vectorizing text with strict English stop-words filtering...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

print("[3] Training Passive-Aggressive Classifier (Industry Standard for Fake News)...")
model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
model.fit(X_train_tfidf, y_train)

print("[4] Saving models...")
joblib.dump(model, os.path.join(BASE_DIR, 'fake_news_model.pkl'))
joblib.dump(tfidf_vectorizer, os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))

print("✅ PAC Model trained and saved successfully!")