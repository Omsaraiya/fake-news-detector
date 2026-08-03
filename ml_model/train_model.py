import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib

def build_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'news_dataset.csv')
    
    print("[1] Loading dataset using Pandas...")
    df = pd.read_csv(dataset_path)
    
    df = df.dropna()
    X = df['text']
    y = df['label']
    
    print("[2] Splitting data for training and testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[3] Vectorizing text (Converting words to numerical matrices)...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    
    print("[4] Training modern SGD Classifier (Passive-Aggressive alternative)...")
    pac = SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    model_output_path = os.path.join(base_dir, 'ml_model', 'fake_news_model.pkl')
    vectorizer_output_path = os.path.join(base_dir, 'ml_model', 'tfidf_vectorizer.pkl')
    
    print("[5] Saving ML models for web integration...")
    joblib.dump(pac, model_output_path)
    joblib.dump(tfidf_vectorizer, vectorizer_output_path)
    
    print("✅ Model training complete! .pkl files generated successfully without warnings.")

if __name__ == '__main__':
    build_model()