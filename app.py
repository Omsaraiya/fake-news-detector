import os
import joblib
from flask import Flask, jsonify
from dotenv import load_dotenv
from models import db, User, NewsHistory

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'ml_model', 'tfidf_vectorizer.pkl')

try:
    ml_model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Machine Learning Models loaded successfully into memory!")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML models. Error: {e}")

with app.app_context():
    db.create_all()
    print("✅ Database tables verified/created successfully!")

@app.route('/')
def home():
    return jsonify({
        "status": "Flask API Server is Running",
        "database_connected": True,
        "ai_models_loaded": "ml_model" in globals()
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)