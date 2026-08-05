import os
import math
import joblib
from flask import Flask, jsonify, request, render_template
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_news():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    news_text = data['text']

    features = tfidf_vectorizer.transform([news_text])
    prediction_label = ml_model.predict(features)[0]
    
    distance = ml_model.decision_function(features)[0]
    real_probability = 1 / (1 + math.exp(-distance))
    
    real_percentage = round(real_probability * 100, 1)
    fake_percentage = round((1 - real_probability) * 100, 1)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    user_text_indices = features.nonzero()[1]
    
    word_weights = []
    for idx in user_text_indices:
        word = feature_names[idx]
        weight = ml_model.coef_[0][idx]
        word_weights.append({'word': word, 'weight': weight})
        
    word_weights.sort(key=lambda x: abs(x['weight']), reverse=True)
    top_triggers = [w['word'] for w in word_weights[:3]]

    return jsonify({
        "prediction": prediction_label,
        "percentages": {
            "real": real_percentage,
            "fake": fake_percentage
        },
        "trigger_words": top_triggers
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)