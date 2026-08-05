import os
import joblib
from flask import Flask, jsonify, request, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
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
except Exception as e:
    print(f"Warning: Could not load ML models. Error: {e}")

with app.app_context():
    db.create_all()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        else:
            flash('Login Failed. Check your email and password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    records = NewsHistory.query.filter_by(user_id=session['user_id']).order_by(NewsHistory.created_at.desc()).all()
    return render_template('history.html', records=records)

@app.route('/admin')
def admin_dashboard():
    total_scans = NewsHistory.query.count()
    fake_count = NewsHistory.query.filter_by(prediction='FAKE').count()
    real_count = NewsHistory.query.filter_by(prediction='REAL').count()
    recent_records = NewsHistory.query.order_by(NewsHistory.created_at.desc()).limit(10).all()
    
    return render_template('admin.html', total=total_scans, fake=fake_count, real=real_count, records=recent_records)

@app.route('/predict', methods=['POST'])
def predict_news():
    import math # Ensure math is imported at the top of your file
    
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    news_text = data['text']
    features = tfidf_vectorizer.transform([news_text])
    
    # 1. Get Prediction
    prediction_label = ml_model.predict(features)[0]
    
    # 2. Calculate Exact Probabilities using a Sigmoid curve on the PAC Decision Function
    distance = ml_model.decision_function(features)[0]
    real_probability = 1 / (1 + math.exp(-distance))
    
    real_percentage = round(real_probability * 100, 1)
    fake_percentage = round((1 - real_probability) * 100, 1)

    # 3. Keyword Extraction
    feature_names = tfidf_vectorizer.get_feature_names_out()
    user_text_indices = features.nonzero()[1]
    
    word_weights = []
    for idx in user_text_indices:
        word = feature_names[idx]
        weight = ml_model.coef_[0][idx]
        word_weights.append({'word': word, 'weight': weight})
        
    word_weights.sort(key=lambda x: abs(x['weight']), reverse=True)
    top_triggers = [w['word'] for w in word_weights[:3]]

    # 4. Save to Database
    try:
        history_record = NewsHistory(
            user_id=session['user_id'],
            news_text=news_text,
            prediction=prediction_label,
            confidence=fake_percentage if prediction_label == 'FAKE' else real_percentage
        )
        db.session.add(history_record)
        db.session.commit()
    except Exception as e:
        print(e)

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