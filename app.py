import pandas as pd
import numpy as np
import joblib  # To save/load the model
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import time  # Import time for delay

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# PostgreSQL database configuration
db_config = {
    'host': 'localhost',
    'database': 'postgres1',
    'user': 'postgres',
    'password': 'root',
    'port': 5432
}

def get_db_connection():
    return psycopg2.connect(**db_config)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS properties (
            id SERIAL PRIMARY KEY,
            city TEXT NOT NULL,
            pincode TEXT NOT NULL,
            survey TEXT NOT NULL,
            price REAL NOT NULL,
            area REAL NOT NULL,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
        );
    ''')
    conn.commit()
    cursor.close()
    conn.close()

# Initialize the database
init_db()

# Load trained model and feature columns
model = joblib.load('xgb_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Home Page after login
@app.route('/home')
def home():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.args.get('logout'):
        flash("Logged out successfully!", "success")

    if request.method == 'GET':
        session.pop('_flashes', None)  # Clear previous flash messages

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user or not check_password_hash(user[1], password):
            time.sleep(1)  # Slow down brute force attacks
            flash("Invalid username or password. Try again.", "error")
            return render_template('login.html')

        session['logged_in'] = True
        session['user_id'] = user[0]
        session['username'] = username
        return redirect(url_for('home'))

    return render_template('login.html')



# Prediction Page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        area_type = request.form['area_type']
        location = request.form['location']
        size = float(request.form['size'])
        total_sqft = float(request.form['total_sqft'])
        bath = float(request.form['bath'])
        balcony = float(request.form['balcony'])

        input_data = pd.DataFrame([[area_type, location, size, total_sqft, bath, balcony]], 
                                  columns=['area_type', 'location', 'size', 'total_sqft', 'bath', 'balcony'])
        input_data = pd.get_dummies(input_data, columns=['area_type', 'location'])
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        predicted_price = model.predict(input_data)[0]

        return render_template('prediction.html', prediction=round(predicted_price, 2))

    return render_template('prediction.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password or len(password) < 8:
            flash('Username and password (min 8 characters) are required!', 'error')
            return redirect(url_for('register'))

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username already exists! Please choose another one.', 'error')
            cursor.close()
            conn.close()
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            cursor.execute('INSERT INTO users (username, password_hash) VALUES (%s, %s)', (username, hashed_password))
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except psycopg2.Error as e:
            flash(f"Database error: {e}", 'error')
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')

# User Form Page
@app.route('/form', methods=['GET', 'POST'])
def form():
    if 'user_id' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))

    if request.method == 'POST':
        city = request.form['city']
        pincode = request.form['pincode']
        survey = request.form['survey']
        price = float(request.form['price'])
        area = float(request.form['area'])
        user_id = session['user_id']

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO properties (city, pincode, survey, price, area, user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (city, pincode, survey, price, area, user_id))
            conn.commit()
            flash('Form submitted successfully!', 'success')
            return redirect(url_for('confirmation'))
        except psycopg2.Error as e:
            flash(f"Database error: {e}", 'error')
        finally:
            cursor.close()
            conn.close()

    return render_template('form.html')

# Confirmation Page
@app.route('/confirmation')
def confirmation():
    return render_template('confirmation.html')

# Admin Panel
@app.route('/admin')
def admin():
    if 'user_id' not in session or session.get('username') != 'admin':
        flash("Access denied! Admins only.", "error")
        return redirect(url_for('home'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT properties.id, city, pincode, survey, price, area, users.username
        FROM properties
        JOIN users ON properties.user_id = users.id
    ''')
    properties = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('admin.html', properties=properties)

# Logout
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for('login')) 
# Handle 404 Page Not Found error
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
