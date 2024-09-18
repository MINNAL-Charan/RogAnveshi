from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


app = Flask(__name__)

disease_model = load_model('symptom_to_disease.keras')
disease_tokenizer = pickle.load(open('disease_tokenizer.pkl', 'rb'))
disease_le = pickle.load(open('disease_label_encoder.pkl', 'rb'))

insurance_model = pickle.load(open('insurance_model.pkl', 'rb'))
insurance_sex_encoder = pickle.load(open('insurance_sex_encoder.pkl', 'rb'))
insurance_smoker_encoder = pickle.load(open('insurance_smoker_encoder.pkl', 'rb'))
insurance_region_encoder = pickle.load(open('insurance_region_encoder.pkl', 'rb'))
insurance_scaler = pickle.load(open('insurance_scaler.pkl', 'rb'))

@app.route('/')
def index():
    print("Index page accessed")
    return render_template('index.html')

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    symptoms = request.form['symptoms']
    symptoms_seq = disease_tokenizer.texts_to_sequences([symptoms])
    symptoms_seq = pad_sequences(symptoms_seq, maxlen=100)
    prediction = disease_model.predict(symptoms_seq)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_disease = disease_le.inverse_transform([predicted_index])[0]
    print("Predicted disease:", predicted_disease)  
    try:
        return render_template('result.html', disease=predicted_disease)
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/predict_insurance', methods=['POST'])
def predict_insurance():
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    sex = insurance_sex_encoder.transform([sex])[0]
    smoker = insurance_smoker_encoder.transform([smoker])[0]
    region = insurance_region_encoder.transform([region])[0]

    input_data = np.array([
        [age, bmi, sex, children, smoker, region]
    ], dtype=np.float64)

    numerical_features = input_data[:, :2] 
    scaled_numerical_features = insurance_scaler.transform(numerical_features)
    input_data[:, :2] = scaled_numerical_features

    try:
        prediction = insurance_model.predict(input_data)
        predicted_cost = np.exp(prediction[0])
        return render_template('result.html', insurance_cost=predicted_cost)
    except Exception as e:
        return render_template('result.html', error=str(e))
if __name__ == '__main__':
    app.run(debug=True)