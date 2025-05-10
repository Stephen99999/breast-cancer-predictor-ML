from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend access

model = joblib.load('model/logistic_regression_top_features.pkl')
scaler = joblib.load('model/scaler.pkl')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    data = np.array(data).reshape(1, -1)
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
