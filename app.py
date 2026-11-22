from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)
MODEL_PATH = 'superkart_model.joblib'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = joblib.load(os.path.join('backend_files', MODEL_PATH))

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "SuperKart Sales Prediction API is up and running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            data = [data]
        input_df = pd.DataFrame(data)
        predictions = model.predict(input_df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)