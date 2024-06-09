import joblib
import logging
import pandas as pd

from flask import Flask, request, render_template, json
from transformers import pipeline
from utils import *


app = Flask(__name__)
punct_corrector = pipeline("token-classification", model="markusiko/rubert-base-punctuation")

@app.route('/', methods=['Get', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['Get', 'POST'])
def predict():
    try:
        upload_file = request.files['text']
        df = pd.read_csv(upload_file)
        df["corrected_text"] = df["raw_text"].apply(lambda text: get_corrected_sentence(punct_corrector, text))
        return df.to_json(orient="split")
                                            
        
    except Exception as e:
        return json.dumps(f"Error: {e}")
    
if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)