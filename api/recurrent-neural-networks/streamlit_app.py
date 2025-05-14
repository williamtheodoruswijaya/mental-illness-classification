import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import re
import string
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

from nltk.data import find
try:
    find('tokenizers/punkt')
    find('corpora/stopwords')
except LookupError as e:
    raise RuntimeError("Missing NLTK data. Make sure 'punkt' and 'stopwords' are available.") from e

# Setup FastAPI app
app = FastAPI(title="Mental Illness Classification API")

# Load model
file_path = os.path.join(os.path.dirname(__file__), 'RNNModel.keras')
model = tf.keras.models.load_model(file_path)

# Load tokenizer
with open(os.path.join(os.path.dirname(__file__), 'tokenizer.pkl'), 'rb') as handle:
    tokenizer = pickle.load(handle)

label_ordered = {
    'Normal': 0,
    'Depression': 1,
    'Suicidal': 2,
    'Anxiety': 3,
    'Bipolar': 4,
    'Stress': 5,
    'Personality disorder': 6
}
reverse_labels = {v: k for k, v in label_ordered.items()}
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_label(text: str) -> str:
    text = preprocess_text(text)
    text = remove_stopwords(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    pred = model.predict(padded).argmax(axis=1)[0]
    return reverse_labels[pred]

def predict_probabilities(text: str):
    text = preprocess_text(text)
    text = remove_stopwords(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0]
    return {label: float(pred[idx]) for label, idx in label_ordered.items()}

# Request schema
class PredictRequest(BaseModel):
    input: str

# Routes
@app.post("/mic-predict")
def classify_text(req: PredictRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input must not be empty")
    
    label = predict_label(req.input)

    return {
        "prediction": label,
    }

@app.post("/mic-predict-many")
def classify_probabilities(req: PredictRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input must not be empty")
    
    probs = predict_probabilities(req.input)
    return probs
