import os
import re
import string
import pickle
import streamlit as st

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware

import nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup nltk path
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
stop_words = set(stopwords.words('english'))

# Load model & vectorizer
with open("voting_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing steps
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def remove_stopwords(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    input: Union[str, List[str]]

@app.get("/")
def read_root():
    return {"message": "Mental Illness Classification API (sklearn-based)"}

@app.post("/mic-predict")
def predict_text(payload: InputText):
    raw = payload.input
    if not isinstance(raw, str) or len(raw.strip()) == 0:
        return {"error": "Input must be a non-empty string"}

    # Step by step
    cleaned = preprocess_text(raw)
    no_stop = remove_stopwords(cleaned)
    vectorized = vectorizer.transform([no_stop])
    prediction = model.predict(vectorized)[0]

    return {"prediction": prediction}

@app.post("/mic-predict-many")
def predict_many(payload: InputText):
    raw_list = payload.input
    if not isinstance(raw_list, list) or not all(isinstance(t, str) for t in raw_list):
        return {"error": "Input must be a list of strings"}

    cleaned = [remove_stopwords(preprocess_text(t)) for t in raw_list]
    vectorized = vectorizer.transform(cleaned)
    predictions = model.predict_proba(vectorized)

    results = []
    for probs in predictions:
        results.append({
            model.classes_[i]: float(probs[i]) for i in range(len(model.classes_))
        })
    return results

st.title("Mental Illness Classification")

user_input = st.text_area("Enter text to classify:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict(user_input)
        st.success(f"Prediction: {result}")
