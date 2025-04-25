import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import re
import string
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st

# Load model
model = tf.keras.models.load_model('RNNModel.keras')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
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

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_label(text):
    text = preprocess_text(text)
    text = remove_stopwords(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    pred = model.predict(padded).argmax(axis=1)[0]
    return reverse_labels[pred]

def predict_probabilities(text):
    text = preprocess_text(text)
    text = remove_stopwords(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0]
    return {label: float(pred[idx]) for label, idx in label_ordered.items()}

# Streamlit UI
st.title("Mental Illness Classification")
input_text = st.text_area("Enter text to analyze:")

if st.button("Predict"):
    if input_text.strip():
        label = predict_label(input_text)
        probs = predict_probabilities(input_text)
        st.write(f"### Prediction: **{label}**")
        st.write("### Class probabilities:")
        st.json(probs)
    else:
        st.warning("Please enter some text.")