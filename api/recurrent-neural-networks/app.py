import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import re
import string
from string import punctuation
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
app = Flask(__name__)
CORS(app)

# step 1: load the model first
file_path = os.path.join(os.path.dirname(__file__), 'RNNModel.keras')
loaded_model = tf.keras.models.load_model(file_path)

# ini bakal ke print kalau modelnya berhasil di load
print(loaded_model.summary())

# step 2: load the tokenizer
with open(os.path.join(os.path.dirname(__file__), 'tokenizer.pkl'), 'rb') as handle:
    tokenizer = pickle.load(handle)

print(tokenizer)

# step 3: load the label encoder
label_ordered = {
    'Normal': 0,
    'Depression': 1,
    'Suicidal': 2,
    'Anxiety': 3,
    'Bipolar': 4,
    'Stress': 5,
    'Personality disorder': 6
}

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

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Mental Illness Classification API. Use the /mic-predict endpoint for predictions."

@app.route('/mic-predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # validating data
        if 'input' not in data:
            return jsonify({'error': 'Please provide the text to be analyzed.'}), 400
        
        text = data['input']

        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({'error': 'Invalid Input Data'}), 400
        
        # preprocessing text
        text = preprocess_text(text)
        text = remove_stopwords(text)
        text = tokenizer.texts_to_sequences([text])
        text = tf.keras.preprocessing.sequence.pad_sequences(text, maxlen=100, padding='post', truncating='post')

        # make prediction
        prediction = loaded_model.predict(text)
        prediction = prediction.argmax(axis=1)[0]
        prediction = [k for k, v in label_ordered.items() if v == prediction][0]

        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/mic-predict-many', methods=['POST'])
def predict_many():
    try:
        data = request.json
        
        # Validate input
        if 'input' not in data:
            return jsonify({'error': 'Please provide the text to be analyzed.'}), 400
        
        text = data['input']
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({'error': 'Input must be a non-empty string'}), 400
        
        # Preprocess text
        text = preprocess_text(text)
        text = remove_stopwords(text)
        tokenized_text = tokenizer.texts_to_sequences([text])
        padded_text = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=100, padding='post', truncating='post')
        
        # Make prediction
        prediction = loaded_model.predict(padded_text)[0]  # Get the first (and only) prediction
        
        # Map probabilities to labels
        label_probabilities = {label: float(prediction[index]) for label, index in label_ordered.items()}
        
        return jsonify(label_probabilities), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

