import os
import re
import string
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware

# Load model & vectorizer
with open(os.path.join(os.path.dirname(__file__), 'voting_classifier.pkl'), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), 'vectorizer.pkl'), "rb") as f:
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

# Manual stopwords
STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at
be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too
under until up very
was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves
""".split())

label_ordered = {
    'Normal': 0,
    'Depression': 1,
    'Suicidal': 2,
    'Anxiety': 3,
    'Bipolar': 4,
    'Stress': 5,
    'Personality disorder': 6
}
reverse_label = {v: k for k, v in label_ordered.items()}

def remove_stopwords(text: str) -> str:
    tokens = re.findall(r'\b\w+\b', text.lower())
    filtered = [word for word in tokens if word not in STOP_WORDS]
    return ' '.join(filtered)

# FastAPI app
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
    cleaned = preprocess_text(raw)
    no_stop = remove_stopwords(cleaned)
    vectorized = vectorizer.transform([no_stop])
    prediction = model.predict(vectorized)[0]
    label = reverse_label.get(int(prediction), str(prediction))
    return {"prediction": label}

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
        label_probabilities = {
            label: float(probs[idx]) for label, idx in label_ordered.items()
        }
        results.append(label_probabilities)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
