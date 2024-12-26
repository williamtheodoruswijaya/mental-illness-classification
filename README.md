# Mental Illness Classification API

This API provides endpoints for predicting mental illness classifications based on textual input. It leverages a recurrent neural network (RNN) architecture built using the TensorFlow framework and trained on the Kaggle dataset [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health). The predictions can be utilized for applications like mental health monitoring or sentiment analysis tools.

## Overview

### Key Features
- **Single Prediction**: Classify a text input into a single mental illness category.
- **Multiple Predictions**: Obtain the probability distribution across multiple mental illness categories.
- **Scalable Deployment**: Hosted on Railway, ensuring high availability and scalability.

### Technology Stack
- **Machine Learning Framework**: TensorFlow
- **Neural Network Architecture**: Recurrent Neural Networks (RNN)
- **Dataset**: [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- **Hosting**: Railway platform

---

## API Endpoints

### Base URL
```
https://mental-illness-classification-production.up.railway.app/
```

### 1. Home Page
#### Endpoint
```
GET /
```
#### Description
Returns a simple home page indicating that the API is running.
#### Example Request
```bash
curl -X GET https://mental-illness-classification-production.up.railway.app/
```
#### Response
```
200 OK
<html><body><h1>Welcome to Mental Illness Classification API</h1></body></html>
```

---

### 2. Single Prediction
#### Endpoint
```
POST /mic-predict
```
#### Description
Predicts the mental illness category for a single text input.
#### Request Body
- **Content-Type**: `application/json`
- **Parameters**:
  - `input` (string): The text to classify.
#### Example Request
```bash
curl -X POST \
  https://mental-illness-classification-production.up.railway.app/mic-predict \
  -H "Content-Type: application/json" \
  -d '{"input": "i am feeling so depressed"}'
```
#### Response
```json
{
    "prediction": "Suicidal"
}
```

---

### 3. Multiple Predictions
#### Endpoint
```
POST /mic-predict-many
```
#### Description
Returns probabilities for each mental illness category based on the input text.
#### Request Body
- **Content-Type**: `application/json`
- **Parameters**:
  - `input` (string): The text to analyze.
#### Example Request
```bash
curl -X POST \
  https://mental-illness-classification-production.up.railway.app/mic-predict-many \
  -H "Content-Type: application/json" \
  -d '{"input": "i am feeling so depressed"}'
```
#### Response
```json
{
    "Anxiety": 0.0004915226018056273,
    "Bipolar": 7.66702214605175e-05,
    "Depression": 0.08516447246074677,
    "Normal": 0.0037606877740472555,
    "Personality disorder": 2.2527174223796465e-06,
    "Stress": 3.49273432220798e-05,
    "Suicidal": 0.9104694128036499
}
```
#### Visualization Suggestion
Use a bar chart to display the probabilities for each category.

---

## Example Workflow
### Single Prediction
Input:
```json
{"input": "i am feeling so stressed"}
```
Output:
```json
{
    "prediction": "Stress"
}
```

### Multiple Predictions
Input:
```json
{"input": "i am feeling so stressed"}
```
Output:
```json
{
    "Anxiety": 0.12,
    "Bipolar": 0.05,
    "Depression": 0.25,
    "Normal": 0.01,
    "Personality disorder": 0.02,
    "Stress": 0.50,
    "Suicidal": 0.05
}
```

---

## Error Handling
- **400 Bad Request**: Invalid or missing input.
- **500 Internal Server Error**: Unexpected server error.

### Example Error Response
```json
{
    "error": "Input text is required."
}
```

