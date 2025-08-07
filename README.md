# Emotion Analysis

A machine learning project that analyzes text and predicts emotions using natural language processing techniques.

## Overview

This project implements an emotion classification system that can identify six different emotions from text input:
- Sadness
- Anger
- Love
- Surprise
- Fear
- Joy

The system uses a trained logistic regression model with TF-IDF vectorization to analyze and classify emotions in text.

## Features

- Text preprocessing including stopword removal, punctuation cleaning, and normalization
- Real-time emotion prediction through a web interface
- Support for six emotion categories
- Clean and intuitive user interface

## Project Structure

```
├── app.py                    # Streamlit web application
├── Model.ipynb              # Jupyter notebook for model training and analysis
├── train.txt                # Training dataset
├── logistic_model.joblib    # Trained logistic regression model
├── tfidf_vectorizer.joblib  # TF-IDF vectorizer
└── requirements.txt         # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shravanramakunja/Senti-Analysis.git
cd Senti-Analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

Start the Streamlit web application:
```bash
streamlit run app.py
```

Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`).

### Using the Application

1. Enter your text in the text area
2. Click the "Prediction Emotion" button
3. View the predicted emotion result

### Model Training

The model training process is documented in the `Model.ipynb` notebook, which includes:
- Data loading and preprocessing
- Feature extraction using TF-IDF
- Model training and evaluation
- Model persistence

## Dependencies

- joblib: Model serialization
- scikit-learn: Machine learning algorithms
- streamlit: Web application framework
- nltk: Natural language processing toolkit

## Technical Details

### Text Preprocessing

The text preprocessing pipeline includes:
- Converting to lowercase
- Removing punctuation and digits
- Filtering non-ASCII characters
- Removing English stopwords
- Tokenization and cleaning

### Model Architecture

- Algorithm: Logistic Regression
- Feature Extraction: TF-IDF Vectorization
- Classes: 6 emotion categories (indexed 0-5)


