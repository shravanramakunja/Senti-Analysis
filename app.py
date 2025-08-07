import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

model = joblib.load("logistic_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

emotion_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

def preprocess_text(text):
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = ''.join([char for char in text if not char.isdigit()])
    # Remove emojis and non-ascii
    text = ''.join([char for char in text if char.isascii()])
    # Remove stopwords
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)
st.title(" Emotionl Analysis")
user_input = st.text_area("Enter text for emotion analysis:")
if st.button("Prediction Emotion"):
    if user_input.strip()=="":
        st.warning("Please enter the proper sentence")
    else:
        cleaned_text=preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        emotion = emotion_map[prediction]
        st.success(f"The predicted emotion is {emotion.upper()}")