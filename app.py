import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.markdown(
    """
    <style>
        body {
            background-color: #d2b48c;  /* Earthy beige */
        }
        .stApp {
            background-color: #f5f5dc;
            padding: 2rem;
        }
        .css-18e3th9 {
            background-color: #f5f5dc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load resources
@st.cache_resource
def load_resources():
    tokenizer = joblib.load("tokenizer.pkl")
    model = tf.keras.models.load_model("pred_model.keras")
    return tokenizer, model

tokenizer, model = load_resources()

# App UI
st.title("🎬 Movie Review Sentiment Analyzer ")
st.write("Enter a movie review and see whether it's positive or negative.")

# User input
review = st.text_area("Enter your movie review here:", "")

# Prediction logic
def predict_sentiment(review):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0][0]
    sentiment = "Positive 😊" if pred > 0.5 else "Negative 😞"
    confidence = pred if pred > 0.5 else 1 - pred
    return sentiment, confidence

# Predict on button click
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment, confidence = predict_sentiment(review)
        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.info(f"Confidence: {confidence:.2%}")
