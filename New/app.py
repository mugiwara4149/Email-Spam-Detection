# ===============================
# Import Required Libraries
# ===============================
import streamlit as st
import pickle
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse import hstack

# Download stopwords (only first time)
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ===============================
# Load Saved Models
# ===============================
model = pickle.load(open("Hybrid_Model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
prob_model = pickle.load(open("prob_model.pkl", "rb"))

# ===============================
# NLP Setup
# ===============================
ps = PorterStemmer()

# ===============================
# Text Preprocessing Function
# ===============================
def Pre(text):
    text = re.sub(r'^(s\s*u\s*b\s*j\s*e\s*c\s*t)\s*:', '', text, flags=re.I)
    text = re.sub(r'^subject\s*:', '', text, flags=re.I)
    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]

    return ' '.join(text)

# ===============================
# Manual Feature Extraction
# ===============================
def extract_features(text):
    return {
        "length": len(text),
        "num_excl": text.count("!"),
        "num_caps": sum(1 for c in text if c.isupper()),
        "num_digits": sum(1 for c in text if c.isdigit()),
        "has_free": int("free" in text.lower()),
        "has_urgent": int("urgent" in text.lower()),
        "has_win": int("win" in text.lower())
    }

# ===============================
# Streamlit UI Configuration
# ===============================
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

st.markdown(
    """
    <style>
    .st-key-Main_Page {
        border: 3px solid red;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 0px 30px rgba(255, 0, 0, 0.9);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Main App Container
# ===============================
with st.container(key="Main_Page"):

    st.title("📧 Email Spam Detection System")
    st.write("Hybrid Machine Learning Model using NLP + Feature Engineering")

    subject = st.text_input("Enter Email Subject")

    if st.button("Check Email"):

        if subject.strip() == "":
            st.warning("⚠️ Please enter an email subject.")

        else:
            # -------- Preprocessing --------
            clean_text = Pre(subject)
            text_vector = vectorizer.transform([clean_text])

            # -------- Manual Features --------
            extra_features = extract_features(subject)
            extra_df = pd.DataFrame([extra_features])
            extra_scaled = scaler.transform(extra_df)

            # -------- Combine Features --------
            final_input = hstack([text_vector, extra_scaled])

            # -------- Prediction --------
            prediction = model.predict(final_input)

            # -------- Confidence --------
            probability = prob_model.predict_proba(final_input)
            spam_confidence = probability[0][1] * 100
            ham_confidence = probability[0][0] * 100

            # -------- Result Display --------
            # After confidence calculation
            if prediction[0] == 1:
                st.error("🚨 SPAM EMAIL DETECTED")
                st.progress(int(spam_confidence))
                st.write(f"Spam Confidence: **{spam_confidence:.2f}%**")
            else:
                st.success("✅ HAM (Safe Email)")
                st.progress(int(ham_confidence))
                st.write(f"Ham Confidence: **{ham_confidence:.2f}%**")

    st.markdown("---")
    st.caption("Mini Project | Streamlit | Hybrid ML Model | NLP")
