import streamlit as st
import joblib
import os
import numpy as np
from news_api import fetch_real_news

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
API_KEY = st.secrets["NEWS_API_KEY"]

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

# ------------------ UI ------------------
st.title("📰 News Authenticity Detector (API + ML)")
st.write("This system combines Machine Learning with real-time news verification.")

user_input = st.text_area("Enter a news article or headline:")

# ------------------ PREDICTION ------------------
if st.button("Check Authenticity"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # -------- ML Prediction --------
        X_user = vectorizer.transform([user_input])
        ml_pred = model.predict(X_user)[0]
        ml_probs = model.predict_proba(X_user)[0]
        ml_confidence = ml_probs[ml_pred]

        # -------- API Similarity --------
        related_articles = fetch_real_news(user_input)
        similarity_score = 0.0

        if related_articles:
            X_related = vectorizer.transform(related_articles)
            similarity_score = np.mean((X_user @ X_related.T).toarray())

        # -------- FINAL DECISION --------
        if ml_pred == 1:
            st.success("✅ Likely REAL")
        else:
            st.error("❌ Likely FAKE")

        if similarity_score > 0.02:
            st.info("📰 Similar content found in trusted news sources")
        else:
            st.info("⚠️ No strong match found in trusted news sources")

        # -------- 🔎 DEBUG INFO (ADD HERE) --------
        st.markdown("### 🔎 Debug Info")
        st.write("ML Prediction (0 = Fake, 1 = Real):", ml_pred)
        st.write("ML Confidence:", round(ml_confidence, 3))
        st.write("Similarity Score:", round(similarity_score, 4))
