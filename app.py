import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Smart Email Classifier", layout="centered")

# --- Load or Train Model ---
@st.cache_resource
def load_model():
    df = pd.read_csv("emails.csv")
    df["text"] = df["subject"] + " " + df["body"]
    X = df["text"]
    y = df["label"]

    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X, y)
    return model

model = load_model()

# --- Streamlit UI ---
st.title("Email Classifier")

subject = st.text_input("Email Subject")
body = st.text_area("Email Body")

if st.button("Classify"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter subject or body.")
    else:
        text = subject + " " + body
        prediction = model.predict([text])[0]

        st.success(f"🧠 Predicted Label: **{prediction}**")
        if prediction in ["Spam", "Promotions"]:
            st.error("❌ Suggestion: Delete")
        else:
            st.info("✅ Suggestion: Keep")

