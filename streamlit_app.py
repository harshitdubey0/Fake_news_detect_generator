"""STREAMLIT WEB INTERFACE FOR FAKE NEWS DETECTION"""

import streamlit as st
import pandas as pd
import numpy as np
from fake_news_ml import FakeNewsDetectionModel
import os

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown("""
<style>
    .header { text-align: center; color: #1f472a; }
    .fake { background-color: #ffcccc; padding: 10px; border-radius: 5px; }
    .real { background-color: #ccffcc; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Fake News Detection System")
st.markdown("AI-powered system to detect fake news with 91.8% accuracy")

# Load model
@st.cache_resource
def load_model():
    model = FakeNewsDetectionModel()
    if model.load_models():
        return model
    return None

model = load_model()

if model is None:
    st.error("❌ Models not found. Please train the model first: `python train_and_evaluate.py`")
else:
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📰 Single Article", "📊 Batch Analysis", "ℹ️ About"])

    with tab1:
        st.subheader("Analyze a Single Article")
        article_text = st.text_area("Paste your article text here:", height=250)

        if st.button("Analyze", key="analyze_single"):
            if article_text:
                pred, confidence = model.predict(article_text)

                col1, col2 = st.columns(2)
                with col1:
                    if pred == 1:
                        st.markdown(f"<div class='fake'><h3>⚠️ LIKELY FAKE</h3><p>Confidence: {confidence*100:.1f}%</p></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='real'><h3>✅ LIKELY REAL</h3><p>Confidence: {confidence*100:.1f}%</p></div>", unsafe_allow_html=True)

                with col2:
                    st.metric("Confidence Score", f"{confidence*100:.1f}%")
            else:
                st.warning("Please paste an article to analyze")

    with tab2:
        st.subheader("Batch Analysis (CSV Upload)")
        uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=['csv'])

        if uploaded_file and st.button("Analyze Batch"):
            df = pd.read_csv(uploaded_file)
            results = []

            progress_bar = st.progress(0)
            for idx, row in df.iterrows():
                text = row.get('text', '')
                if text:
                    pred, conf = model.predict(text)
                    results.append({'text': text[:100], 'prediction': 'FAKE' if pred == 1 else 'REAL', 'confidence': conf})
                progress_bar.progress((idx+1)/len(df))

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)
            st.download_button("Download Results", csv, "results.csv")

    with tab3:
        st.subheader("About This System")
        st.info("""
        **Model Accuracy:** 91.8%
        **Algorithms:** 5 (Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost)
        **Ensemble:** Soft Voting Classifier
        **Features:** TF-IDF with 5000 features and bigrams
        """)
