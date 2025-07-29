import streamlit as st
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Define job description
job_description = """
We are looking for a Python developer with experience in data analysis, machine learning, and NLP.
The ideal candidate should be proficient in Python, pandas, scikit-learn, and have good communication skills.
"""

# Streamlit app UI
st.set_page_config(page_title="AI Resume Ranker", layout="centered")
st.title("📄 AI Resume Ranker")
st.markdown("Upload multiple `.txt` resumes and see how well they match a job description.")

uploaded_files = st.file_uploader("📤 Upload Resume Files", type="txt", accept_multiple_files=True)

if uploaded_files:
    documents = []
    file_names = []

    # Read and preprocess resumes
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        if text.strip() == "":
            st.warning(f"⚠️ {file.name} is empty.")
            continue
        cleaned = preprocess(text)
        documents.append(cleaned)
        file_names.append(file.name)

    if len(documents) == 0:
        st.error("❌ No valid resumes to process.")
    else:
        # Preprocess job description and add to documents
        documents.insert(0, preprocess(job_description))

        # TF-IDF + cosine similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)
        scores = cosine_similarity(vectors[0:1], vectors[1:])[0]

        # Create DataFrame for results
        result_df = pd.DataFrame({
            "Resume": file_names,
            "Match Score": [round(s * 100, 2) for s in scores]
        }).sort_values(by="Match Score", ascending=False)

        st.subheader("📊 Ranked Resumes")
        st.dataframe(result_df.reset_index(drop=True))