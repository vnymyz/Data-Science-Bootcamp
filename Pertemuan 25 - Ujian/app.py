import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# -----------------------------
# 🔧 Setup
# -----------------------------
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# -----------------------------
# 🧹 Fungsi Preprocessing
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z]', ' ', text)
    words = [w for w in text.split() if w not in stop]
    return ' '.join(words)

# -----------------------------
# 📦 Load Model dan TF-IDF
# -----------------------------
try:
    model = pickle.load(open("model_logreg.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Gagal memuat model atau TF-IDF: {e}")

# -----------------------------
# 🧠 Fungsi Prediksi
# -----------------------------
def predict_sentiment(text):
    clean = preprocess_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
    return sentiment

# -----------------------------
# 🌐 UI Streamlit
# -----------------------------
st.set_page_config(page_title="Sentiment Analysis IMDB", page_icon="🎬")
st.title("🎬 Sentiment Analysis - IMDB Movie Reviews")
st.write("Masukkan review film kamu dan lihat apakah sentimennya **positif** atau **negatif**!")

st.divider()

# Input user
user_input = st.text_area("📝 Tulis review film di sini:")

# Tombol prediksi
if st.button("🔍 Prediksi Sentimen"):
    if not model_ready:
        st.error("Model belum siap. Pastikan file model_logreg.pkl dan tfidf_vectorizer.pkl ada di folder yang sama.")
    elif user_input.strip():
        hasil = predict_sentiment(user_input)

        # Warna & emoji sesuai hasil
        if "Positive" in hasil:
            st.markdown(
                f"""
                <div style="background-color:#0f5132;padding:15px;border-radius:10px;color:white;text-align:center;">
                    <strong></strong> {hasil}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#842029;padding:15px;border-radius:10px;color:white;text-align:center;">
                    <strong></strong> {hasil}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")


# Footer kecil
st.divider()
st.caption("Made with ❤️ by Vanya & her students")
