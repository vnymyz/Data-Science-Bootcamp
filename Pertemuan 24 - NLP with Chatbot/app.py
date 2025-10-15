import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("airline_sentiment_logreg.joblib")

# Judul aplikasi
st.title("✈️ Airline Tweet Sentiment Classifier")
st.write("Masukkan teks tweet untuk memprediksi sentimen (negative, neutral, positive).")

# Input teks
user_input = st.text_area("Masukkan Tweet", "")

if st.button("Prediksi"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]

        st.subheader("Hasil Prediksi:")
        st.write(f"**Sentimen:** {prediction}")

        # Probabilitas dalam bentuk dict
        st.subheader("Probabilitas:")
        prob_dict = {
            "negative": round(probabilities[0], 3),
            "neutral": round(probabilities[1], 3),
            "positive": round(probabilities[2], 3),
        }
        st.write(prob_dict)

        # Visualisasi bar chart
        proba_df = pd.DataFrame([prob_dict])
        st.bar_chart(proba_df.T)
    else:
        st.warning("Harap masukkan tweet terlebih dahulu.")
