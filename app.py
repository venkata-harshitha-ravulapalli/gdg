import streamlit as st
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Setup
openai.api_key = "sk-your-api-key"
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

st.title("🧠 AI Mental Health Support")
st.write("Chat with an AI and track your mood.")

msg = st.text_area("How are you feeling today?", height=150)

if st.button("Send"):
    if msg:
        sentiment = sia.polarity_scores(msg)
        compound = sentiment["compound"]
        mood = "Positive 😀" if compound >= 0.5 else "Negative 😟" if compound <= -0.5 else "Neutral 😐"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": msg}]
        )

        reply = response.choices[0].message.content.strip()

        st.markdown("### 💬 AI Response")
        st.success(reply)

        st.markdown("### 📊 Detected Mood")
        st.info(f"{mood} (score: {compound:.2f})")
    else:
        st.warning("Please enter something.")
