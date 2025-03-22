import streamlit as st
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Get API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Download VADER sentiment lexicon
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Set up Streamlit app
st.set_page_config(page_title="Mental Health AI Companion", page_icon="🧠")
st.title("🧠 Mental Health AI Companion")
st.write("This is a private, stigma-free space. Talk to the AI about how you're feeling.")

# User input
user_input = st.text_area("🗣️ What's on your mind today?", height=150)

if st.button("Send"):
    if not user_input.strip():
        st.warning("Please type something first.")
    else:
        sentiment = sia.polarity_scores(user_input)
        score = sentiment["compound"]
        if score >= 0.5:
            mood = "😊 Positive"
        elif score <= -0.5:
            mood = "😟 Negative"
        else:
            mood = "😐 Neutral"

        try:
            # OpenAI ChatCompletion call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}]
            )
            ai_reply = response.choices[0].message.content.strip()

            st.markdown("### 💬 AI Response")
            st.success(ai_reply)

            st.markdown("### 🧭 Mood Detection")
            st.info(f"**{mood}** (Sentiment Score: `{score}`)")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
