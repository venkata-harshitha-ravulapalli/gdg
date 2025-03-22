import streamlit as st
from openai import OpenAI
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# 🔐 Option 1: For local testing — paste your key here directly
# openai_api_key = "sk-your-real-api-key"

# 🔐 Option 2: For Streamlit Cloud (securely store in Settings > Secrets)
openai_api_key = st.secrets["AIzaSyBZZ6JSwO7V6dH2I6qqLUH8_v9OiQGDO_o"]

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Download VADER sentiment lexicon (used by NLTK)
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Set up Streamlit page
st.set_page_config(page_title="Mental Health AI Companion", page_icon="🧠")
st.title("🧠 Mental Health AI Companion")
st.write("This is a private, stigma-free space. Talk to the AI about how you're feeling.")

# Get user input
user_input = st.text_area("🗣️ What's on your mind today?", height=150)

if st.button("Send"):
    if not user_input.strip():
        st.warning("Please type something first.")
    else:
        # Analyze mood
        sentiment = sia.polarity_scores(user_input)
        score = sentiment["compound"]
        if score >= 0.5:
            mood = "😊 Positive"
        elif score <= -0.5:
            mood = "😟 Negative"
        else:
            mood = "😐 Neutral"

        try:
            # Ask OpenAI for a response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}]
            )
            ai_reply = response.choices[0].message.content.strip()

            # Display AI reply and mood
            st.markdown("### 💬 AI Response")
            st.success(ai_reply)

            st.markdown("### 🧭 Mood Detection")
            st.info(f"**{mood}** (Sentiment Score: `{score}`)")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
