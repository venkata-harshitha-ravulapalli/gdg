import streamlit as st
from openai import OpenAI
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ✅ Set your OpenAI API key here (or use Streamlit secrets in production)
openai_api_key = "sk-Your-OpenAI-Key-Here"
client = OpenAI(api_key=openai_api_key)

# 📥 Download VADER sentiment model
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# 🧠 Streamlit UI
st.set_page_config(page_title="Mental Health AI", page_icon="🧠")
st.title("🧠 Mental Health AI Companion")
st.write("Talk about how you're feeling. This app offers a private, stigma-free space.")

# 📝 User input
user_input = st.text_area("🗣️ What's on your mind?", height=150)

if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # 🧠 Mood detection
        sentiment = sia.polarity_scores(user_input)
        score = sentiment["compound"]
        mood = "😊 Positive" if score >= 0.5 else "😟 Negative" if score <= -0.5 else "😐 Neutral"

        try:
            # 🤖 Get response from OpenAI (ChatGPT)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}]
            )
            ai_reply = response.choices[0].message.content.strip()

            # ✅ Show results
            st.markdown("### 💬 AI Response")
            st.success(ai_reply)

            st.markdown("### 🧭 Mood Detection")
            st.info(f"**{mood}** (Sentiment Score: `{score}`)")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
