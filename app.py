import streamlit as st
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set your OpenAI API key here
openai.api_key = "sk-your-openai-api-key-here"

# Download NLTK sentiment model
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Streamlit UI
st.set_page_config(page_title="Mental Health AI", page_icon="🧠")
st.title("🧠 AI Mental Health Companion")
st.write("Talk about how you feel, and get support from AI 💬")

user_input = st.text_area("📝 How are you feeling today?", height=150)

if st.button("Send to AI"):
    if not user_input.strip():
        st.warning("Please type something first.")
    else:
        with st.spinner("Analyzing and generating response..."):
            # Sentiment Analysis
            sentiment = sia.polarity_scores(user_input)
            compound = sentiment["compound"]
            mood = "Positive 😀" if compound >= 0.5 else "Negative 😟" if compound <= -0.5 else "Neutral 😐"

            try:
                # Get AI reply from OpenAI
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": user_input}]
                )
                reply = response.choices[0].message.content.strip()

                # Show results
                st.markdown("### 💬 AI's Reply")
                st.success(reply)
                st.markdown("### 📊 Mood Detected")
                st.info(f"{mood} (Sentiment Score: {compound:.2f})")

            except Exception as e:
                st.error(f"Error from OpenAI: {e}")
