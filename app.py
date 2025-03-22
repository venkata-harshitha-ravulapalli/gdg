import streamlit as st
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set your OpenAI API key here
openai.api_key = "sk-your-api-key-here"

# Download NLTK resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ------------------- Flask Backend ------------------- #
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    sentiment = sia.polarity_scores(user_input)
    compound = sentiment['compound']
    mood = "positive" if compound >= 0.5 else "negative" if compound <= -0.5 else "neutral"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}]
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "reply": reply,
        "mood": mood,
        "score": compound
    })

def run_flask():
    app.run(port=5000)

# ------------------- Streamlit Frontend ------------------- #
def run_streamlit():
    st.set_page_config(page_title="Mental Health AI", page_icon="🧠")
    st.title("🧠 AI Mental Health Companion")

    user_input = st.text_area("How are you feeling today?", height=150)

    if st.button("Send to AI"):
        if not user_input.strip():
            st.warning("Please type something.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        "http://localhost:5000/chat",
                        json={"message": user_input}
                    )
                    data = response.json()
                    if "reply" in data:
                        st.markdown("### 💬 AI Reply")
                        st.success(data["reply"])
                        st.markdown("### 📊 Mood Detected")
                        st.info(f"{data['mood'].capitalize()} (Score: {data['score']:.2f})")
                    else:
                        st.error("Something went wrong.")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# ------------------- Start Both ------------------- #
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(2)  # Allow Flask to start
    run_streamlit()
