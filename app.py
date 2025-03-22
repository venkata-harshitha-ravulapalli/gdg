from flask import Flask, request, jsonify
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = "sk-your-api-key-here"

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return jsonify({"message": "Mental Health AI is running"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    sentiment = sia.polarity_scores(user_input)
    mood = (
        "positive" if sentiment['compound'] >= 0.5 else
        "negative" if sentiment['compound'] <= -0.5 else
        "neutral"
    )

    try:
        ai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}]
        )
        reply = ai_response.choices[0].message.content.strip()

        return jsonify({
            "reply": reply,
            "mood": mood,
            "score": sentiment['compound']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
