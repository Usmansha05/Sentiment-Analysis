import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import random

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
sentiment_pipeline = pipeline("sentiment-analysis")

# Sentiment analysis function
def analyze_sentiment(text):
    vader_result = sia.polarity_scores(text)
    vader_sentiment = "Positive" if vader_result["compound"] > 0.05 else "Negative" if vader_result["compound"] < -0.05 else "Neutral"
    
    bert_result = sentiment_pipeline(text)[0]
    bert_sentiment = "Positive" if bert_result['label'] == "POSITIVE" else "Negative"
    
    return vader_sentiment, bert_sentiment

# Suggestions for negative sentiment
def get_suggestion():
    suggestions = [
        "How about taking a short break to clear your mind? 🌞",
        "Maybe share your thoughts with a friend — it could lighten the load! 🤗",
        "Try focusing on something you enjoy for a bit! 🎨",
        "A quick walk might turn things around — fresh air works wonders! 🌳",
        "You’ve got this! How about a small step forward? 💪"
    ]
    return random.choice(suggestions)

# Appreciation for positive sentiment
def get_appreciation():
    appreciations = [
        "Wow, your positivity is contagious! Keep shining! ✨",
        "Love the good vibes — you’re inspiring! 🌟",
        "That’s awesome — thanks for spreading joy! 😊",
        "You’re on fire with that energy! 🔥",
        "Amazing outlook — keep it up! 🎉"
    ]
    return random.choice(appreciations)

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 20px;
        }
        .emoji {
            font-size: 80px;
            display: inline-block;
            animation: bounce 1s infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-15px); }
        }
        .result-box {
            background-color: #F0F2F6;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .text-area {
            font-size: 18px;
            border: 2px solid #FF4B4B;
            border-radius: 10px;
        }
        .button {
            background-color: #FF4B4B;
            color: white;
            font-size: 20px;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #FF6B6B;
        }
        .message {
            font-size: 24px;
            color: #333;
            font-weight: bold;
        }
        .suggestion {
            font-size: 20px;
            color: #4CAF50;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App layout
st.markdown('<div class="title">🎭 Sentiment Analysis Hub</div>', unsafe_allow_html=True)

user_text = st.text_area("Tell us how you feel:", height=150, key="text_input", 
                        help="Type your thoughts here!", 
                        placeholder="What’s on your mind? 😊")

if st.button("Analyze Now", key="analyze_button", help="Click to analyze your text!"):
    if user_text:
        vader_sentiment, bert_sentiment = analyze_sentiment(user_text)
        
        emoji_dict = {
            "Positive": "😃",
            "Negative": "😢",
            "Neutral": "😐"
        }
        
        # Determine final sentiment (using VADER as primary, BERT as tiebreaker for Neutral)
        if vader_sentiment == "Neutral":
            final_sentiment = bert_sentiment
        else:
            final_sentiment = vader_sentiment

        emoji = emoji_dict[final_sentiment]
        
        # Display result
        st.markdown(f'<div class="result-box"><span class="emoji">{emoji}</span>', unsafe_allow_html=True)
        
        if final_sentiment == "Positive":
            st.markdown(f'<p class="message">That sounds wonderful!</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="suggestion">{get_appreciation()}</p>', unsafe_allow_html=True)
        elif final_sentiment == "Negative":
            st.markdown(f'<p class="message">Hmm, that seems tough...</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="suggestion">{get_suggestion()}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="message">Keeping it balanced, huh?</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Oops! Please share some text first! 😅")
