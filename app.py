import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import random
from streamlit_lottie import st_lottie
import requests

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

# Function to load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Suggestions for negative sentiment
def get_suggestion():
    suggestions = [
        "Take a moment to reboot — a break might help! 🤖",
        "Ping a friend for a quick sync — it could lift your circuits! ⚡️",
        "Switch to a fun task for a bit — recharge those vibes! 🕹️",
        "Step into the grid for a walk — fresh data incoming! 🌐",
        "System check: You're stronger than this glitch! 🚀"
    ]
    return random.choice(suggestions)

# Appreciation for positive sentiment
def get_appreciation():
    appreciations = [
        "Your positivity is overclocking the system — epic! ⚡️",
        "High-voltage vibes detected — you're electric! 🌩️",
        "Processing joy at max capacity — awesome! 🖥️",
        "You're running at peak performance — stellar! 🌌",
        "System approves: Keep that energy flowing! 🤖"
    ]
    return random.choice(appreciations)

# Custom CSS for robotic theme
st.markdown("""
    <style>
        .title {
            font-size: 50px;
            font-weight: bold;
            color: #00FFFF; /* Cyan */
            text-align: center;
            margin-bottom: 25px;
            text-shadow: 0 0 10px #00FFFF;
            font-family: 'Courier New', monospace;
        }
        .emoji {
            font-size: 24px; /* Smaller emoji box */
            display: inline-block;
            margin-right: 5px;
        }
        .result-box {
            background-color: #1A1A1A; /* Dark gray */
            padding: 25px;
            border-radius: 10px;
            border: 2px solid #00FFFF;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
            margin-top: 30px;
            min-height: 200px; /* Bigger response field */
        }
        .text-area {
            font-size: 20px;
            border: 2px solid #00FFFF;
            border-radius: 8px;
            background-color: #2D2D2D;
            color: #FFFFFF;
            padding: 10px;
        }
        .button {
            background-color: #00FFFF;
            color: #000000;
            font-size: 22px;
            padding: 12px 25px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            box-shadow: 0 0 10px #00FFFF;
            font-family: 'Courier New', monospace;
        }
        .button:hover {
            background-color: #00CCCC;
            box-shadow: 0 0 15px #00CCCC;
        }
        .message {
            font-size: 28px; /* Larger message */
            color: #FFFFFF;
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }
        .suggestion {
            font-size: 24px; /* Larger suggestions */
            color: #00FFFF;
            margin-top: 15px;
            text-shadow: 0 0 5px #00FFFF;
            font-family: 'Courier New', monospace;
        }
        .lottie-container {
            display: flex;
            justify-content: center;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        body {
            background-color: #000000; /* Black background */
            color: #FFFFFF;
        }
        /* For custom iframe embedding */
        .lottie-iframe {
            width: 150px;
            height: 150px;
            border: none;
            margin: 0 auto;
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

# App layout
st.markdown('<div class="title">🤖 Sentiment Analysis Core</div>', unsafe_allow_html=True)

# Add Lottie animation (using streamlit_lottie method)
lottie_url = "https://lottie.host/9c1a089d-c174-4478-9aed-a2360b447aaa/7G56ObfSMl.lottie"
lottie_json = load_lottieurl(lottie_url)

# Display Lottie animation
if lottie_json:
    st.markdown('<div class="lottie-container">', unsafe_allow_html=True)
    st_lottie(lottie_json, speed=1, height=200, key="lottie")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Fallback to iframe method if streamlit_lottie doesn't work
    st.markdown("""
    <div class="lottie-container">
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <dotlottie-player src="https://lottie.host/9c1a089d-c174-4478-9aed-a2360b447aaa/7G56ObfSMl.lottie" 
                         background="transparent" speed="1" style="width: 200px; height: 200px" loop autoplay>
        </dotlottie-player>
    </div>
    """, unsafe_allow_html=True)

user_text = st.text_area("Input your data stream:", height=200, key="text_input", 
                        help="Transmit your thoughts here!", 
                        placeholder="What's processing in your system? ⚡️")

if st.button("Execute Analysis", key="analyze_button", help="Initiate sentiment processing!"):
    if user_text:
        vader_sentiment, bert_sentiment = analyze_sentiment(user_text)
        
        emoji_dict = {
            "Positive": "🚀",  # More robotic-themed icons
            "Negative": "⚠️",
            "Neutral": "🤖"
        }
        
        # Determine final sentiment
        if vader_sentiment == "Neutral":
            final_sentiment = bert_sentiment
        else:
            final_sentiment = vader_sentiment

        emoji = emoji_dict[final_sentiment]
        
        # Display result
        st.markdown(f'<div class="result-box"><span class="emoji">{emoji}</span>', unsafe_allow_html=True)
        
        if final_sentiment == "Positive":
            st.markdown(f'<p class="message">System Status: Optimal!</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="suggestion">{get_appreciation()}</p>', unsafe_allow_html=True)
        elif final_sentiment == "Negative":
            st.markdown(f'<p class="message">Alert: Low Energy Detected...</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="suggestion">{get_suggestion()}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="message">Status: Stable Output</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Error: No data input detected! ⚠️")
