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

# Expanded suggestions for negative sentiment
def get_suggestion():
    suggestions = [
        "Reboot with a quick break — refresh your circuits! 🤖",
        "Sync with a friend — their input might debug the mood! ⚡️",
        "Switch tasks to something fun — recharge initiated! 🕹️",
        "Step into the grid — a walk could reset your system! 🌐",
        "Error detected: You’re tougher than this glitch! 🚀",
        "Try a power nap — 10 minutes could flip the script! 💤",
        "Engage distraction mode: Music or a game might help! 🎶",
        "Log out of stress — deep breaths could stabilize you! 🧘"
    ]
    return random.choice(suggestions)

# Expanded appreciation for positive sentiment
def get_appreciation():
    appreciations = [
        "Overclocking positivity — you’re unstoppable! ⚡️",
        "High-voltage vibes — system admires your spark! 🌩️",
        "Joy levels at max — you’re a core processor! 🖥️",
        "Peak performance detected — galactic energy! 🌌",
        "System status: Elite — keep that flow online! 🤖",
        "Radiating awesomeness — you’re a neon beacon! 🌟",
        "Positivity bandwidth full — inspiring output! 📡",
        "You’re a code maestro — flawless execution! 🎹"
    ]
    return random.choice(appreciations)

# Custom CSS with neon background simulation
st.markdown("""
    <style>
        .title {
            font-size: 50px;
            font-weight: bold;
            color: #00FFFF;
            text-align: center;
            margin-bottom: 25px;
            text-shadow: 0 0 10px #00FFFF;
            font-family: 'Courier New', monospace;
        }
        .emoji {
            font-size: 40px;
            display: inline-block;
            margin-right: 10px;
        }
        .result-box {
            background-color: rgba(26, 26, 26, 0.8);
            padding: 25px;
            border-radius: 10px;
            border: 2px solid #00FFFF;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
            margin-top: 30px;
            min-height: 200px;
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
            font-size: 28px;
            color: #FFFFFF;
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }
        .suggestion {
            font-size: 24px;
            color: #00FFFF;
            margin-top: 15px;
            text-shadow: 0 0 5px #00FFFF;
            font-family: 'Courier New', monospace;
        }
        body {
            background: linear-gradient(135deg, #000000 0%, #00FFFF 50%, #000000 100%);
            background-size: 200% 200%;
            animation: neonFlow 10s ease infinite;
            color: #FFFFFF;
        }
        @keyframes neonFlow {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
        }
    </style>
""", unsafe_allow_html=True)

# Lottie animation
st.markdown("""
    <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
    <dotlottie-player src="https://lottie.host/9c1a089d-c174-4478-9aed-a2360b447aaa/7G56ObfSMl.lottie" 
        background="transparent" speed="1" style="width: 300px; height: 300px; margin: auto;" loop autoplay>
    </dotlottie-player>
""", unsafe_allow_html=True)

# App layout with form for Enter key support
st.markdown('<div class="title">🤖 Senti-Aly App</div>', unsafe_allow_html=True)

with st.form(key="sentiment_form"):
    user_text = st.text_area("Input your data stream:", height=200, key="text_input", 
                            help="Transmit your thoughts here!", 
                            placeholder="What’s processing in your system? ⚡️")
    submit_button = st.form_submit_button(label="Execute Analysis", help="Initiate sentiment processing!")

if submit_button and user_text:
    vader_sentiment, bert_sentiment = analyze_sentiment(user_text)
    
    emoji_dict = {
        "Positive": random.choice(["🌟", "⚡️", "🚀", "🎉", "💡", "🌈"]),  # More diverse emojis
        "Negative": random.choice(["⚠️", "💥", "🌀", "📉", "🌩️", "🛑"]),
        "Neutral": random.choice(["🤖", "🧠", "⚙️", "📊", "🔍", "🔔"])
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
elif submit_button and not user_text:
    st.warning("Error: No data input detected! ⚠️")
