import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import random

nltk.download("vader_lexicon")

# Sentiment analysis setup
sia = SentimentIntensityAnalyzer()
sentiment_pipeline = pipeline("sentiment-analysis")

# Sentiment analysis function
def analyze_sentiment(text):
    vader_result = sia.polarity_scores(text)
    vader_sentiment = "Positive" if vader_result["compound"] > 0.05 else "Negative" if vader_result["compound"] < -0.05 else "Neutral"
    bert_result = sentiment_pipeline(text)[0]
    bert_sentiment = "Positive" if bert_result['label'] == "POSITIVE" else "Negative"
    return vader_sentiment, bert_sentiment

# Suggestions and appreciations
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

# Custom CSS for intro and app pages
image_path = "galaxy-night-view.jpg"
st.markdown("""
    <style>
        /* Space-themed background */
        body {
            background: url("https://raw.github.com/Usmansha05/Sentiment-Analysis/main/galaxy-night-view.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #FFFFFF;
            font-family: 'Courier New', monospace;
        }
        @keyframes spaceFlow {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
        }

        /* Glowing RGB text */
        .glow-text {
            font-size: 60px;
            font-weight: bold;
            text-align: center;
            color: #FFFFFF;
            animation: rgbGlow 3s infinite;
        }
        @keyframes rgbGlow {
            0% { text-shadow: 0 0 10px #FF0000, 0 0 20px #FF0000; }
            33% { text-shadow: 0 0 10px #00FF00, 0 0 20px #00FF00; }
            66% { text-shadow: 0 0 10px #0000FF, 0 0 20px #0000FF; }
            100% { text-shadow: 0 0 10px #FF0000, 0 0 20px #FF0000; }
        }

        /* Button styles */
        .space-button {
            background: linear-gradient(45deg, #FF00FF, #00FFFF);
            color: #000000;
            font-size: 22px;
            padding: 12px 25px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            box-shadow: 0 0 15px #00FFFF;
            transition: all 0.3s ease;
            font-family: 'Courier New', monospace;
        }
        .space-button:hover {
            transform: scale(1.1);
            box-shadow: 0 0 25px #FF00FF, 0 0 35px #00FFFF;
            background: linear-gradient(45deg, #00FFFF, #FF00FF);
        }

        /* Input field */
        .space-input {
            font-size: 20px;
            border: 2px solid #00FFFF;
            border-radius: 8px;
            background-color: rgba(29, 53, 87, 0.8);
            color: #FFFFFF;
            padding: 10px;
            width: 300px;
            margin: 20px auto;
            display: block;
        }

        /* App-specific styles */
        .title {
            font-size: 50px;
            font-weight: bold;
            color: #00FFFF;
            text-align: center;
            margin-bottom: 25px;
            text-shadow: 0 0 10px #00FFFF;
        }
        .emoji {
            font-size: 40px;
            display: inline-block;
            margin-right: 10px;
        }
        .result-box {
            background-color: rgba(26, 26, 26, 0.8);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #00FFFF;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
            margin-top: 20px;
        }
        .text-area {
            font-size: 20px;
            border: 2px solid #00FFFF;
            border-radius: 8px;
            background-color: rgba(29, 53, 87, 0.8);
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
        }
        .button:hover {
            background-color: #00CCCC;
            box-shadow: 0 0 15px #00CCCC;
        }
        .message {
            font-size: 32px;
            color: #FFFFFF;
            font-weight: bold;
            padding: 10px 0;
        }
        .suggestion {
            font-size: 28px;
            color: #00FFFF;
            margin-top: 10px;
            text-shadow: 0 0 5px #00FFFF;
            padding: 15px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Intro page
if not st.session_state.logged_in:
    st.markdown("""
        <div style="text-align: center; padding-top: 50px;">
            <h1 class="glow-text">Welcome to Senti-Aly App 🌌</h1>
            <p style="font-size: 20px; color: #E0E1DD;">Analyze your cosmic emotions with interstellar precision!</p>
    """, unsafe_allow_html=True)

    # Lottie animation using st.components.v1.html
    lottie_html = """
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <dotlottie-player src="https://lottie.host/9c1a089d-c174-4478-9aed-a2360b447aaa/7G56ObfSMl.lottie" 
            background="transparent" speed="1" style="width: 300px; height: 300px; margin: auto;" loop autoplay>
        </dotlottie-player>
    """
    st.components.v1.html(lottie_html, height=300)

    # Login form
    with st.form(key="login_form"):
        username = st.text_input("Enter your galactic ID:", key="username_input", placeholder="Cosmonaut Name")
        submit = st.form_submit_button(label="Launch 🚀", help="Enter the Senti-Aly universe!")

        if submit and username:
            # Safely update session state
            st.session_state["logged_in"] = True
            st.session_state["username"] = username

    # No st.rerun() here, let Streamlit handle the state transition naturally

# Main app page
else:
    st.markdown(f'<div class="title">🤖 Senti-Aly App | Welcome, {st.session_state.username}!</div>', unsafe_allow_html=True)
    
    # Logout button
    if st.button("Logout 🌠", key="logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        # No st.rerun() needed here, Streamlit will handle it

    with st.form(key="sentiment_form"):
        user_text = st.text_area("Input your data stream:", height=200, key="text_input", 
                                help="Transmit your thoughts here!", 
                                placeholder="What’s processing in your system? ⚡️")
        submit_button = st.form_submit_button(label="Execute Analysis", help="Initiate sentiment processing!")

    if submit_button and user_text:
        vader_sentiment, bert_sentiment = analyze_sentiment(user_text)
        
        emoji_dict = {
            "Positive": random.choice(["🌟", "⚡️", "🚀", "🎉", "💡", "🌈", "🪐"]),
            "Negative": random.choice(["⚠️", "💥", "🌀", "📉", "🌩️", "🛑", "☄️"]),
            "Neutral": random.choice(["🤖", "🧠", "⚙️", "📊", "🔍", "🔔", "🌍"])
        }
        
        final_sentiment = bert_sentiment if vader_sentiment == "Neutral" else vader_sentiment
        emoji = emoji_dict[final_sentiment]
        
        st.markdown(f'<div class="result-box"><span class="emoji">{emoji}</span>', unsafe_allow_html=True)
        
        if final_sentiment == "Positive":
            st.markdown(f'<p class="message">System Status: Stellar!</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="suggestion">{get_appreciation()}</p>', unsafe_allow_html=True)
        elif final_sentiment == "Negative":
            st.markdown(f'<p class="message">Alert: Cosmic Disturbance...</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="suggestion">{get_suggestion()}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="message">Status: Orbital Balance</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    elif submit_button and not user_text:
        st.warning("Error: No data input detected! ⚠️")
