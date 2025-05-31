import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import random
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

# Download required NLTK data
try:
    nltk.download("vader_lexicon", quiet=True)
except:
    pass

# Initialize sentiment analyzers
@st.cache_resource
def load_sentiment_analyzers():
    sia = SentimentIntensityAnalyzer()
    sentiment_pipeline = pipeline("sentiment-analysis", return_all_scores=True)
    return sia, sentiment_pipeline

sia, sentiment_pipeline = load_sentiment_analyzers()

# Enhanced sentiment analysis function
def analyze_sentiment(text):
    # VADER analysis
    vader_result = sia.polarity_scores(text)
    vader_compound = vader_result["compound"]
    
    if vader_compound > 0.05:
        vader_sentiment = "Positive"
    elif vader_compound < -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"
    
    # BERT analysis
    bert_result = sentiment_pipeline(text)[0]
    bert_scores = {item['label']: item['score'] for item in bert_result}
    
    # Get confidence scores
    positive_confidence = bert_scores.get('POSITIVE', 0)
    negative_confidence = bert_scores.get('NEGATIVE', 0)
    
    # Determine final sentiment with confidence
    if positive_confidence > negative_confidence:
        bert_sentiment = "Positive"
        confidence = positive_confidence * 100
    else:
        bert_sentiment = "Negative"
        confidence = negative_confidence * 100
    
    # Combine results for final decision
    final_sentiment = bert_sentiment if vader_sentiment == "Neutral" else vader_sentiment
    
    return {
        'sentiment': final_sentiment,
        'confidence': confidence,
        'vader_scores': vader_result,
        'bert_scores': bert_scores
    }

# Enhanced suggestions and appreciations
def get_suggestion():
    suggestions = [
        "🤖 Reboot with a quick break — refresh your neural circuits!",
        "⚡️ Sync with a friend — their input might debug the mood!",
        "🕹️ Switch tasks to something fun — recharge protocol initiated!",
        "🌐 Step into the grid — a walk could reset your system!",
        "🚀 Error detected: You're tougher than any system glitch!",
        "💤 Try a power nap — 10 minutes could flip the script!",
        "🎶 Engage distraction mode: Music or games might help!",
        "🧘 Log out of stress — deep breaths could stabilize you!",
        "☕ Execute caffeine.exe — fuel up your processors!",
        "🌟 Run self-care.bat — you deserve some maintenance time!"
    ]
    return random.choice(suggestions)

def get_appreciation():
    appreciations = [
        "⚡️ Overclocking positivity — you're absolutely unstoppable!",
        "🌩️ High-voltage vibes — system admires your incredible spark!",
        "🖥️ Joy levels at maximum — you're a premium core processor!",
        "🌌 Peak performance detected — radiating galactic energy!",
        "🤖 System status: Elite mode — keep that flow online!",
        "🌟 Radiating pure awesomeness — you're a brilliant neon beacon!",
        "📡 Positivity bandwidth at full capacity — truly inspiring!",
        "🎹 You're a code maestro — executing flawless life algorithms!",
        "🔥 Energy levels: Supernova — lighting up the digital cosmos!",
        "✨ Stellar performance detected — you're absolutely phenomenal!"
    ]
    return random.choice(appreciations)

# Create confidence meter visualization
def create_confidence_meter(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Analysis Confidence", 'font': {'color': 'white', 'size': 20}},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': "white"},
            'bar': {'color': "#00FFFF"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255, 0, 110, 0.3)"},
                {'range': [50, 80], 'color': "rgba(255, 255, 0, 0.3)"},
                {'range': [80, 100], 'color': "rgba(0, 255, 0, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Courier New"},
        height=300
    )
    return fig

# Create sentiment breakdown chart
def create_sentiment_breakdown(vader_scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [vader_scores['pos'], vader_scores['neu'], vader_scores['neg']]
    colors = ['#00FF00', '#FFFF00', '#FF0000']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.3,
        marker_colors=colors,
        textfont={'color': 'white', 'size': 14}
    )])
    
    fig.update_layout(
        title={'text': 'Sentiment Breakdown', 'font': {'color': 'white', 'size': 20}, 'x': 0.5},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Courier New"},
        height=400
    )
    return fig

# Enhanced CSS with more animations and better styling
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0c0c0c 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            font-family: 'Orbitron', 'Courier New', monospace;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Custom Title Animation */
        .cosmic-title {
            font-family: 'Orbitron', monospace;
            font-size: 4rem;
            font-weight: 900;
            text-align: center;
            background: linear-gradient(45deg, #ff006e, #8338ec, #3a86ff, #06ffa5);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientText 3s ease infinite, pulse 2s ease-in-out infinite;
            text-shadow: 0 0 50px rgba(255, 0, 110, 0.5);
            margin-bottom: 2rem;
        }
        
        @keyframes gradientText {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        /* Glowing Cards */
        .analysis-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .analysis-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 35px 60px rgba(0, 0, 0, 0.4);
            border-color: rgba(0, 255, 255, 0.3);
        }
        
        /* Result Cards with Sentiment Colors */
        .result-positive {
            background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 255, 255, 0.1));
            border: 2px solid rgba(0, 255, 0, 0.3);
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.2);
        }
        
        .result-negative {
            background: linear-gradient(135deg, rgba(255, 0, 0, 0.1), rgba(255, 100, 100, 0.1));
            border: 2px solid rgba(255, 0, 0, 0.3);
            box-shadow: 0 0 30px rgba(255, 0, 0, 0.2);
        }
        
        .result-neutral {
            background: linear-gradient(135deg, rgba(255, 255, 0, 0.1), rgba(100, 100, 255, 0.1));
            border: 2px solid rgba(255, 255, 0, 0.3);
            box-shadow: 0 0 30px rgba(255, 255, 0, 0.2);
        }
        
        /* Animated Emoji */
        .sentiment-emoji {
            font-size: 4rem;
            text-align: center;
            animation: bounce 2s infinite, rotate 4s linear infinite;
            margin: 1rem 0;
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0) rotate(0deg); }
            40% { transform: translateY(-10px) rotate(5deg); }
            60% { transform: translateY(-5px) rotate(-5deg); }
        }
        
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Enhanced Text Styling */
        .status-text {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            color: #00FFFF;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
            margin: 1rem 0;
        }
        
        .message-text {
            font-size: 1.3rem;
            text-align: center;
            color: #FFFFFF;
            margin: 1rem 0;
            line-height: 1.6;
        }
        
        .suggestion-box {
            background: linear-gradient(45deg, rgba(255, 0, 110, 0.15), rgba(131, 56, 236, 0.15));
            border: 1px solid rgba(255, 0, 110, 0.4);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { box-shadow: 0 0 10px rgba(255, 0, 110, 0.3); }
            to { box-shadow: 0 0 20px rgba(255, 0, 110, 0.6); }
        }
        
        /* Login Enhancement */
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            text-align: center;
        }
        
        .welcome-message {
            font-size: 1.2rem;
            color: #a0a9c0;
            margin-bottom: 2rem;
            animation: fadeInUp 1s ease;
        }
        
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Button Enhancements */
        .stButton > button {
            background: linear-gradient(45deg, #ff006e, #8338ec) !important;
            color: white !important;
            border: none !important;
            border-radius: 15px !important;
            padding: 0.75rem 2rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 10px 25px rgba(255, 0, 110, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 15px 35px rgba(255, 0, 110, 0.5) !important;
        }
        
        /* Text Area Enhancement */
        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.08) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 15px !important;
            color: white !important;
            font-size: 1.1rem !important;
            min-height: 150px !important;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #00FFFF !important;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3) !important;
        }
        
        /* Input Enhancement */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.08) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 15px !important;
            color: white !important;
            font-size: 1.1rem !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #00FFFF !important;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3) !important;
        }
        
        /* Sidebar Enhancement */
        .css-1d391kg {
            background: rgba(0, 0, 0, 0.3) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Metrics Enhancement */
        .metric-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

# Main application
def main():
    st.set_page_config(
        page_title="Senti-Aly | Cosmic Sentiment Analysis",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    load_css()
    init_session_state()
    
    # Login Screen
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown('<h1 class="cosmic-title">Senti-Aly</h1>', unsafe_allow_html=True)
            st.markdown('<p class="welcome-message">🌌 Analyze your cosmic emotions with interstellar precision!</p>', unsafe_allow_html=True)
            
            # Lottie animation placeholder
            st.markdown("""
                <div style="text-align: center; margin: 2rem 0;">
                    <div style="font-size: 8rem; animation: pulse 2s ease-in-out infinite;">🚀</div>
                </div>
            """, unsafe_allow_html=True)
            
            with st.form(key="login_form"):
                username = st.text_input("🌟 Enter your galactic ID:", placeholder="Cosmonaut Name", key="username_input")
                submit = st.form_submit_button("🚀 Launch into the Cosmos", use_container_width=True)
                
                if submit and username:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main App
    else:
        # Header
        st.markdown(f'<h1 class="cosmic-title">🤖 Senti-Aly Universe</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<p class="welcome-message">Welcome back, <strong>{st.session_state.username}</strong>! 🌟</p>', unsafe_allow_html=True)
        
        with col3:
            if st.button("🌠 Logout", key="logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.analysis_history = []
                st.rerun()
        
        # Main Analysis Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Sentiment Analysis Engine")
            
            with st.form(key="sentiment_form"):
                user_text = st.text_area(
                    "🌐 Input your data stream:",
                    height=200,
                    placeholder="What's processing in your neural networks today? Share your thoughts, feelings, or any text you'd like me to analyze... ⚡️",
                    help="Transmit your thoughts here for deep cosmic analysis!"
                )
                submit_button = st.form_submit_button("🚀 Execute Deep Analysis", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis Results
            if submit_button and user_text:
                with st.spinner('🔄 Analyzing cosmic frequencies...'):
                    time.sleep(1)  # Add dramatic pause
                    result = analyze_sentiment(user_text)
                
                sentiment = result['sentiment']
                confidence = result['confidence']
                
                # Add to history
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'text': user_text[:100] + "..." if len(user_text) > 100 else user_text,
                    'sentiment': sentiment,
                    'confidence': confidence
                })
                
                # Display results with enhanced styling
                sentiment_class = f"result-{sentiment.lower()}"
                st.markdown(f'<div class="analysis-card {sentiment_class}">', unsafe_allow_html=True)
                
                # Emoji and status
                emoji_dict = {
                    "Positive": random.choice(["🌟", "⚡️", "🚀", "🎉", "💡", "🌈", "🪐", "✨"]),
                    "Negative": random.choice(["⚠️", "💥", "🌀", "📉", "🌩️", "🛑", "☄️", "🔧"]),
                    "Neutral": random.choice(["🤖", "🧠", "⚙️", "📊", "🔍", "🔔", "🌍", "⚖️"])
                }
                
                emoji = emoji_dict[sentiment]
                st.markdown(f'<div class="sentiment-emoji">{emoji}</div>', unsafe_allow_html=True)
                
                # Status messages
                if sentiment == "Positive":
                    status_messages = ["System Status: Stellar!", "Cosmic Positivity Detected!", "Galactic Happiness Active!"]
                    st.markdown(f'<p class="status-text">{random.choice(status_messages)}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="message-text">Your energy levels are off the charts! 🌟</p>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                    st.markdown(f'<p class="message-text">{get_appreciation()}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                elif sentiment == "Negative":
                    status_messages = ["Alert: Cosmic Disturbance...", "System Anomaly Detected", "Emotional Glitch Found"]
                    st.markdown(f'<p class="status-text">{random.choice(status_messages)}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="message-text">Don\'t worry, every system has temporary bugs! 🔧</p>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                    st.markdown(f'<p class="message-text">{get_suggestion()}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.markdown(f'<p class="status-text">Status: Orbital Balance ⚖️</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="message-text">Maintaining perfect cosmic equilibrium! 🌍</p>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar with analytics
        with col2:
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Analysis Metrics")
            
            if submit_button and user_text:
                # Confidence meter
                fig_confidence = create_confidence_meter(confidence)
                st.plotly_chart(fig_confidence, use_container_width=True)
                
                # Sentiment breakdown
                fig_breakdown = create_sentiment_breakdown(result['vader_scores'])
                st.plotly_chart(fig_breakdown, use_container_width=True)
                
                # Detailed scores
                st.markdown("#### 🔢 Detailed Scores")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_b:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Compound", f"{result['vader_scores']['compound']:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analysis History
            if st.session_state.analysis_history:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown("### 📈 Analysis History")
                
                for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                    sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                    st.markdown(f"""
                        <div style="padding: 0.5rem; margin: 0.3rem 0; background: rgba(255,255,255,0.05); border-radius: 8px;">
                            <small>{analysis['timestamp'].strftime('%H:%M')}</small><br>
                            {sentiment_color[analysis['sentiment']]} <strong>{analysis['sentiment']}</strong> ({analysis['confidence']:.0f}%)<br>
                            <em>{analysis['text']}</em>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
