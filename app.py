import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    vader_result = sia.polarity_scores(text)
    vader_sentiment = "Positive" if vader_result["compound"] > 0.05 else "Negative" if vader_result["compound"] < -0.05 else "Neutral"
    
    bert_result = sentiment_pipeline(text)[0]
    bert_sentiment = bert_result['label']
    
    return vader_sentiment, bert_sentiment

st.markdown("""
    <style>
        .emoji {
            font-size: 50px;
            display: inline-block;
            animation: bounce 1s infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎭 Sentiment Analysis App")
user_text = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if user_text:
        vader_sentiment, bert_sentiment = analyze_sentiment(user_text)
        
        emoji_dict = {
            "Positive": "😃",
            "Negative": "😢",
            "Neutral": "😐"
        }

        st.markdown(f'<span class="emoji">{emoji_dict[vader_sentiment]}</span>', unsafe_allow_html=True)
        st.write(f"**VADER Sentiment:** {vader_sentiment}")
        st.write(f"**BERT Sentiment:** {bert_sentiment}")
    else:
        st.warning("Please enter some text!")
