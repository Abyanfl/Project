import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    st.stop()

# Load model and tokenizer
try:
    model = tf.keras.models.load_model('sentiment_model_best.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Load and preprocess dataset for EDA
@st.cache
def load_data():
    try:
        df = pd.read_csv('twitter_training.csv', names=['ID', 'Entity', 'Sentiment', 'Tweet'])
        df['Tweet'] = df['Tweet'].fillna('')
        df['Sentiment'] = df['Sentiment'].replace('Irrelevant', 'Neutral')
        df['Tweet_Length'] = df['Tweet'].apply(lambda x: len(str(x).split()))
        return df, None
    except Exception as e:
        return None, str(e)

# Preprocessing function for inference
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess tweet text for inference."""
    if not isinstance(text, str) or not text.strip():
        return "", "Error: Empty or invalid input"
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens), None

# Inference function
def predict_sentiment(tweet):
    """Predict sentiment for a given tweet."""
    try:
        processed_tweet, error = preprocess_text(tweet)
        if error:
            return error
        if not processed_tweet:
            return "Error: No valid text after preprocessing"

        max_len = 50  # Matches training configuration
        seq = tokenizer.texts_to_sequences([processed_tweet])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        pred = model.predict(padded, verbose=0)
        sentiment_idx = np.argmax(pred, axis=1)[0]
        sentiment_map = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}  # Matches training
        sentiment = sentiment_map[sentiment_idx]
        return f"Predicted Sentiment: {sentiment}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Streamlit interface
st.title("Twitter Sentiment Analysis")
st.write("""
Aplikasi ini menggunakan model TensorFlow berbasis ANN dengan arsitektur bidirectional LSTM untuk mengklasifikasikan sentimen tweet menjadi Positif, Negatif, atau Netral.
Model ini dilatih menggunakan data Twitter yang membahas entitas seperti Borderlands dan Nvidia, dengan sentimen 'Irrelevant' digabungkan ke dalam kategori Netral.
Jelajahi wawasan dari dataset pada tab Analisis Data Eksplorasi (EDA) atau lakukan prediksi sentimen untuk tweet baru pada tab Prediksi.""")

# Create tabs for Prediction and EDA
tab1, tab2 = st.tabs(["Sentiment Prediction", "Exploratory Data Analysis"])

# Prediction Tab
with tab1:
    st.header("Predict Tweet Sentiment")
    tweet = st.text_area("Enter a tweet", height=100, placeholder="e.g., I love playing Borderlands, it's so much fun!")
    if st.button("Predict"):
        if tweet:
            result = predict_sentiment(tweet)
            st.write(f"**Tweet**: {tweet}")
            st.write(f"**{result}**")
        else:
            st.write("Please enter a tweet to predict.")

# EDA Tab
with tab2:
    st.header("Exploratory Data Analysis")
    df, error = load_data()
    if error:
        st.error(f"Error loading dataset: {error}")
    elif df is not None:
        st.subheader("Dataset Overview")
        st.write(f"Dataset terdiri dari {df.shape[0]} tweet yang mencakup {df['Entity'].nunique()} entitas berbeda.")
        
        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig1 = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                      title="Sentiment Distribution in Training Data")
        st.plotly_chart(fig1)
        st.write("menggambarkan distribusi sentimen, di mana sentimen Netral menjadi yang paling dominan akibat penggabungan kategori 'Irrelevant'.")

        # Entity Distribution
        st.subheader("Entity Distribution")
        entity_counts = df['Entity'].value_counts().reset_index()
        entity_counts.columns = ['Entity', 'Count']
        fig2 = px.bar(entity_counts, x='Count', y='Entity', orientation='h',
                      title="Entity Distribution in Training Data",
                      height=600)
        st.plotly_chart(fig2)
        st.write("Plot ini menunjukkan frekuensi tweet berdasarkan entitas, menyoroti topik-topik yang paling banyak dibahas.")
        # Tweet Length Distribution
        st.subheader("Tweet Length Distribution")
        fig3 = px.histogram(df, x='Tweet_Length', nbins=30,
                            title="Tweet Length Distribution (Word Count)",
                            labels={'Tweet_Length': 'Number of Words'})
        st.plotly_chart(fig3)
        st.write(f"memperlihatkan distribusi panjang tweet dalam jumlah kata, dengan rata-rata sekitar {df['Tweet_Length'].mean():.2f} kata dan panjang maksimum mencapai {df['Tweet_Length'].max()} kata. Informasi ini menjadi dasar penentuan nilai max_len=50 untuk proses padding.")