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