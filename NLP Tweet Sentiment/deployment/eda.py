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