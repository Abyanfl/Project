NLP Sentiment Analysis for Tweets
Project Overview

This project develops an Artificial Neural Network (ANN) model using TensorFlow for sentiment analysis on tweets related to various entities, such as Borderlands and Nvidia. The model classifies tweets into three sentiment categories: Positive, Negative, and Neutral (with the "Irrelevant" label merged into Neutral). The goal is to build a robust Natural Language Processing (NLP) model to help businesses monitor public perception on social media, enabling improved customer engagement and faster issue resolution.
Objective

To create an NLP model that accurately classifies tweet sentiments, assisting marketing teams and social media analysts in understanding brand reputation and customer feedback.
Dataset

    Training Data: twitter_training.csv (74,682 rows)
    Validation Data: twitter_validation.csv (1,000 rows)
    Both datasets contain tweets, associated entities, and sentiment labels.
    Data is stored on Google Drive (link provided in url.txt).

Problem Statement

Develop a sentiment analysis model to enable companies to understand public sentiment on social media, enhancing customer interaction and brand monitoring.
Justification

Sentiment analysis is critical for businesses to track brand reputation. Negative sentiments can highlight product issues (e.g., Nvidia driver complaints), while positive sentiments can inform marketing strategies (e.g., engaging Borderlands fans).

Source: Sprout Social.
Target Users

    Marketing teams
    Social media analysts

Project Structure

    Main.ipynb: Main notebook containing data loading, preprocessing, exploratory data analysis (EDA), model training, evaluation, and saving.
    inference.ipynb: Notebook for model inference on new tweet data.
    sentiment_model_best.h5: Saved trained model.
    tokenizer.pkl: Saved tokenizer for text preprocessing.
    twitter_training.csv: Training dataset.
    twitter_validation.csv: Validation dataset.
    url.txt: Contains Google Drive link to datasets.

Methodology

    Data Loading:
        Loaded twitter_training.csv and twitter_validation.csv.
        Handled missing values by dropping rows with null tweets.
        Merged "Irrelevant" sentiment into "Neutral" to simplify classification into three classes.
    Data Preprocessing:
        Encoded sentiment labels: Positive (2), Neutral (1), Negative (0).
        Performed text preprocessing (tokenization, stopword removal, etc.) using NLTK.
        Used TensorFlow's Tokenizer and pad_sequences for text vectorization.
    Exploratory Data Analysis (EDA):
        Analyzed sentiment distribution to ensure balanced classes.
        Visualized results using Seaborn and Matplotlib (e.g., confusion matrix).
    Model Development:
        Model 1: Baseline Sequential ANN model.
        Model 2: Advanced model using Functional API with Bidirectional LSTM and Dropout layers to prevent overfitting and capture contextual relationships.
        Model 2 outperformed Model 1, achieving high accuracy:
            Negative: 243/266 correct predictions
            Neutral: 425/457 correct predictions
            Positive: 256/277 correct predictions
    Model Evaluation:
        Evaluated using confusion matrix, classification report, and F1-score.
        Visualized performance with a confusion matrix heatmap.
    Model Saving:
        Saved the best model (sentiment_model_best.h5) and tokenizer (tokenizer.pkl).
    Inference:
        Conducted in a separate notebook (
        inference.ipynb) for testing on new tweet data.

Model Performance

    Model 2 (Bidirectional LSTM):
        Strengths:
            Captures bidirectional context for better understanding of tweet sentiment.
            Dropout layers prevent overfitting.
            High accuracy across all sentiment classes.
        Weaknesses:
            Computationally intensive, requiring longer training time.
            Minor prediction errors, particularly for ambiguous "Neutral" tweets.
        Improvement Suggestions:
            Use pre-trained embeddings (e.g., GloVe, BERT) for better accuracy.
            Implement an attention mechanism to focus on key words.
            Tune hyperparameters (e.g., learning rate) for optimization.
            Set a random seed to ensure consistent results across runs.