#step1: Import libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

#load the pre-trained model with tanh activation
model = load_model('simple_rnn_imdb.h5')

#step:2 helper functions
#function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text, word_index, max_len=500):  # Changed to 500, typical IMDB length
    words = text.lower().split()
    # Add 3 to align with IMDB indexing (0=pad, 1=start, 2=unknown)
    encoded = [word_index.get(word, 2) + 3 for word in words]
    padded_sequence = pad_sequences([encoded], maxlen=max_len)
    return padded_sequence

def predict_sentiment(model, processed_input):
    prediction = model.predict(processed_input)[0][0]
    # Changed threshold to 0.5 (standard for binary classification)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    # Add confidence level
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, float(prediction), float(confidence)

import streamlit as st

st.title('IMDb Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review', height=150)

if st.button('Classify'):
    if user_input.strip():  # Check if input is not empty
        processed_input = preprocess_text(user_input, word_index)
        sentiment, prediction, confidence = predict_sentiment(model, processed_input)
        
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Prediction Score:** {prediction:.4f}')
        st.write(f'**Confidence:** {confidence:.2%}')
        
        # Visual indicator
        if sentiment == "Positive":
            st.success(f'✓ Positive sentiment detected with {confidence:.1%} confidence')
        else:
            st.error(f'✗ Negative sentiment detected with {confidence:.1%} confidence')
    else:
        st.warning('Please enter a movie review.')
else:
    st.info('Enter a review above and click "Classify" to analyze sentiment.')
    
st.write('---')
st.write('**Sample Reviews to Try:**')
st.write('• Positive: "This movie was fantastic and so thrilling! Best film I\'ve seen this year."')
st.write('• Negative: "Terrible waste of time. Poor acting and boring plot."')