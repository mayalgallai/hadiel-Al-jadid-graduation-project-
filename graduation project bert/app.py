import streamlit as st
import joblib
import pandas as pd
import torch
from transformers import BertTokenizer

# Load the model and vectorizer
model = joblib.load('best_model_bert.pkl')
tokenizer = joblib.load('best_vectorizer_bert.pkl')

# Prediction function
def predict(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
    )
    
    with torch.no_grad():
        outputs = model.model(inputs['input_ids'].to(model.device), 
                              attention_mask=inputs['attention_mask'].to(model.device))
        predictions = torch.argmax(outputs.logits, dim=1)
        
    return predictions.cpu().numpy()[0]

# User interface
st.title("Message Classification Model")
st.write("Enter the message text to predict if it's Spam or Ham.")

# Input text area
user_input = st.text_area("Message Text:")

if st.button("Predict"):
    if user_input:
        result = predict(user_input)
        if result == 1:
            st.write("Prediction: Spam")
        else:
            st.write("Prediction: Ham")
    else:
        st.write("Please enter the message text.")