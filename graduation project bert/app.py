import streamlit as st
import joblib
import pandas as pd
import torch
from transformers import BertTokenizer

# Load the model and tokenizer
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

# Display the logo at the top (small size)
st.image('imglogo.png', width=100)

# Project description
st.markdown("""
    <h3 style='text-align: center;'>Graduation Project By The Students</h3>
    <ul style='text-align: center; list-style-type: none'>
        <li><b>Hadiel Ali Aljadid</b></li>
        <li><b>May Moktar Algallai</b></li>
    </ul>
""", unsafe_allow_html=True)

# Main content
st.markdown("<h1 style='text-align: center; color: #2986cc'>📧 Email Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict whether the email is <b>Spam</b> or <b>Ham</b></h4>", unsafe_allow_html=True)

# Input text area
user_input = st.text_area("✍️ Enter Email Text Below:")

if st.button("🔮 Submit"):
    if user_input:
        result = predict(user_input)
        if result == 1:
            st.markdown("<h2 style='text-align: center; color: #FF4136;'>Spam 🚫</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Be cautious! This email looks like spam ❗.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: #2ECC40;'>Ham ✅</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>📬 This email seems safe and legitimate.</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter the message text to proceed.")
