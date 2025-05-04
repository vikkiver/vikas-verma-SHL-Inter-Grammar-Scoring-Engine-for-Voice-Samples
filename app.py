
import streamlit as st
import pandas as pd
import whisper
import os

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

st.title("Grammar Scoring App")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio("temp.wav")

    st.write("Transcribing...")
    result = model.transcribe("temp.wav")
    text = result["text"]
    st.write("**Transcription:**", text)

    # Dummy scoring logic
    if len(text.split()) > 15:
        score = 3
    elif len(text.split()) > 10:
        score = 2
    elif len(text.split()) > 5:
        score = 1
    else:
        score = 0

    labels = ["Poor", "Fair", "Good", "Excellent"]
    st.success(f"Grammar Score: {score} ({labels[score]})")
