import sys
import os

# Add parent directory (project root) to sys.path before importing run_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from orchestrator.manager import run_manager
import pyttsx3

# Set up the TTS engine
engine = pyttsx3.init()
AUDIO_FILE = "output_audio.mp3"

def save_audio(text, filename):
    engine.save_to_file(text, filename)
    engine.runAndWait()

st.title("Run Manager Text and Audio Generator")

# Text input
user_input = st.text_area("Enter your text here:")

if st.button("Run"):
    if user_input.strip():
        # Run the manager function
        output_text = run_manager(user_input)
        
        # Display the output text
        st.subheader("Generated Text")
        st.write(output_text)
        
        # Save audio file
        save_audio(output_text, AUDIO_FILE)

        # Play audio file
        if os.path.exists(AUDIO_FILE):
            st.subheader("Generated Audio")
            audio_file = open(AUDIO_FILE, 'rb')
            st.audio(audio_file.read(), format='audio/mp3')
    else:
        st.warning("Please enter some input text.")
