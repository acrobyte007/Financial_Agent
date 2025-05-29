import os
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS

load_dotenv()

def voice_to_text(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"API request error: {e}"

def text_to_voice(text: str, output_path: str = "./output/output_speech.mp3") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)
    return output_path

if __name__ == "__main__":
    audio_file = "sample.wav"  # Use a WAV file here
    llm_response = "This is the LLM response."  # Replace with real response

    transcribed = voice_to_text(audio_file)
    audio_out = text_to_voice(llm_response)

    print(f"Transcribed Text: {transcribed}")
    print(f"Response Audio File: {audio_out}")
