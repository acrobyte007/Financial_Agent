import os
from dotenv import load_dotenv
import whisper
from gtts import gTTS

load_dotenv()

def voice_to_text(audio_path: str) -> str:
    model = whisper.load_model("base")
    return model.transcribe(audio_path)["text"]

def text_to_voice(text: str, output_path: str = "./output/output_speech.mp3") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)
    return output_path

if __name__ == "__main__":
    audio_file = "input.wav"  # Input audio file
    llm_response = "This is the LLM response."  # Replace with real response

    transcribed = voice_to_text(audio_file)
    audio_out = text_to_voice(llm_response)

    print(f"Transcribed Text: {transcribed}")
    print(f"Response Audio File: {audio_out}")
