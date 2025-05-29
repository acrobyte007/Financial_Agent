
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.Voice_Agent import voice_to_text,text_to_voice
from agents.Retriever_Agent import embed_chunks,get_chunks
from data_ingestion.data_collection import all_data_collection
from agents.Analysis_Agent import Analysis

def run_manager(user_prompt: str):
    text_prompt = voice_to_text(user_prompt)
    text_data=all_data_collection(text_prompt)
    embed_chunks(text_data)
    chunks = get_chunks(user_prompt)
    final=Analysis(chunks)
    audio = text_to_voice(final)
    return final,audio
