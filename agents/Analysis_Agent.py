
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
def embed_chunks(chunks: list[str]) -> None:
    document =""
    for doc in chunks:
        document += doc
    llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=api_key,
)

    prompt_tamplet = """
    Provide intellience stock market reporrt on {document}
    
"""  
    response = llm.invoke(prompt_tamplet)
    return response.content

        