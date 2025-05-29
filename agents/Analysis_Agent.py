
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
def Analysis(chunks: list[str]) -> None:
    document =""
    for doc in chunks:
        document += doc
    llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=api_key,
)

    prompt_tamplet = """
    Provide intelligent stock market reporrt on {document}
    
"""  
    response = llm.invoke(prompt_tamplet)
    return response.content




if __name__ == "__main__":
    # Example chunks list you want to analyze
    example_chunks = [
        "Stock A had a strong quarter with revenue up 20%. ",
        "Stock B showed a decline due to regulatory issues. ",
        "The overall market is volatile due to geopolitical tensions."
    ]
    analysis_result = Analysis(example_chunks)
    print("Analysis Result:\n", analysis_result)