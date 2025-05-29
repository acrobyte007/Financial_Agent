from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

def language(chunks: str) -> str:
    # Retrieve the Groq API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    # Validate input
    if not chunks or not isinstance(chunks, str):
        raise ValueError("Input chunks must be a non-empty string.")

    # Initialize the Groq LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=api_key,
    )

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["chunks"],
        template="""
        You are a financial analyst. Provide an intelligent and concise stock market report based on the following data:

        {chunks}

        The report should summarize key insights, trends, or actionable information in a clear and professional manner.
        """
    )

    # Generate the report using the LLM
    try:
        response = llm.invoke(prompt_template.format(chunks=chunks))
        return response.content
    except Exception as e:
        raise Exception(f"Error generating report: {str(e)}")

# Example usage
if __name__ == "__main__":
    sample_chunks = """
    Company X reported a 10% increase in Q2 revenue. Stock price rose 5% after the earnings report.
    Recent market trends show increased volatility in the tech sector.
    """
    try:
        report = language(sample_chunks)
        print("Generated Stock Market Report:")
        print(report)
    except Exception as e:
        print(f"Error: {str(e)}")