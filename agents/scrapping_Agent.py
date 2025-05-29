from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")  # Fallback to 'demo'

# Validate API key
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

# Define Pydantic model for URL variables
class URL_variables(BaseModel):
    """Variables for constructing Alpha Vantage API URLs"""
    Tickers: List[str] = Field(..., description="List of stock market index tickers (e.g., ASIA50, HSI, KOSPI)")
    Time_From: str = Field(..., description="Start time in YYYYMMDDTHHMM format, e.g., 20250429T0000")

# Define the Groq LLM with tools
groq_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=api_key,
)

llm_with_tools = groq_model.bind_tools([URL_variables])

# Define a function to calculate a default time_from (last 1 month in UTC)
def get_default_time_from() -> str:
    """Returns start date for the last 1 month in YYYYMMDDTHHMM format (UTC)"""
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    return start_date.strftime("%Y%m%dT%H%M")

# Define a function to parse and validate time_from
def parse_time_from(time_from: Optional[str]) -> str:
    """Validates and formats time_from; returns default if invalid or outdated"""
    if not time_from:
        return get_default_time_from()
    try:
        dt = datetime.strptime(time_from, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        if dt.year < 2024:  # Prevent outdated dates
            return get_default_time_from()
        return time_from
    except (ValueError, TypeError):
        return get_default_time_from()

# Define the main function to generate Alpha Vantage URL from prompt
def generate_alpha_vantage_url_from_prompt(prompt: str) -> str:
    """
    Generates an Alpha Vantage API URL for news sentiment data based on a user prompt.
    
    Args:
        prompt (str): User prompt specifying tickers and time_from (e.g., "Provide stock markets data for ASIA50 and HSI from 20250501T0000").
    
    Returns:
        str: The constructed API URL in the format:
             https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=...&time_from=...&limit=1000&apikey=...
    
    Raises:
        ValueError: If no valid tickers are extracted from the prompt.
    """
    # Define the LLM prompt template
    llm_prompt = f"""You are a financial assistant tasked with providing data for stock market indices. Follow these instructions:

1. Extract a list of valid ticker symbols from the user prompt. If none are specified, use default Asian stock market indices: ASIA50, HSI, KOSPI, STI, TAIEX.
2. Extract a time_from in YYYYMMDDTHHMM format (e.g., 20250429T0000). If none is provided, use the last 1 month from today (in UTC).
3. Ensure time_from is recent (not older than 2024).
4. Do not include invalid or unrelated tickers.
5. Output the variables using the URL_variables tool.

User prompt: {prompt}

Output the variables using the URL_variables tool.
"""

    # Invoke the LLM to extract tickers and time_from
    try:
        response = llm_with_tools.invoke(llm_prompt).tool_calls
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        # Fallback to default values
        response = [{
            "args": {
                "Tickers": ["ASIA50", "HSI", "KOSPI", "STI", "TAIEX"],
                "Time_From": get_default_time_from()
            }
        }]

    # Process the response
    if not response:
        raise ValueError("No valid response from the model")

    args = response[0]["args"]
    
    # Extract and validate tickers
    tickers = args.get("Tickers")
    if not tickers or not all(isinstance(t, str) and t.strip() for t in tickers):
        raise ValueError("No valid tickers provided")

    # Parse and validate time_from
    time_from = parse_time_from(args.get("Time_From"))

    # Construct URL
    base_url = "https://www.alphavantage.co/query"
    function = "NEWS_SENTIMENT"
    limit = 10
    tickers_str = ",".join(tickers)  # Combine tickers into a comma-separated string
    url = f"{base_url}?function={function}&tickers={tickers_str}&time_from={time_from}&limit={limit}&apikey={alpha_vantage_api_key}"
    
    return url





from bs4 import BeautifulSoup

def scrape_article_content(url):
    try:
        # Send HTTP request to the URL
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article content (e.g., text within <p> tags)
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

        return article_text if article_text else "No article content found."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"
    

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def get_relevant_articles_from_prompt(prompt):
    url = generate_alpha_vantage_url_from_prompt(prompt)
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: Failed to fetch data. Status code: {response.status_code}")
        return []

    try:
        data = response.json()
    except ValueError:
        print("Error: Invalid JSON response")
        return []

    feed = data.get('feed', [])
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for item in feed:
        ticker_sentiment = item.get('ticker_sentiment', [])
        if any(float(ts['relevance_score']) >= 0.5 for ts in ticker_sentiment):
            url = item.get('url', '')
            article_content = scrape_article_content(url)
            if article_content:
                chunks = splitter.create_documents([article_content])
                all_docs.extend(chunks)

    return all_docs



if __name__ == "__main__":
    prompt = "Provide stock markets data of microsoft and apple"
    docs = get_relevant_articles_from_prompt(prompt)
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:\n{doc.page_content[:300]}...\n")
