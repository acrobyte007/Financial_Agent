from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

@tool
def yahoo_finance_news(query: str) -> List[str]:
    """
    Fetches the latest Yahoo Finance news for a given stock symbol or company name and splits the content into chunks.
    Example input: "MSFT" or "Microsoft"
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Fetch news using the Yahoo Finance tool
    news_content = YahooFinanceNewsTool().run(query)
    print("Fetched news content is called")
    # Split the news content into chunks
    try:
        chunks = text_splitter.split_text(news_content)
        return chunks if chunks else [news_content]  # Return original if splitting fails
    except Exception as e:
        return [f"Error splitting news content: {str(e)}"]

class FinanceData(BaseModel):
    tickers: List[str]
    time_range: str  

@tool
def get_data(request: dict) -> Dict[str, Any]:
    
    """
    Retrieves historical stock data, current price, and options data for given tickers and time range.
    Input format: {"tickers": ["MSFT", "AAPL"], "time_range": "1mo"}
    """
    print("get_data called")
    if not isinstance(request, dict):
        return {"error": "Input must be a dict with 'tickers' and 'time_range'."}
    
    try:
        finance_request = FinanceData(**request)
    except Exception as e:
        return {"error": f"Invalid input: {str(e)}"}

    data = {}
    for symbol in finance_request.tickers:
        ticker = yf.Ticker(symbol)
        
        # Historical data
        try:
            hist = ticker.history(period=finance_request.time_range)
            hist_data = hist.to_dict()
        except Exception as e:
            hist_data = {"error": str(e)}

        # Current price
        try:
            current_price = ticker.fast_info.get("last_price") or ticker.info.get("regularMarketPrice", "N/A")
        except Exception as e:
            current_price = f"Error: {str(e)}"

        # Options data
        try:
            if ticker.options:
                expiry = ticker.options[0]
                chain = ticker.option_chain(expiry)
                options_data = {
                    "expiration_date": expiry,
                    "calls": chain.calls.to_dict(),
                    "puts": chain.puts.to_dict()
                }
            else:
                options_data = "No options data available"
        except Exception as e:
            options_data = f"Error: {str(e)}"

        data[symbol] = {
            "history": hist_data,
            "current_price": current_price,
            "options": options_data
        }
    print("Data fetched successfully")
    return data

# Define the Groq LLM
groq_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=api_key,
    max_retries=1
)

# Update the agent to include both tools
agent = create_react_agent(
    model=groq_model,
    tools=[get_data, yahoo_finance_news],  
    prompt="""You are a helpful financial assistant. When the user asks about a company or stock:
1.Use the yahoo_finance_news tool to fetch and summarize the latest news in concise chunks.
2.Use the get_data tool to retrieve real-time stock prices and historical performance (e.g., daily, weekly, monthly).
3.Present data in a readable, human-friendly format, including stock price trends, volume changes, and key metrics.
4.Clearly label the timeframe for historical data and highlight notable events or shifts.
5.Keep summaries clear, focused, and concise for quick understanding."""
)

def run_financial_assistant(user_prompt: str):
    # Modify the prompt inline with required instructions
    modified_prompt = f"""
You are a helpful financial assistant.

The user has asked: "{user_prompt}"

Instructions:
- In 500 words
- Use the yahoo_finance_news tool to fetch and summarize the latest related news in concise chunks.
- Use the get_data tool to retrieve both real-time and historical stock data (daily, weekly, monthly).
- Present stock prices, trends, volume, and key metrics in a clear, human-readable format.
- Highlight any important market events or sudden changes.
- Ensure the timeframe is mentioned when showing historical data.
- Keep responses clear, concise, and easy to interpret.
"""

    # Run the agent
    response = agent.invoke({
        "messages": [{"role": "user", "content": modified_prompt}]
    })

    # Output the final response
    return (response["messages"][-1].content)


if __name__ == "__main__":
    user_input = "What happened today with Microsoft stocks?"
    print(run_financial_assistant(user_input))
