import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.API_Agent import run_financial_assistant
from agents.scrapping_Agent import get_relevant_articles_from_prompt

# Global data list
all_data = []

def all_data_collection(prompt: str) -> list:
    print("[DEBUG] Step 1: Calling run_financial_assistant...")
    data = run_financial_assistant(prompt)
    print(f"[DEBUG] Step 2: Retrieved {len(data)} items from financial assistant.")

    # Split data into 5 parts
    length = len(data)
    part_size = length // 5
    parts = [data[i * part_size : (i + 1) * part_size] for i in range(4)]
    parts.append(data[4 * part_size:])
    print(f"[DEBUG] Step 3: Data split into 5 parts: {[len(p) for p in parts]}")

    all_data.extend(parts)
    print(f"[DEBUG] Step 4: Added split parts to all_data. Total length now: {len(all_data)}")

    # Get articles
    print("[DEBUG] Step 5: Calling get_relevant_articles_from_prompt...")
    articles = get_relevant_articles_from_prompt(prompt)
    print(f"[DEBUG] Step 6: Retrieved {len(articles)} articles.")

    all_data.extend(articles)
    print(f"[DEBUG] Step 7: Final all_data length after adding articles: {len(all_data)}")

    return all_data

if __name__ == "__main__":
    prompt = "Stock market analysis of Microsoft and Apple"
    print(f"[DEBUG] Starting all_data_collection for prompt: {prompt}")
    collected_data = all_data_collection(prompt)
    print(f"[DEBUG] Final collected data: {collected_data}")
