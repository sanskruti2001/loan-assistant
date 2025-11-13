# src/scrape.py
import requests, time, json, os
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

# Directory where raw scraped data will be stored
OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

# Custom user-agent to identify the scraping bot
HEADERS = {"User-Agent": "LoanAssistantBot/1.0 (+contact)"}

# List of Bank of Maharashtra loan-related URLs to scrape
URLS = [
    "https://bankofmaharashtra.bank.in/personal-banking/loans/home-loan",
    "https://bankofmaharashtra.bank.in/educational-loans",
    "https://bankofmaharashtra.bank.in/personal-banking/loans/personal-loan"
]

def fetch(url):
    """
    Send an HTTP GET request and return the HTML content.
    
    Args:
        url (str): Target webpage URL.
    
    Returns:
        str: Raw HTML content of the page.
    """
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()  # Raise an error if the request fails
    return r.text

def extract_main_text(html):
    """
    Extract the main textual content from an HTML page by removing unwanted tags.
    
    Args:
        html (str): Raw HTML content.
    
    Returns:
        str: Cleaned text extracted from the HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content or noisy HTML elements
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        tag.decompose()

    # Heuristically select the main content section (if available)
    main = soup.find("main") or soup.find("article") or soup

    # Extract visible text and clean up whitespace
    text = main.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def main():
    """
    Scrape loan-related pages from the Bank of Maharashtra website.
    Extracts clean text and saves both raw HTML and text to JSON files.
    """
    meta = []  # Track saved file paths for logging

    for url in URLS:
        try:
            print("Fetching:", url)

            # Fetch HTML content
            html = fetch(url)

            # Extract readable text
            text = extract_main_text(html)

            # Create a safe filename based on the URL
            slug = url.replace("https://", "").replace("/", "_")[:120]
            out_file = OUT / f"{slug}.json"

            # Build a structured record with metadata
            rec = {
                "url": url,
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "raw_html": html,
                "raw_text": text
            }

            # Save the record as a JSON file
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

            # Add path to metadata log
            meta.append(str(out_file))

            # Be polite — add a short delay between requests
            time.sleep(1.0)

        except Exception as e:
            # Handle network or parsing errors gracefully
            print("Error for", url, e)

    # Print summary of saved files
    print("Saved files:", meta)

# Entry point for standalone execution
if __name__ == "__main__":
    main()
