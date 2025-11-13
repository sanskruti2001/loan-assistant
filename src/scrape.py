# src/scrape.py
import requests, time, json, os
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent":"LoanAssistantBot/1.0 (+contact)"}


URLS = [
    "https://bankofmaharashtra.bank.in/personal-banking/loans/home-loan",
    "https://bankofmaharashtra.bank.in/educational-loans",
    "https://bankofmaharashtra.bank.in/personal-banking/loans/personal-loan"
]

def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text

def extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")
    # remove noisy tags
    for tag in soup(["script","style","header","footer","nav","aside","noscript"]):
        tag.decompose()
    # Heuristic: get content from main article or container tags
    main = soup.find("main") or soup.find("article") or soup
    text = main.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def main():
    meta = []
    for url in URLS:
        try:
            print("Fetching:", url)
            html = fetch(url)
            text = extract_main_text(html)
            slug = url.replace("https://","").replace("/","_")[:120]
            out_file = OUT / f"{slug}.json"
            rec = {"url": url, "fetched_at": datetime.utcnow().isoformat()+"Z", "raw_html": html, "raw_text": text}
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            meta.append(str(out_file))
            time.sleep(1.0)
        except Exception as e:
            print("Error for", url, e)
    print("Saved files:", meta)

if __name__ == "__main__":
    main()
