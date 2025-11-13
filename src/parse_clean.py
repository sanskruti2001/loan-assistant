import json, glob
from pathlib import Path

# Define output path for the cleaned knowledge base file
OUT = Path("data/kb.jsonl")

# Ensure the output directory exists
OUT.parent.mkdir(parents=True, exist_ok=True)

def simple_clean(text):
    """
    Clean text by removing extra whitespace and empty lines.
    
    Args:
        text (str): The raw input text.
    
    Returns:
        str: Cleaned text with trimmed and joined lines.
    """
    # Remove blank lines and strip whitespace from each line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def main():
    """
    Read raw JSON files from 'data/raw', clean their text,
    and combine them into a single JSONL knowledge base file.
    """
    # Find all JSON files in data/raw directory
    files = glob.glob("data/raw/*.json")
    cnt = 0  # Counter for processed documents

    # Open the output JSONL file for writing
    with open(OUT, "w", encoding="utf-8") as outf:
        for f in files:
            # Load each raw JSON document
            j = json.load(open(f, encoding="utf-8"))
            
            # Clean the 'raw_text' field
            text = simple_clean(j.get("raw_text", ""))
            
            # Build a standardized record for each document
            rec = {
                "id": f"doc_{cnt}",
                "title": (text.splitlines()[0] if text else ""),  # Use first line as title
                "text": text,
                "source_url": j.get("url"),
                "fetched_at": j.get("fetched_at")
            }

            # Write record as a single JSON line
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt += 1  # Increment document counter

    # Print completion message with count
    print("Wrote", cnt, "documents to", OUT)

# Entry point of the script
if __name__ == "__main__":
    main()
