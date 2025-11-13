import json
from pathlib import Path
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Define input and output file paths
IN = Path("data/kb.jsonl")
OUT = Path("data/chunks.jsonl")

# Ensure output directory exists
OUT.parent.mkdir(parents=True, exist_ok=True)

# Parameters for chunking text
SENTS_PER_CHUNK = 6  # Number of sentences per chunk
OVERLAP = 1          # Number of overlapping sentences between chunks

def chunk_text(text, sents_per_chunk=SENTS_PER_CHUNK, overlap=OVERLAP):
    """
    Split the given text into overlapping chunks based on sentence count.
    
    Args:
        text (str): The full text to be chunked.
        sents_per_chunk (int): Number of sentences per chunk.
        overlap (int): Number of overlapping sentences between consecutive chunks.
    
    Returns:
        list: A list of text chunks.
    """
    # Tokenize text into individual sentences
    sents = sent_tokenize(text)
    chunks = []
    i = 0

    # Create chunks with specified overlap
    while i < len(sents):
        chunk_sents = sents[i:i+sents_per_chunk]
        chunks.append(" ".join(chunk_sents))
        i += sents_per_chunk - overlap  # Move index forward with overlap considered
    return chunks

def main():
    """
    Read documents from kb.jsonl, split them into chunks, 
    and write the chunks to chunks.jsonl.
    """
    total = 0  # Counter for total number of chunks written
    
    # Open input and output files
    with open(IN, encoding="utf-8") as inf, open(OUT, "w", encoding="utf-8") as outf:
        for line in inf:
            # Load each document (JSON object)
            doc = json.loads(line)

            # Chunk the document text into smaller parts
            chs = chunk_text(doc["text"])

            # Write each chunk as a separate JSON record
            for idx, ch in enumerate(chs):
                rec = {
                    "chunk_id": f"{doc['id']}_c{idx}",
                    "doc_id": doc["id"],
                    "text": ch,
                    "source_url": doc["source_url"],
                    "fetched_at": doc.get("fetched_at")
                }
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1  # Increment chunk count

    # Print summary message
    print("Wrote", total, "chunks to", OUT)

# Entry point of the script
if __name__ == "__main__":
    main()
