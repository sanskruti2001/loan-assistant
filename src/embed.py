import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and model name from environment variables
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Define input and output file paths
IN = Path("data/chunks.jsonl")
OUT = Path("data/embeddings.jsonl")

# Ensure output directory exists
OUT.parent.mkdir(parents=True, exist_ok=True)

def embed_openai(texts):
    """
    Generate embeddings using the OpenAI API.
    
    Args:
        texts (list[str]): List of text strings to embed.
    
    Returns:
        list[list[float]]: List of embedding vectors.
    """
    # Initialize OpenAI client with provided API key
    client = OpenAI(api_key=OPENAI_KEY)

    # Request embeddings for all input texts
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)

    # Extract embedding vectors from response
    return [r.embedding for r in resp.data]


def embed_local(texts):
    """
    Generate embeddings locally using SentenceTransformer.
    Used as a fallback when no OpenAI API key is available.
    
    Args:
        texts (list[str]): List of text strings to embed.
    
    Returns:
        list[list[float]]: List of embedding vectors.
    """
    from sentence_transformers import SentenceTransformer

    # Load a lightweight local embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode texts into embeddings
    embs = model.encode(texts, show_progress_bar=True)

    # Convert embeddings to standard Python lists
    return [emb.tolist() for emb in embs]


def main():
    """
    Main function to read text chunks, generate embeddings (OpenAI or local),
    and write them to a JSONL output file.
    """
    texts = []  # Stores text data for embedding
    recs = []   # Stores original records for later merging

    # Read all text chunks from input file
    with open(IN, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs.append(r)
            texts.append(r["text"])

    # Decide which embedding method to use
    if OPENAI_KEY:
        print("Using OpenAI embeddings...")
        embs = embed_openai(texts)
    else:
        print("Using local sentence-transformers embeddings...")
        embs = embed_local(texts)

    # Write output file with original metadata + embeddings
    with open(OUT, "w", encoding="utf-8") as outf:
        for r, e in zip(recs, embs):
            outrec = {**r, "embedding": e}
            outf.write(json.dumps(outrec, ensure_ascii=False) + "\n")

    # Print success message
    print("✅ Wrote embeddings to", OUT)


# Entry point of the script
if __name__ == "__main__":
    main()
