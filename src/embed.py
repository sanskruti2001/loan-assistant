import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

IN = Path("data/chunks.jsonl")
OUT = Path("data/embeddings.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)


def embed_openai(texts):
    """Create embeddings using the latest OpenAI SDK."""
    client = OpenAI(api_key=OPENAI_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in resp.data]


def embed_local(texts):
    """Fallback to local embeddings if no OpenAI key found."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode(texts, show_progress_bar=True)
    return [emb.tolist() for emb in embs]


def main():
    texts = []
    recs = []

    # Read input JSONL
    with open(IN, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs.append(r)
            texts.append(r["text"])

    # Use OpenAI or local embeddings
    if OPENAI_KEY:
        print("Using OpenAI embeddings...")
        embs = embed_openai(texts)
    else:
        print("Using local sentence-transformers embeddings...")
        embs = embed_local(texts)

    # Write output with embeddings
    with open(OUT, "w", encoding="utf-8") as outf:
        for r, e in zip(recs, embs):
            outrec = {**r, "embedding": e}
            outf.write(json.dumps(outrec, ensure_ascii=False) + "\n")

    print("✅ Wrote embeddings to", OUT)


if __name__ == "__main__":
    main()
