import json, faiss, numpy as np
from pathlib import Path

# Define file paths for input embeddings and output index/meta files
IN = Path("data/embeddings.jsonl")
INDEX_PATH = Path("indexes/faiss.index")
META_PATH = Path("indexes/meta.jsonl")

# Ensure the 'indexes' directory exists
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    # Lists to store embeddings and metadata
    embeddings = []
    metas = []
    
    # Read embeddings and metadata line by line from JSONL file
    for line in open(IN, encoding="utf-8"):
        r = json.loads(line)
        embeddings.append(r["embedding"])
        metas.append({
            "chunk_id": r["chunk_id"],
            "source_url": r["source_url"],
            "text": r["text"],
            "doc_id": r["doc_id"],
            "fetched_at": r.get("fetched_at")
        })
    
    # Convert list of embeddings to a NumPy float32 array
    vecs = np.array(embeddings).astype('float32')
    dim = vecs.shape[1]  # Determine vector dimensionality
    
    # Create a simple FAISS L2 (Euclidean) index for exact search
    index = faiss.IndexFlatL2(dim)
    
    # Add all vectors to the index
    index.add(vecs)
    
    # Save the FAISS index to disk
    faiss.write_index(index, str(INDEX_PATH))
    
    # Save corresponding metadata to a JSONL file
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    
    # Print confirmation message after successful save
    print("Saved FAISS index and meta.")

# Entry point for script execution
if __name__ == "__main__":
    main()
