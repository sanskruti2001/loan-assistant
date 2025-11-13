import json, faiss, numpy as np
from pathlib import Path

IN = Path("data/embeddings.jsonl")
INDEX_PATH = Path("indexes/faiss.index")
META_PATH = Path("indexes/meta.jsonl")
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    embeddings = []
    metas = []
    for line in open(IN, encoding="utf-8"):
        r = json.loads(line)
        embeddings.append(r["embedding"])
        metas.append({"chunk_id": r["chunk_id"], "source_url": r["source_url"], "text": r["text"], "doc_id": r["doc_id"], "fetched_at": r.get("fetched_at")})
    vecs = np.array(embeddings).astype('float32')
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)  # simple exact index (lightweight)
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("Saved FAISS index and meta.")

if __name__ == "__main__":
    main()
