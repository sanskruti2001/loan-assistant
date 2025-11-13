import json
from pathlib import Path
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

IN = Path("data/kb.jsonl")
OUT = Path("data/chunks.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

SENTS_PER_CHUNK = 6
OVERLAP = 1

def chunk_text(text, sents_per_chunk=SENTS_PER_CHUNK, overlap=OVERLAP):
    sents = sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sents):
        chunk_sents = sents[i:i+sents_per_chunk]
        chunks.append(" ".join(chunk_sents))
        i += sents_per_chunk - overlap
    return chunks

def main():
    total = 0
    with open(IN, encoding="utf-8") as inf, open(OUT, "w", encoding="utf-8") as outf:
        for line in inf:
            doc = json.loads(line)
            chs = chunk_text(doc["text"])
            for idx, ch in enumerate(chs):
                rec = {
                    "chunk_id": f"{doc['id']}_c{idx}",
                    "doc_id": doc["id"],
                    "text": ch,
                    "source_url": doc["source_url"],
                    "fetched_at": doc.get("fetched_at")
                }
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
    print("Wrote", total, "chunks to", OUT)

if __name__ == "__main__":
    main()
