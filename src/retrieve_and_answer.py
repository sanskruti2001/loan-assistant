# src/retrieve_and_answer.py
import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = Path("indexes/faiss.index")
META_PATH = Path("indexes/meta.jsonl")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# Load FAISS and metadata
import faiss

if not INDEX_PATH.exists():
    raise SystemExit(f"FAISS index not found at {INDEX_PATH}. Run build_index.py first.")

index = faiss.read_index(str(INDEX_PATH))
metas = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]

# If OpenAI key present, create client once (new SDK)
openai_client = None
if OPENAI_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_KEY)


def embed_query_openai(text):
    """Use new OpenAI SDK to create embedding for the query"""
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    # resp.data is a list; take first embedding
    emb = np.array(resp.data[0].embedding).astype("float32")
    return emb


def embed_query_local(text):
    """Fallback to sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode([text])[0].astype("float32")
    return emb


def retrieve(query, top_k=5):
    """Return top_k metadata dicts for query."""
    if openai_client:
        qv = embed_query_openai(query)
    else:
        qv = embed_query_local(query)
    D, I = index.search(np.array([qv]), top_k)
    results = []
    for idx in I[0]:
        # defensive: if idx out of range
        if idx < 0 or idx >= len(metas):
            continue
        results.append(metas[idx])
    return results


def build_prompt(question, retrieved):
    snippets = []
    for i, r in enumerate(retrieved, 1):
        snippets.append(
            f"[SNIPPET {i}] (source: {r.get('source_url')}, fetched_at: {r.get('fetched_at', 'unknown')})\n{r.get('text')}"
        )
    context = "\n\n".join(snippets) if snippets else "No context available."
    prompt = f"""
You are a helpful assistant answering questions ONLY using the provided Bank of Maharashtra loan information.
If the answer cannot be found in the snippets, reply exactly: "I don't know — please check the Bank of Maharashtra website" and list the sources you checked.

Context snippets:
{context}

Question: {question}

Answer concisely (2-4 sentences). For each factual claim, include the source URL in parentheses.
"""
    return prompt


def call_llm(prompt):
    """Call OpenAI chat completion using new SDK. Returns text."""
    # make sure client exists
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY not set. Set it or run without generation to only see retrieved snippets.")
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0,
    )
    # In the new SDK, the generated content lives at resp.choices[0].message.content
    return resp.choices[0].message.content


def answer_question(question, top_k=5):
    retrieved = retrieve(question, top_k)
    # For prompt, keep top 3 snippets (to stay within token limits)
    prompt = build_prompt(question, retrieved[:3])
    if not openai_client:
        # No LLM available — return retrieved snippets only
        return {"answer": None, "retrieved": retrieved, "note": "OPENAI_API_KEY not set — only retrieval performed."}
    answer_text = call_llm(prompt)
    return {"answer": answer_text, "retrieved": retrieved}


if __name__ == "__main__":
    q = input("Question: ").strip()
    if not q:
        print("No question provided.")
        raise SystemExit(1)
    out = answer_question(q, top_k=5)
    print("\n--- Retrieved sources (top results) ---")
    for r in out["retrieved"]:
        print("-", r.get("source_url"), f"(chunk_id: {r.get('chunk_id','?')})")
    print("\n--- Answer ---")
    print(out["answer"] or "No LLM answer (see note).")
    if out.get("note"):
        print("\nNote:", out["note"])
