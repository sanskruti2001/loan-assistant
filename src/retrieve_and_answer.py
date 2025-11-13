# src/retrieve_and_answer.py

import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths to FAISS index and metadata files
INDEX_PATH = Path("indexes/faiss.index")
META_PATH = Path("indexes/meta.jsonl")

# Load configuration from environment variables
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# Import FAISS for vector similarity search
import faiss

# Ensure FAISS index exists before proceeding
if not INDEX_PATH.exists():
    raise SystemExit(f"FAISS index not found at {INDEX_PATH}. Run build_index.py first.")

# Load FAISS index and metadata (list of document chunks)
index = faiss.read_index(str(INDEX_PATH))
metas = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]

# Initialize OpenAI client (if API key is available)
openai_client = None
if OPENAI_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_KEY)


def embed_query_openai(text):
    """
    Create an embedding vector for a query using OpenAI's embedding model.
    
    Args:
        text (str): The query text.
    
    Returns:
        np.ndarray: 1D NumPy array representing the query embedding.
    """
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    # Extract the first embedding from the response
    emb = np.array(resp.data[0].embedding).astype("float32")
    return emb


def embed_query_local(text):
    """
    Create an embedding vector for a query using a local model (SentenceTransformer).
    Used as a fallback when no OpenAI key is set.
    
    Args:
        text (str): The query text.
    
    Returns:
        np.ndarray: 1D NumPy array representing the query embedding.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode([text])[0].astype("float32")
    return emb


def retrieve(query, top_k=5):
    """
    Retrieve the top_k most relevant text chunks for a given query.
    
    Args:
        query (str): The user question or search text.
        top_k (int): Number of most similar results to return.
    
    Returns:
        list[dict]: Metadata for the top_k retrieved chunks.
    """
    # Generate embedding for the query (OpenAI or local)
    if openai_client:
        qv = embed_query_openai(query)
    else:
        qv = embed_query_local(query)
    
    # Perform similarity search on FAISS index
    D, I = index.search(np.array([qv]), top_k)
    
    results = []
    for idx in I[0]:
        # Skip invalid indexes
        if idx < 0 or idx >= len(metas):
            continue
        results.append(metas[idx])
    return results


def build_prompt(question, retrieved):
    """
    Construct a prompt that combines retrieved snippets and the user's question.
    This prompt will be sent to the LLM for answering.
    
    Args:
        question (str): The user's question.
        retrieved (list[dict]): List of retrieved document chunks.
    
    Returns:
        str: Formatted prompt text.
    """
    snippets = []
    for i, r in enumerate(retrieved, 1):
        snippets.append(
            f"[SNIPPET {i}] (source: {r.get('source_url')}, fetched_at: {r.get('fetched_at', 'unknown')})\n{r.get('text')}"
        )
    # Join all snippets with double newlines
    context = "\n\n".join(snippets) if snippets else "No context available."

    # Instructional prompt for the model
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
    """
    Send a prompt to the OpenAI Chat Completion API and return the model's response.
    
    Args:
        prompt (str): The prompt text containing context and question.
    
    Returns:
        str: Model-generated answer.
    """
    if not openai_client:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Set it or run without generation to only see retrieved snippets."
        )

    # Call the LLM model using the latest OpenAI SDK
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0,
    )

    # Extract model output text
    return resp.choices[0].message.content


def answer_question(question, top_k=5):
    """
    Retrieve relevant snippets for a question and generate an answer using the LLM.
    
    Args:
        question (str): User's question.
        top_k (int): Number of top snippets to retrieve.
    
    Returns:
        dict: Dictionary containing the final answer, retrieved snippets, and optional note.
    """
    retrieved = retrieve(question, top_k)
    
    # Use top 3 snippets to keep prompt short
    prompt = build_prompt(question, retrieved[:3])

    # If OpenAI key is missing, return retrieved snippets only
    if not openai_client:
        return {
            "answer": None,
            "retrieved": retrieved,
            "note": "OPENAI_API_KEY not set — only retrieval performed."
        }

    # Otherwise, generate an answer using the LLM
    answer_text = call_llm(prompt)
    return {"answer": answer_text, "retrieved": retrieved}


if __name__ == "__main__":
    # Allow interactive testing from the command line
    q = input("Question: ").strip()
    if not q:
        print("No question provided.")
        raise SystemExit(1)

    out = answer_question(q, top_k=5)

    # Display retrieved sources
    print("\n--- Retrieved sources (top results) ---")
    for r in out["retrieved"]:
        print("-", r.get("source_url"), f"(chunk_id: {r.get('chunk_id','?')})")

    # Display generated answer (if any)
    print("\n--- Answer ---")
    print(out["answer"] or "No LLM answer (see note).")

    # Display note (if applicable)
    if out.get("note"):
        print("\nNote:", out["note"])
