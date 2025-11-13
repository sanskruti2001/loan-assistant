Loan Product Assistant — RAG Chatbot (Bank of Maharashtra):
    - This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers user questions about Bank of Maharashtra loan products.
    - It scrapes official loan web pages, builds a searchable vector knowledge base using FAISS, and uses a language model (OpenAI GPT or local model) to generate context-aware, source-cited responses.

RAG Pipeline Overview:
    Web Pages -> Scraping -> Cleaning -> Chunking -> Embeddings -> FAISS Index
                                                    ↓
                                        Query Embedding + Retrieval
                                                    ↓
                                        LLM (GPT) + Final Answer

Environment Setup:

    1) Create and activate environment:
        - python3 -m venv venv
        - source venv/bin/activate

    2) Install dependencies:
        - pip install requirements.txt

    3) Environment variables
        - Create a file named .env in the project root:
            OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
            EMBED_MODEL=text-embedding-3-small
            LLM_MODEL=gpt-4o-mini
        - Add .env, venv/, data/raw/, and indexes/ to your .gitignore.


Step-by-Step Build Process:
    
    Step A — Scrape pages:
        - Fetch loan details pages using requests and save HTML + plain text.
        - python src/scrape.py   --> Output: data/raw/*.json

    Step B — Clean and consolidate:
        - Remove noise and create structured kb.jsonl.    
        - python src/parse_clean.py  --> Output: data/kb.jsonl

    Step C — Chunking:
        - Split text into small overlapping blocks (6 sentences per chunk, 1 overlap).
        - python src/chunk.py  --> Output: data/chunks.jsonl

    Step D — Embeddings:
        - Convert chunks to numeric vectors using either OpenAI embeddings or local sentence-transformers.
        - python src/embed.py  --> Output: data/embeddings.
        
    Step E — Build FAISS Index:
        - FAISS vector index and store parallel metadata.    
        - python src/build_index.py  --> Output: indexes/faiss.index and indexes/meta.jsonl
    
    Step F — Retrieval & Answer Generation:
        - Embed user query, search FAISS for top-K relevant chunks, and generate answer via GPT model.
        - python src/retrieve_and_answer.py
        - Then type a sample question such as: What are the interest rates for a Bank of Maharashtra home loan?   --> Outputs retrieved sources + final LLM answer.

    Step G — Simple FastAPI Web Demo: 
        - Run the chatbot locally in browser:
        - uvicorn src.app:app --reload
        - Then open http://localhost:8000 and enter any loan-related question.


RAG Pipeline Explanation: 

    This project follows a Retrieval-Augmented Generation (RAG) approach:
    It scrapes and cleans Bank of Maharashtra loan pages to build a document knowledge base.
    Each document is chunked and converted to vector embeddings stored in FAISS.
    When a user asks a question, the system computes its embedding, retrieves top-K relevant chunks, and passes them to a Large Language Model (LLM) such as GPT-4 for final response generation with citation to source URLs.