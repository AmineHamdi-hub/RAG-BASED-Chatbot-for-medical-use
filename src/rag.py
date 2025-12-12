from pathlib import Path
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from src.llm_wrapper import GroqLLM
import os

llm = GroqLLM()
DOCS_DIR = Path("data/docs")

# --------------------------
# Load docs from folder
# --------------------------
def load_docs_from_folder():
    docs = []
    for file_path in DOCS_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

# --------------------------
# Simple chunking
# --------------------------
def ingest_documents(docs, chunk_size=500):
    chunks = []
    for doc in docs:
        words = doc.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
    return chunks

# --------------------------
# Store embeddings in Postgres
# --------------------------
def store_embeddings(chunks):
    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = [embeddings_model.encode(c).tolist() for c in chunks]

    conn = psycopg2.connect(os.getenv("POSTGRES_URI"))
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS medical_docs (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(384)
        )
    """)
    for chunk, vector in zip(chunks, vectors):
        cur.execute(
            "INSERT INTO medical_docs (content, embedding) VALUES (%s, %s)",
            (chunk, vector)
        )
    conn.commit()
    cur.close()
    conn.close()

# --------------------------
# Simple RAG query
# --------------------------
def retrieve_similar(query, top_k=3):
    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embeddings_model.encode(query).tolist()

    conn = psycopg2.connect(os.getenv("POSTGRES_URI"))
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("""
    SELECT content 
    FROM medical_docs
    ORDER BY embedding <=> %s::vector
    LIMIT %s
    """, (query_vector, top_k))
    results = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return results

def answer_question(query):
    context = "\n".join(retrieve_similar(query))
    prompt = (
        "You are a careful medical assistant. Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}"
    )
    return llm(prompt)
