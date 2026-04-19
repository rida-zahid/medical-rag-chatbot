import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

DB_DIR = "db"
TOP_K = 5

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(f"{DB_DIR}/index.faiss")

with open(f"{DB_DIR}/chunks.pkl", "rb") as f:
    chunks, sources = pickle.load(f)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def retrieve(question: str, top_k: int = TOP_K) -> list[str]:
    q_vec = embed_model.encode([question]).astype("float32")
    _, ids = index.search(q_vec, top_k)
    results = []
    for i in ids[0]:
        i = int(i)
        if i != -1 and i < len(chunks):
            results.append(chunks[i])
    return results


def ask(question: str) -> str:
    try:
        context_chunks = retrieve(question)
        if not context_chunks:
            return "No relevant medical information found in the database."

        context = "\n\n".join(context_chunks)[:1500]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical assistant specializing in dermatology and skin cancer research. "
                        "Answer in 3-4 lines only. Be concise and to the point. "
                        "Mention specific names and types where relevant."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=150,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"