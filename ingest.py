import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_DIR = "data"
DB_DIR = "db"

CHUNK_SIZE = 500
OVERLAP = 50


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


def build_index():
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    sources = []

    print(f"Reading PDFs from '{PDF_DIR}/'...")
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(PDF_DIR, file)
            text = extract_text(path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            sources.extend([file] * len(chunks))
            print(f"  {file} → {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("Creating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(DB_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(DB_DIR, "index.faiss"))

    with open(os.path.join(DB_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump((all_chunks, sources), f)

    print("\n Done! Index saved to db/")


if __name__ == "__main__":
    build_index()