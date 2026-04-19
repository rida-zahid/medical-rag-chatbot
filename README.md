# 🔬 Medical RAG Chatbot — Skin Cancer Research

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions **grounded in real skin cancer research papers**. Instead of relying on an LLM's parametric memory, the system retrieves the most relevant passages from ingested PDFs and feeds them as context to the model — dramatically reducing hallucinations.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-vector--search-orange)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-purple)

---

## 🎥 Demo

[▶ Watch the chatbot in action on LinkedIn](https://www.linkedin.com/posts/rida-zahid-382730309_python-machinelearning-ai-ugcPost-7451744601670926336-Omoc?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE6YmRIBaC1HcMJou3ai5oJl0Ay8NYbm7-I)

---

## 🧠 How It Works

```
User Question
     │
     ▼
Sentence Transformer  ──►  Query Embedding
     │
     ▼
FAISS Index  ──►  Top-K Relevant Chunks  (from research PDFs)
     │
     ▼
Groq LLM (Llama 3.3 70B)  ──►  Grounded Answer
```

1. **Ingest** — PDFs are chunked, embedded with `all-MiniLM-L6-v2`, and stored in a FAISS flat index.
2. **Retrieve** — At query time, the question is embedded and the nearest chunks are fetched via L2 search.
3. **Generate** — The retrieved context is passed to Llama 3.3 70B on Groq for a concise, grounded answer.

---

## 🗂️ Project Structure

```
medical-rag-chatbot/
├── data/               # Place your PDF research papers here (git-ignored)
├── db/                 # Auto-generated FAISS index & chunk store (git-ignored)
├── ingest.py           # PDF → chunks → embeddings → FAISS index
├── query.py            # Retrieval + Groq LLM inference
├── app.py              # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/rida-zahid/medical-rag-chatbot.git
cd medical-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key

```bash
cp .env.example .env
# Edit .env and paste your key from https://console.groq.com
```

Then export it before running:

```bash
export GROQ_API_KEY=your_key_here   # Windows: set GROQ_API_KEY=your_key_here
```

### 5. Add your PDFs

Drop your skin cancer research paper PDFs into the `data/` folder.

### 6. Build the vector index

```bash
python ingest.py
```

### 7. Run the app

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Vector Search | [FAISS](https://github.com/facebookresearch/faiss) |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` |
| LLM | [Groq](https://groq.com/) — Llama 3.3 70B Versatile |
| PDF Parsing | [pypdf](https://pypdf.readthedocs.io/) |
| UI | [Streamlit](https://streamlit.io/) |
| Language | Python 3.10+ |

---

## 🔒 Security Note

Never commit API keys. This project reads `GROQ_API_KEY` from the environment. The `.env` file is git-ignored — use `.env.example` as a template.

---

## 🚀 Future Improvements

- [ ] Add source citations alongside answers
- [ ] Support multiple document collections / specialties
- [ ] Hybrid BM25 + dense retrieval for better recall
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces
- [ ] Streaming responses from the LLM

---

## 👤 Author

Built by **Rida Zahid** — learning ML/AI in public.  
[LinkedIn](https://www.linkedin.com/in/rida-zahid-382730309/) · Feel free to open issues or PRs!
