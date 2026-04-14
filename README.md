# ⚡ DocChat — RAG Chatbot (Bot is live at : https://8na5qx6dgi2bxi4tj4njzc.streamlit.app/)

> A lightning-fast **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload documents and have a conversation with them — powered by **Groq's high-speed inference**, **FastEmbed** embeddings, and a zero-dependency vector store.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-API-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📸 Preview

```
┌─────────────────────┬────────────────────────────────────────┐
│  SIDEBAR            │  MAIN CHAT                             │
│                     │                                        │
│  ⚡ DocChat         │  Chat with your Documents              │
│  RAG · Groq         │                                        │
│                     │  ┌──────────────────────────────────┐  │
│  MODEL              │  │ 👤 You                           │  │
│  Llama 3.1 8B ▼    │  │ What is this document about?    │  │
│                     │  └──────────────────────────────────┘  │
│  UPLOAD DOCUMENTS   │  ┌──────────────────────────────────┐  │
│  [ Drop PDF / TXT ] │  │ ⚡ DocChat                       │  │
│                     │  │ This document covers...         │  │
│  ⚡ Process Docs    │  │ 📄 report.pdf                   │  │
│                     │  └──────────────────────────────────┘  │
│  ● Vector DB ready  │                                        │
│  📄 report.pdf      │  [ Ask a question...        ] [Send]   │
└─────────────────────┴────────────────────────────────────────┘
```

---

## ✨ Features

- **Multi-Document Support** — Upload multiple PDF and TXT files at once
- **High-Speed Inference** — Groq API delivers near-instant responses
- **3 Model Choices** — Switch between Llama 3.1 8B, Mixtral 8x7B, and Gemma 2 9B on the fly
- **Zero-Dependency Vector Store** — Pure NumPy cosine similarity search (no ChromaDB, no protobuf issues)
- **Free-Tier Optimized** — Smart chunking and token-guard logic to stay within Groq's 6,000 TPM limit
- **Source Attribution** — Every answer shows which document it came from
- **Session Memory** — Chat history persists across the session
- **Secrets-Based Auth** — API key loaded securely from Streamlit secrets, no UI input needed

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Inference | Groq API |
| Embeddings | `BAAI/bge-small-en-v1.5` via FastEmbed |
| Vector Search | Pure NumPy (cosine similarity) |
| PDF Parsing | PyPDF |
| Hosting | Streamlit Cloud |

---

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/docchat-rag.git
cd docchat-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `fastembed` downloads the `BAAI/bge-small-en-v1.5` model (~130 MB) on first run. It is cached automatically after that.

### 4. Add your Groq API key

Create a file at `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Cloud

1. Push your code to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Select your repo, branch, and set `app.py` as the main file
4. Go to **Settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

5. Click **Deploy** — that's it!

---

## 📁 Project Structure

```
docchat-rag/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
└── .streamlit/
    └── secrets.toml     # API keys (local only, never commit this!)
```

---

## 📦 Dependencies

```txt
streamlit>=1.35.0
groq>=0.9.0
fastembed>=0.3.1
pypdf>=4.2.0
numpy>=1.24.0
```

No `chromadb`, no `protobuf`, no version conflict headaches.

---

## 🛡️ Rate Limit Protection (Groq Free Tier)

Groq's free tier allows **6,000 tokens per minute (TPM)**. DocChat handles this automatically:

| Protection | Detail |
|---|---|
| Chunk size | 400 words per chunk with 60-word overlap |
| Retrieved chunks | Max 5 chunks per query |
| Token guard | Trims oldest chat turns if prompt exceeds 5,500 tokens |
| Max answer length | 512 tokens per response |

---

## 🔒 API Key Security

The app reads your Groq API key **only from secrets** — never from user input:

```python
# Priority 1: Streamlit Cloud secrets (dashboard)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Priority 2: Local environment variable fallback
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
```

Never commit your `.streamlit/secrets.toml` to Git. Add it to `.gitignore`:

```
.streamlit/secrets.toml
```

---

## 🤖 Available Models

| Model | Groq ID | Best For |
|---|---|---|
| ⚡ Llama 3.1 8B | `llama-3.1-8b-instant` | Speed, free tier |
| 🔥 Mixtral 8x7B | `mixtral-8x7b-32768` | Balance of speed & quality |
| 💎 Gemma 2 9B | `gemma2-9b-it` | Quality answers |

---

## ⚠️ Limitations

- **In-memory only** — Vector store is rebuilt each session. Re-upload docs after refresh.
- **Text only** — Images inside PDFs are not extracted (coming soon).
- **Session-scoped** — Chat history clears on page refresh.
- **Free tier** — 6k TPM limit on Groq. Upgrade to paid for higher throughput.

---

## 🗺️ Roadmap

- [ ] Image support (OCR + Vision LLM)
- [ ] Persistent vector store across sessions
- [ ] Download chat history
- [ ] Support for DOCX and CSV files
- [ ] Multi-user support

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  Built with ⚡ by <strong>DocChat</strong> · Powered by <a href="https://groq.com">Groq</a> · Deployed on <a href="https://streamlit.io">Streamlit</a>
</div>
