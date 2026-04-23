# 📄 Legal RAG Assistant (Streamlit + Groq)

A minimal **Retrieval-Augmented Generation (RAG)** system for querying legal documents with **exact source citations (document + page)**.

This project allows users to upload PDF documents, ask questions, and receive answers grounded strictly in the source content.

---

## 🚀 Features

- 📂 Upload multiple PDF documents
- 🔍 Semantic search using **FastEmbed (BGE model)**
- 🤖 Answer generation using **Groq (Llama 3.1)**
- 📄 **Page-level citations** for every answer
- 📊 Confidence score based on similarity
- ⚡ Lightweight, fast, and fully local vector store (NumPy)
- 🖥️ Simple Streamlit UI

---

## 🧠 How It Works

1. PDFs are split into page-level text
2. Text is chunked with overlap
3. Chunks are converted into embeddings
4. Stored in a vector space (NumPy)
5. Query is embedded and matched via cosine similarity
6. Top chunks are passed to Groq LLM
7. Response is generated strictly from retrieved context

---

## 📁 Project Structure

---
* the web Application is live at : "https://8na5qx6dgi2bxi4tj4njzc.streamlit.app/"
