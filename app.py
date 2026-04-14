import streamlit as st
import os
import time
import re
from pathlib import Path

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="DocChat RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (heavy libs) ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ── Constants ────────────────────────────────────────────────────────────────
MODELS = {
    "⚡ Llama 3.1 8B (fastest)": "llama-3.1-8b-instant",
    "🔥 Mixtral 8x7B (balanced)": "mixtral-8x7b-32768",
    "💎 Gemma 2 9B (quality)":    "gemma2-9b-it",
}

CHUNK_SIZE   = 400   # tokens ≈ words * 0.75
CHUNK_OVERLAP = 60
MAX_CHUNKS   = 5     # retrieved per query
TPM_GUARD    = 5_500 # stay under 6000 TPM free tier


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.33)


def build_vector_store(docs: list[dict]):
    """
    Pure-Python vector store using numpy cosine similarity.
    No ChromaDB, no protobuf, works on any Python version.
    Returns a dict with embeddings, chunks, and metadata.
    """
    import numpy as np
    embed_model = get_embedding_model()

    all_chunks, all_metas = [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metas.append({"source": doc["name"], "chunk_idx": idx})

    if not all_chunks:
        return {"embeddings": None, "chunks": [], "metas": []}

    embeddings = list(embed_model.embed(all_chunks))
    emb_matrix = np.array(embeddings, dtype=np.float64)
    # L2-normalise rows for cosine similarity via dot product
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_matrix = emb_matrix / norms

    return {"embeddings": emb_matrix, "chunks": all_chunks, "metas": all_metas}


def retrieve_chunks(store: dict, query: str, n: int = MAX_CHUNKS):
    import numpy as np
    embed_model = get_embedding_model()

    if store["embeddings"] is None:
        return [], []

    q_emb = np.array(list(embed_model.embed([query]))[0], dtype=np.float64)
    q_emb = q_emb / (np.linalg.norm(q_emb) or 1)

    scores = store["embeddings"] @ q_emb          # cosine similarity
    top_n  = int(min(n, len(store["chunks"])))
    top_idx = scores.argsort()[::-1][:top_n]

    chunks = [store["chunks"][i] for i in top_idx]
    metas  = [store["metas"][i]  for i in top_idx]
    return chunks, metas


def build_prompt(context_chunks: list[str], metas: list[dict], query: str, history: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[Source: {m['source']}, chunk {m['chunk_idx']}]\n{c}"
        for c, m in zip(context_chunks, metas)
    )
    system = (
        "You are DocChat, an expert document assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer isn't in the context, say so honestly. "
        "Be concise, accurate, and cite the source filename when helpful.\n\n"
        f"CONTEXT:\n{context}"
    )
    messages = [{"role": "system", "content": system}]
    # Include last 6 turns for memory
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": query})
    return messages


def stream_answer(client, model_id: str, messages: list[dict]):
    total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    if total_tokens > TPM_GUARD:
        # Trim oldest non-system messages
        while total_tokens > TPM_GUARD and len(messages) > 2:
            messages.pop(1)
            total_tokens = sum(estimate_tokens(m["content"]) for m in messages)

    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=512,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        yield delta


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0e1a; }
section[data-testid="stSidebar"] {
    background: #0d1221;
    border-right: 1px solid #1e2d4a;
}

/* ── Header ── */
.docchat-header {
    display: flex; align-items: center; gap: 14px;
    padding: 0 0 24px 0; border-bottom: 1px solid #1e2d4a; margin-bottom: 24px;
}
.docchat-logo {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.docchat-title { font-family: 'Space Mono', monospace; font-size: 1.6rem;
    font-weight: 700; color: #e8f0fe; letter-spacing: -0.5px; line-height: 1; }
.docchat-sub { font-size: 0.75rem; color: #5a7ab0; margin-top: 3px; letter-spacing: 0.5px; text-transform: uppercase; }

/* ── Chat messages ── */
.chat-wrap { display: flex; flex-direction: column; gap: 16px; margin-bottom: 16px; }

.msg-user {
    align-self: flex-end; max-width: 72%;
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2444 100%);
    border: 1px solid #2a4a7a; border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; color: #cfe3ff; font-size: 0.93rem; line-height: 1.55;
}
.msg-ai {
    align-self: flex-start; max-width: 80%;
    background: #111827; border: 1px solid #1e2d4a;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px; color: #d1dff5; font-size: 0.93rem; line-height: 1.6;
}
.msg-label {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;
}
.msg-user .msg-label { color: #4a8fd4; }
.msg-ai   .msg-label { color: #7c3aed; }

/* ── Input area ── */
.stTextInput > div > div > input {
    background: #111827 !important; border: 1px solid #1e2d4a !important;
    color: #d1dff5 !important; border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.93rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00d4ff !important; box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #3a5070 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 9px 20px !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover { opacity: .9 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Sidebar widgets ── */
.stSelectbox > div, .stFileUploader > div {
    background: #111827 !important; border-radius: 10px !important;
}
label, .stSelectbox label, .stFileUploader label {
    color: #7a9cc8 !important; font-size: 0.8rem !important;
    text-transform: uppercase !important; letter-spacing: 0.6px !important;
}

/* ── Status / metrics ── */
.stat-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #111827; border: 1px solid #1e2d4a; border-radius: 20px;
    padding: 4px 12px; font-size: 0.75rem; color: #5a7ab0;
    font-family: 'Space Mono', monospace;
}
.stat-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; }
.stat-pill .dot.off { background: #ef4444; }

/* ── Scrollable chat area ── */
.chat-container {
    max-height: 60vh; overflow-y: auto; padding-right: 6px;
    scrollbar-width: thin; scrollbar-color: #1e2d4a transparent;
}
.chat-container::-webkit-scrollbar { width: 4px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 4px; }

/* ── Source badge ── */
.src-badge {
    display: inline-block; background: #0d1221; border: 1px solid #1e3a5f;
    border-radius: 6px; padding: 2px 8px; font-size: 0.7rem;
    color: #4a8fd4; font-family: 'Space Mono', monospace; margin: 4px 3px 0 0;
}

/* ── Divider ── */
hr { border-color: #1e2d4a !important; }

/* ── Alerts / info ── */
.stAlert { border-radius: 10px !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────
for key, val in {
    "messages": [],
    "collection": None,
    "doc_names": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load API key from Streamlit secrets (set in Streamlit Cloud dashboard) ───
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="docchat-header">
        <div class="docchat-logo">⚡</div>
        <div>
            <div class="docchat-title">DocChat</div>
            <div class="docchat-sub">RAG · Groq · FastEmbed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    model_label = st.selectbox("Model", list(MODELS.keys()), index=0)
    model_id = MODELS[model_label]

    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="PDF or TXT files — multiple allowed",
    )

    process_btn = st.button("⚡ Process Documents", use_container_width=True, disabled=not uploaded_files)

    if process_btn and uploaded_files:
        with st.spinner("Extracting & indexing…"):
            docs = []
            for f in uploaded_files:
                raw = f.read()
                if f.name.endswith(".pdf"):
                    text = extract_text_from_pdf(raw)
                else:
                    text = raw.decode("utf-8", errors="replace")
                docs.append({"name": f.name, "text": text})

            st.session_state.collection = build_vector_store(docs)
            st.session_state.doc_names  = [d["name"] for d in docs]
            st.session_state.messages   = []  # fresh chat on new docs
        st.success(f"✅ Indexed {len(docs)} document(s)")

    # Status
    st.markdown("---")
    if st.session_state.doc_names:
        st.markdown(f'<span class="stat-pill"><span class="dot"></span>Vector DB ready</span>', unsafe_allow_html=True)
        for n in st.session_state.doc_names:
            st.markdown(f'<span class="src-badge">📄 {n}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="stat-pill"><span class="dot off"></span>No docs loaded</span>', unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown("---")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Main Chat Area ───────────────────────────────────────────────────────────
col_main, col_pad = st.columns([3, 1])

with col_main:
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h2 style="font-family:'Space Mono',monospace; color:#e8f0fe; font-size:1.3rem; margin:0;">
            Chat with your Documents
        </h2>
        <p style="color:#3a5070; font-size:0.82rem; margin:4px 0 0 0;">
            Upload PDFs or TXT files → Process → Ask anything
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Render history ─────────────────────────────────────────────────────
    if st.session_state.messages:
        chat_html = '<div class="chat-container"><div class="chat-wrap">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f'<div class="msg-user"><div class="msg-label">You</div>{msg["content"]}</div>'
            else:
                sources = msg.get("sources", [])
                src_html = ""
                if sources:
                    unique_src = list(dict.fromkeys(s["source"] for s in sources))
                    src_html = "<div style='margin-top:8px;'>" + "".join(
                        f'<span class="src-badge">📄 {s}</span>' for s in unique_src
                    ) + "</div>"
                chat_html += f'<div class="msg-ai"><div class="msg-label">DocChat</div>{msg["content"]}{src_html}</div>'
        chat_html += "</div></div>"
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:48px 24px; color:#2a3d5a;">
            <div style="font-size:3rem; margin-bottom:16px;">📚</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.9rem; color:#3a5570;">
                Upload documents and start chatting
            </div>
            <div style="font-size:0.78rem; color:#2a3d5a; margin-top:8px;">
                PDF • TXT • Multi-file supported
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Input ──────────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input(
                "Ask a question",
                placeholder="What does the document say about…",
                label_visibility="collapsed",
            )
        with c2:
            submit = st.form_submit_button("Send", use_container_width=True)

    # ── Handle submit ──────────────────────────────────────────────────────
    if submit and user_input.strip():
        if not GROQ_API_KEY:
            st.error("🔑 GROQ_API_KEY not found. Add it to Streamlit Cloud secrets.")
        elif st.session_state.collection is None:
            st.error("📂 Please upload and process at least one document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})

            with st.spinner("Thinking…"):
                try:
                    client = get_groq_client(GROQ_API_KEY)
                    chunks, metas = retrieve_chunks(st.session_state.collection, user_input.strip())
                    messages = build_prompt(chunks, metas, user_input.strip(), st.session_state.messages[:-1])

                    answer = ""
                    for token in stream_answer(client, model_id, messages):
                        answer += token

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": metas,
                    })

                except Exception as e:
                    err = str(e)
                    if "rate_limit" in err.lower():
                        msg = "⏳ Rate limit hit. Wait ~60 s and try again (Groq free tier: 6k TPM)."
                    elif "authentication" in err.lower() or "api_key" in err.lower():
                        msg = "🔑 Invalid API key. Check your Groq key."
                    else:
                        msg = f"❌ Error: {err}"
                    st.session_state.messages.append({"role": "assistant", "content": msg})

            st.rerun()
