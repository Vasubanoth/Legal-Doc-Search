import streamlit as st
import io
import os
from pypdf import PdfReader
from pipeline import build_vector_store, RAGPipeline

st.set_page_config(page_title="Legal RAG", layout="wide")

# Load API key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

st.title("📄 Legal Document Assistant")
st.caption("Ask questions with exact citations")

# Upload
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if "store" not in st.session_state:
    st.session_state.store = None

if st.button("Process Documents"):
    docs = []

    for file in uploaded_files:
        reader = PdfReader(io.BytesIO(file.read()))
        for i, page in enumerate(reader.pages):
            docs.append({
                "name": file.name,
                "text": page.extract_text() or "",
                "page": i + 1
            })

    st.session_state.store = build_vector_store(docs)
    st.success("Documents processed!")

# Query
query = st.text_input("Ask your question")

if st.button("Get Answer"):
    if not st.session_state.store:
        st.warning("Upload and process documents first")
    else:
        pipeline = RAGPipeline(st.session_state.store, GROQ_API_KEY)
        result = pipeline.query(query)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for s in result["sources"]:
            st.markdown(
                f"**{s['document']} (Page {s['page']})**\n\n{s['chunk']}..."
            )

        st.metric("Confidence", result["confidence"])
