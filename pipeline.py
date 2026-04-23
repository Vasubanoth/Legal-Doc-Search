import numpy as np
from fastembed import TextEmbedding
from groq import Groq

# Init embedding model
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def chunk_text(text, chunk_size=400, overlap=60):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_vector_store(docs):
    all_chunks, all_metas = [], []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metas.append({
                "document": doc["name"],
                "page": doc["page"],
                "chunk_id": idx
            })

    if not all_chunks:
        return None

    embeddings = list(embed_model.embed(all_chunks))
    emb_matrix = np.array(embeddings)

    # normalize for cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / (norms + 1e-10)

    return {
        "embeddings": emb_matrix,
        "chunks": all_chunks,
        "metas": all_metas
    }


def retrieve(query, store, top_k=3):
    if store is None:
        return [], [], []

    q_emb = np.array(list(embed_model.embed([query]))[0])
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

    scores = store["embeddings"] @ q_emb
    idxs = scores.argsort()[::-1][:top_k]

    chunks = [store["chunks"][i] for i in idxs]
    metas = [store["metas"][i] for i in idxs]
    sims = [float(scores[i]) for i in idxs]

    return chunks, metas, sims


class RAGPipeline:
    def __init__(self, store, groq_api_key):
        self.store = store
        self.client = Groq(api_key=groq_api_key)

    def query(self, question: str):
        chunks, metas, sims = retrieve(question, self.store)

        if not chunks:
            return {
                "answer": "Insufficient information",
                "sources": [],
                "confidence": 0.0
            }

        context = "\n\n".join([
            f"[{m['document']} | Page {m['page']}]\n{c}"
            for c, m in zip(chunks, metas)
        ])

        prompt = f"""
Answer ONLY from the given context.
If not found, say "Insufficient information".

Context:
{context}

Question:
{question}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content

        sources = [
            {
                "document": m["document"],
                "page": m["page"],
                "chunk": c[:200]
            }
            for c, m in zip(chunks, metas)
        ]

        confidence = float(np.mean(sims)) if sims else 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 3)
        }
