"""
Prerna's Portfolio RAG Backend
Uses sentence-transformers for local embeddings + Gemini Flash for generation
Run: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import numpy as np
from pathlib import Path

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ── Config ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
KNOWLEDGE_DIR  = Path("knowledge")
CHUNK_SIZE     = 400   # words per chunk
TOP_K          = 3     # chunks to retrieve

# ── Load knowledge base ──────────────────────────────────────────────────────
def load_knowledge_base():
    chunks = []
    for md_file in KNOWLEDGE_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        # Split into paragraphs, then group into chunks
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk, current_words = [], 0
        for para in paragraphs:
            words = len(para.split())
            if current_words + words > CHUNK_SIZE and current_chunk:
                chunks.append({
                    "source": md_file.stem,
                    "text": "\n\n".join(current_chunk)
                })
                current_chunk, current_words = [], 0
            current_chunk.append(para)
            current_words += words
        if current_chunk:
            chunks.append({
                "source": md_file.stem,
                "text": "\n\n".join(current_chunk)
            })
    return chunks

# ── Embedding & retrieval ────────────────────────────────────────────────────
embedder      = None
chunk_data    = []
chunk_vectors = None

def init_embeddings():
    global embedder, chunk_data, chunk_vectors
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        embedder   = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_data = load_knowledge_base()
        texts      = [c["text"] for c in chunk_data]
        chunk_vectors = embedder.encode(texts, show_progress_bar=False)
        print(f"Loaded {len(chunk_data)} knowledge chunks.")
    except ImportError:
        print("sentence-transformers not installed. Using keyword fallback.")

def retrieve(query: str) -> list[dict]:
    """Return top-k relevant chunks for the query."""
    if embedder is None or chunk_vectors is None:
        return keyword_fallback(query)
    q_vec = embedder.encode([query])
    # Cosine similarity
    norms   = np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q_vec)
    norms   = np.where(norms == 0, 1e-9, norms)
    scores  = (chunk_vectors @ q_vec.T).flatten() / norms
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [chunk_data[i] for i in top_idx]

def keyword_fallback(query: str) -> list[dict]:
    """Simple keyword search when embeddings unavailable."""
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunk_data:
        chunk_words = set(chunk["text"].lower().split())
        score = len(query_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:TOP_K]]

# ── Gemini call ──────────────────────────────────────────────────────────────
def ask_gemini(query: str, context_chunks: list[dict]) -> str:
    import urllib.request
    import urllib.error

    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)

    system_prompt = """You are Prerna's portfolio assistant. You speak warmly, in first person on behalf of Prerna Gupta — an AI Product Manager at American Express who specializes in conversational AI, trust & reliability, and AI safety.

Answer questions about Prerna's experience, projects, skills, and philosophy using ONLY the context provided. Be specific and reference real projects and numbers when available. If something isn't in the context, say so honestly rather than guessing.

Keep answers conversational, confident, and human — like Prerna herself is explaining her work to a curious hiring manager or fellow PM. 2-4 sentences is usually the right length unless the question needs detail."""

    payload = json.dumps({
        "contents": [{
            "parts": [{
                "text": f"{system_prompt}\n\nContext about Prerna:\n{context}\n\nQuestion: {query}"
            }]
        }]
    }).encode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"Gemini API error: {e.code} — {error_body}")
        return "Sorry, I couldn't reach the AI model right now. Please try again in a moment."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Something went wrong. Please try again."

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data  = request.get_json()
    query = (data or {}).get("message", "").strip()
    if not query:
        return jsonify({"error": "No message provided"}), 400
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set. Add it to your environment."}), 500

    chunks   = retrieve(query)
    response = ask_gemini(query, chunks)
    return jsonify({"response": response})

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "chunks_loaded": len(chunk_data),
        "embedder": "sentence-transformers" if embedder else "keyword-fallback"
    })

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_embeddings()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Portfolio running at http://localhost:{port}\n")
    app.run(debug=True, port=port)