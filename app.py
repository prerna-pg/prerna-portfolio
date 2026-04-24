"""
Prerna's Portfolio RAG Backend
Lightweight keyword-based retrieval + Gemini for generation
No heavy ML models — fits comfortably in Render free tier (512MB)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()  # loads .env file locally; on Render uses environment variables

app = Flask(__name__)
CORS(app)

# ── Config ──────────────────────────────────────────────────────────────────

KNOWLEDGE_DIR  = Path("knowledge")
CHUNK_SIZE     = 400
TOP_K          = 3

# ── Load knowledge base ──────────────────────────────────────────────────────
chunk_data = []

def load_knowledge_base():
    global chunk_data
    chunks = []
    for md_file in KNOWLEDGE_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk, current_words = [], 0
        for para in paragraphs:
            words = len(para.split())
            if current_words + words > CHUNK_SIZE and current_chunk:
                chunks.append({"source": md_file.stem, "text": "\n\n".join(current_chunk)})
                current_chunk, current_words = [], 0
            current_chunk.append(para)
            current_words += words
        if current_chunk:
            chunks.append({"source": md_file.stem, "text": "\n\n".join(current_chunk)})
    chunk_data = chunks
    print(f"Loaded {len(chunk_data)} knowledge chunks.")

# ── Keyword retrieval ────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","i","you","he","she",
    "it","we","they","what","who","how","and","or","but","in","on","at",
    "to","for","of","with","about","as","by","from","that","this","my","me"
}

def tokenize(text):
    return [w for w in re.findall(r'\b[a-z]+\b', text.lower()) if w not in STOPWORDS]

def retrieve(query):
    query_tokens = Counter(tokenize(query))
    if not query_tokens:
        return chunk_data[:TOP_K]
    scored = []
    for chunk in chunk_data:
        chunk_tokens = Counter(tokenize(chunk["text"]))
        score = sum(chunk_tokens.get(t, 0) * w for t, w in query_tokens.items())
        if any(w in chunk["source"] for w in query_tokens):
            score *= 1.5
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:TOP_K]]

# ── Gemini call ──────────────────────────────────────────────────────────────
def ask_gemini(query, context_chunks, api_key):
    import urllib.request, urllib.error

    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    system_prompt = """You are Prerna's portfolio assistant. Speak warmly in first person on behalf of Prerna Gupta — an AI Product Manager at American Express specializing in conversational AI, trust & reliability, and AI safety.

Answer using ONLY the context provided. Be specific with real projects and numbers. If something is not in the context, say: "I'm not trained on that yet, but I've noted your question!"

Keep answers conversational and human — 2-4 sentences unless more detail is needed."""

    payload = json.dumps({
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"}]}]
    }).encode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        print(f"Gemini API error: {e.code} — {e.read().decode()}")
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
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set."}), 500
    chunks   = retrieve(query)
    response = ask_gemini(query, chunks, api_key)
    return jsonify({"response": response})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "chunks_loaded": len(chunk_data)})

# ── Start ─────────────────────────────────────────────────────────────────────
load_knowledge_base()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Portfolio running at http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)