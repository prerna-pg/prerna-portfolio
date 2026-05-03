"""
Prerna's Portfolio RAG Backend
Lightweight keyword-based retrieval + Gemini for generation
No heavy ML models — fits comfortably in Render free tier (512MB)

Changes from v1:
  - Cache layer for 6 common questions (no Gemini call needed)
  - SQLite question logging with admin review endpoint
  - CORS restricted to production + localhost
  - In-memory rate limiting: 20 req / IP / hour
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import re
import sqlite3                          # stdlib — no pip install needed
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()  # loads .env locally; on Render uses environment variables

app = Flask(__name__)

# ── CORS — restricted to prod + localhost ────────────────────────────────────
CORS(app, origins=[
    "https://prerna-portfolio.onrender.com",
    "http://localhost:5000",
])

# ── Config ───────────────────────────────────────────────────────────────────
KNOWLEDGE_DIR = Path("knowledge")
CHUNK_SIZE    = 400
TOP_K         = 3
DB_PATH       = "questions.db"
ADMIN_TOKEN   = os.environ.get("ADMIN_TOKEN", "change-me-in-prod")

# ── Cache layer ───────────────────────────────────────────────────────────────
# Warm, first-person answers for high-frequency questions.
# Checked before retrieve() / ask_gemini() — zero latency, zero API cost.
CACHE = {
    "what did you build at amex": (
        "At AmEx I've shipped across both analytics and AI product. "
        "Most recently I led Trust & Reliability for our first agentic AI voicebot — "
        "targeting ~1 million annual calls — owning four workstreams: Observability, "
        "Data Ingestion, Guardrails, and Red Teaming. "
        "Before that I ran end-to-end delivery of an IVR LLM pilot across 80K annual calls "
        "with 230 intent classes, and launched SpendSmart, a budgeting tool for our "
        "@Work corporate card platform used by 1.5M clients across 120+ countries."
    ),
    "how do you approach ai safety": (
        "I treat AI safety as a first-class product problem, not an engineering checkbox. "
        "For our voicebot I partnered with enterprise Infosec and third-party red-team vendors "
        "to define the full guardrails strategy — covering adversarial prompt injection, "
        "hallucination containment, and what 'safe to launch' actually means in a regulated "
        "banking environment. "
        "I believe you can't outsource safety thinking to engineers; a PM has to own the "
        "threat model and the acceptable-risk tradeoffs."
    ),
    "what's your background in data": (
        "I started in analytics before product, which shapes how I think about everything I build. "
        "I defined 20+ KPI frameworks across six @Work user journeys, owned global measurement "
        "for CSAT, Digital Self-Serve Rate, and Revenue across five cross-functional teams, "
        "and built Inferno — an NLP model that turned unstructured customer-care text into "
        "strategic roadmap signal across 5+ product journeys. "
        "My rule: design the metric before you design the feature."
    ),
    "tell me about red teaming": (
        "Red teaming for our agentic voicebot was a multi-round process I coordinated end-to-end. "
        "I brought in a third-party red-team vendor, ran experiential adversarial sessions internally, "
        "and worked with enterprise Infosec to translate findings into concrete prompt-based guardrails. "
        "The goal wasn't just to find failure modes — it was to define the criteria we'd accept "
        "before putting the system in front of real customers on real calls."
    ),
    "what are you building next": (
        "A few things are in the pipeline! "
        "I'm working on a Transaction Parser that turns raw Indian UPI and bank-statement data "
        "into structured spending insights — because personal finance tools in India are still "
        "built around Western spending patterns. "
        "I'm also planning a lightweight Budgeting App and a Fitness Tracker with AI-assisted "
        "planning. Each one will be documented here as it ships — the point is learning in public, "
        "not perfection."
    ),
    "what makes you different as a pm": (
        "The data background is the honest answer. "
        "Most PMs wait for engineers to flag when an AI system is broken; I can read the "
        "signals myself — containment rates, escalation patterns, intent drift, pipeline failures. "
        "In agentic systems where decisions happen autonomously at scale, that early-warning "
        "instinct matters a lot. "
        "I also came up through zero-to-one delivery — not just roadmap slides — so I'm "
        "comfortable getting hands-on with prototypes, PRDs, red-team sessions, and "
        "observability dashboards in the same week."
    ),
}

# ── Rate limiting — 20 requests / IP / hour ──────────────────────────────────
# { ip: [datetime, datetime, ...] }  — timestamps of recent requests
_rate_limit_store: dict[str, list] = defaultdict(list)
RATE_LIMIT      = 20
RATE_WINDOW     = timedelta(hours=1)


def is_rate_limited(ip: str) -> bool:
    now   = datetime.utcnow()
    cutoff = now - RATE_WINDOW
    # prune old timestamps
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > cutoff]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT:
        return True
    _rate_limit_store[ip].append(now)
    return False


# ── SQLite — question logging ─────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            question         TEXT    NOT NULL,
            asked_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            gemini_response  TEXT,
            reviewed         INTEGER DEFAULT 0
        )
    """)
    con.commit()
    con.close()
    print(f"SQLite DB ready at {DB_PATH}")


def log_question(question: str, response: str):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO questions (question, gemini_response) VALUES (?, ?)",
            (question, response),
        )
        con.commit()
        con.close()
    except Exception as exc:
        print(f"DB log error: {exc}")


# ── Load knowledge base ───────────────────────────────────────────────────────
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


# ── Keyword retrieval ─────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","i","you","he","she",
    "it","we","they","what","who","how","and","or","but","in","on","at",
    "to","for","of","with","about","as","by","from","that","this","my","me",
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


# ── Gemini call ───────────────────────────────────────────────────────────────
def ask_gemini(query, context_chunks, api_key):
    import urllib.request, urllib.error

    context = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    system_prompt = (
        "You are Prerna's portfolio assistant. Speak warmly in first person on behalf of "
        "Prerna Gupta — an AI Product Manager at American Express specializing in "
        "conversational AI, trust & reliability, and AI safety.\n\n"
        "Answer using ONLY the context provided. Be specific with real projects and numbers. "
        "If something is not in the context, say: "
        "\"I'm not trained on that yet, but I've noted your question!\"\n\n"
        "Keep answers conversational and human — 2-4 sentences unless more detail is needed."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"}]}]
    }).encode("utf-8")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    # ── Rate limit check ──────────────────────────────────────────────────────
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    if is_rate_limited(ip):
        return jsonify({"error": "Rate limit exceeded. Please wait before sending more messages."}), 429

    data  = request.get_json()
    query = (data or {}).get("message", "").strip()
    if not query:
        return jsonify({"error": "No message provided"}), 400

    # ── Cache check (case-insensitive) ────────────────────────────────────────
    cache_key = query.lower().rstrip("?").strip()
    if cache_key in CACHE:
        return jsonify({"response": CACHE[cache_key], "cached": True})

    # ── Gemini path ───────────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set."}), 500

    chunks   = retrieve(query)
    response = ask_gemini(query, chunks, api_key)

    # ── Log every non-cached question + response ──────────────────────────────
    log_question(query, response)

    return jsonify({"response": response})


@app.route("/questions", methods=["GET"])
def get_questions():
    """Admin endpoint — returns all logged questions as JSON."""
    token = request.args.get("token", "")
    if token != ADMIN_TOKEN:
        return jsonify({"error": "Forbidden"}), 403

    con  = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, question, asked_at, gemini_response, reviewed FROM questions ORDER BY asked_at DESC"
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@app.route("/questions/<int:question_id>/reviewed", methods=["POST"])
def mark_reviewed(question_id):
    """Admin endpoint — marks a question as reviewed."""
    token = request.args.get("token", "")
    if token != ADMIN_TOKEN:
        return jsonify({"error": "Forbidden"}), 403

    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE questions SET reviewed = 1 WHERE id = ?", (question_id,))
    con.commit()
    con.close()
    return jsonify({"ok": True, "id": question_id})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "chunks_loaded": len(chunk_data)})


# ── Startup ───────────────────────────────────────────────────────────────────
init_db()
load_knowledge_base()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Portfolio running at http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
