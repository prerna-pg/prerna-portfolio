# Prerna Gupta — AI Portfolio

A personal portfolio site with a RAG-powered chat assistant built on your own knowledge base.
**Stack:** HTML/CSS/JS frontend · Python (Flask) backend · Sentence Transformers (local embeddings) · Gemini Flash API

---

## Project Structure

```
prerna-portfolio/
├── index.html              ← The website (deploy this to GitHub Pages)
├── app.py                  ← Python RAG backend
├── requirements.txt        ← Python dependencies
├── knowledge/
│   ├── about.md            ← Your story and background
│   ├── projects.md         ← Detailed project writeups
│   ├── skills_philosophy.md← Skills, tools, and product philosophy
│   └── philosophy.md       ← Personal values and goals
└── README.md
```

---

## Weekend Setup Guide

### Step 1 — Get your Gemini API key (free)

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google → Create API key
3. Copy it somewhere safe

Gemini 1.5 Flash free tier: **1,500 requests/day** — more than enough.

---

### Step 2 — Set up Python locally

```bash
# Clone or navigate to this folder
cd prerna-portfolio

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# or: venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY="your_key_here"   # Mac/Linux
# or: set GEMINI_API_KEY=your_key_here  # Windows
```

---

### Step 3 — Run locally

```bash
python app.py
```

Open http://localhost:5000 — the site loads, the chat widget connects to your backend.

Test it by asking: *"What did Prerna build at AmEx?"*

---

### Step 4 — Update your knowledge base

Edit the `.md` files in `/knowledge/` to add more detail, correct numbers, or update projects.
Restart `app.py` after editing — embeddings reload on startup.

---

### Step 5 — Deploy the frontend to GitHub Pages

1. Create a new GitHub repo: `prerna-portfolio`
2. Push only the frontend files:
   ```
   index.html
   knowledge/   (optional — for reference)
   README.md
   ```
3. Go to repo → **Settings → Pages → Source: main branch / root**
4. Your site will be live at: `https://yourusername.github.io/prerna-portfolio`

> Note: GitHub Pages hosts static files only. The chat widget needs the Python backend running separately.

---

### Step 6 — Deploy the backend (free options)

#### Option A: Render.com (easiest, recommended)
1. Push the full project to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python app.py`
6. Add environment variable: `GEMINI_API_KEY = your_key`
7. Deploy → you get a URL like `https://prerna-portfolio.onrender.com`

#### Option B: Railway.app
1. Go to https://railway.app → New Project → Deploy from GitHub
2. Add `GEMINI_API_KEY` in Variables tab
3. Deploy → Railway gives you a public URL

After deploying the backend, update this line in `index.html`:
```javascript
// Find this line near the bottom of the <script> tag:
: '/chat'; // update to your deployed backend URL

// Change to:
: 'https://your-render-url.onrender.com/chat';
```

---

## Updating Your Knowledge Base

To add a new project (e.g., UPI Transaction Parser when you build it):

1. Open `knowledge/projects.md`
2. Add a new section:
   ```markdown
   ## Project 7: UPI Transaction Parser
   **Status:** Building · Personal project
   ### What I'm Building
   ...
   ```
3. Restart the Python server — it re-embeds automatically

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Chat says "can't reach backend" | Make sure `python app.py` is running |
| Gemini returns an error | Check your API key is set correctly |
| Embeddings slow on first load | Normal — sentence-transformers downloads model once |
| Chat gives wrong answers | Improve the `.md` files with more specific details |

---

## Next Projects to Add

- [ ] UPI Transaction Parser
- [ ] Budgeting App
- [ ] Fitness Tracker

Each project gets a card on the homepage and a section in `knowledge/projects.md`.

---

*Built by Prerna Gupta · gprernapg@gmail.com*
