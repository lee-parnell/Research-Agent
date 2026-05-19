# 🔬 Autonomous Research Agent API

> **Submit a topic. Get back a fully sourced, hallucination-scored research report — in seconds.**

A production-ready AI research pipeline built with FastAPI, LangChain, Groq, and FAISS. Give it any topic and it autonomously searches the web, builds a local vector store, retrieves the most relevant context, synthesizes a grounded summary using a state-of-the-art LLM, and scores the output for hallucination risk — all in one API call.

---

## ✨ What It Does

Most AI tools just answer questions. This one **researches** them.

Every response is:
- **Grounded** — built from real, live web sources via Tavily search
- **Ranked** — sources scored by relevance using FAISS vector similarity
- **Synthesized** — summarized by Groq's LLaMA 3.3 70B with strict source-only rules
- **Verified** — assigned a hallucination risk score so you know how much to trust it

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   Tavily Search  │  ← Live web search (5–10 sources)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Indexing  │  ← Embed sources with sentence-transformers (local, free)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Retrieval   │  ← Top-k most relevant chunks retrieved
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Groq LLaMA 3   │  ← Structured JSON synthesis (source-grounded only)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Hallucination Score  │  ← Cosine similarity between summary & sources
└──────────────────────┘
         │
         ▼
    JSON Response
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **API** | FastAPI | Fast, async, auto-docs |
| **LLM** | Groq + LLaMA 3.3 70B | Free tier, blazing fast inference |
| **Web Search** | Tavily | Purpose-built for AI agents |
| **Embeddings** | sentence-transformers (local) | Free, no API calls |
| **Vector Store** | FAISS (in-memory) | Fast similarity search |
| **Tracing** | LangSmith | Full pipeline observability |
| **Container** | Docker | One-command deployment |

---

## 🚀 Getting Started

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Free API keys (takes 2 minutes):
  - [Groq](https://console.groq.com) — LLM inference
  - [Tavily](https://app.tavily.com) — Web search
  - [LangSmith](https://smith.langchain.com) — Tracing (optional)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/autonomous-research-agent.git
cd autonomous-research-agent
```

### 2. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and fill in your API keys:
```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here   # optional
```

### 3. Build and run
```bash
docker-compose up --build
```

### 4. Open the interactive docs
```
http://localhost:8000/docs
```

That's it. The API is live.

---

## 📡 API Reference

### `POST /api/v1/research`

Run a full autonomous research pipeline on any topic.

**Request body:**
```json
{
  "topic": "Latest breakthroughs in multimodal AI 2024",
  "depth": 3,
  "language": "en"
}
```

| Field | Type | Description |
|---|---|---|
| `topic` | string | What to research (3–500 chars) |
| `depth` | int (1–5) | 1 = quick scan, 5 = deep dive |
| `language` | string | Output language code |

**Response:**
```json
{
  "topic": "Latest breakthroughs in multimodal AI 2024",
  "summary": "In 2024, multimodal AI saw significant advances...",
  "key_findings": [
    "GPT-4o introduced real-time voice and vision capabilities [Source 1]",
    "Google DeepMind's Gemini 1.5 Pro extended context to 1M tokens [Source 2]",
    "..."
  ],
  "sources": [
    {
      "title": "The State of Multimodal AI in 2024",
      "url": "https://...",
      "snippet": "...",
      "relevance_score": 0.94
    }
  ],
  "hallucination_score": 0.08,
  "confidence": "high",
  "tokens_used": 1243,
  "duration_seconds": 8.4,
  "timestamp": "2024-11-15T10:30:00"
}
```

### `GET /health`

Check that the API and models are loaded correctly.

---

## 🎯 Example Queries

```bash
# Quick lookup
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "What is GPT-4?", "depth": 1}'

# Deep research
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "CRISPR gene editing latest medical applications 2024", "depth": 5}'
```

---

## 📊 Understanding the Response

**`hallucination_score`** — how grounded the summary is in the retrieved sources:
- `0.0 – 0.2` ✅ Fully grounded, high trust
- `0.2 – 0.45` ⚠️ Mostly grounded, review sources
- `0.45+` ❌ Low grounding, treat with caution

**`confidence`** — derived from the same grounding calculation:
- `high` — summary closely mirrors source content
- `medium` — some paraphrasing or inference
- `low` — significant deviation from sources

**`depth`** — controls the research effort:
- `1–2` Basic search, fast (~5s)
- `3–4` Advanced search, balanced (~15s)
- `5` Deep search, thorough (~30s+)

---

## 📁 Project Structure

```
autonomous-research-agent/
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── config.py             # Settings via pydantic-settings
│   ├── models/
│   │   └── schemas.py        # Request/response models
│   ├── routers/
│   │   └── research.py       # API route handler
│   ├── services/
│   │   ├── agent.py          # Main orchestration pipeline
│   │   ├── search.py         # Tavily web search
│   │   ├── embeddings.py     # FAISS vector store
│   │   └── llm.py            # Groq LLM synthesis
│   └── utils/
│       └── hallucination.py  # Grounding score calculator
├── tests/
│   └── test_research.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 🔒 Security Notes

- Never commit your `.env` file — it contains live API keys
- The `.gitignore` excludes `.env` by default
- Use `.env.example` as a safe template to share

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  Built with FastAPI · LangChain · Groq · FAISS · Tavily
</div>
