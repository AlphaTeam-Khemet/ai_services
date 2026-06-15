# 🏛️ KHEMET — Egyptian RAG Chatbot

> **RAG-powered AI guide for ancient Egyptian knowledge**  
> Built with Groq API · ChromaDB · FastAPI

---

## 📌 What is this service

**KHEMET** is a multilingual **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about ancient Egypt and generates monument descriptions. It is the AI backbone of the **Egyptian AI Guide** application.

**How it works:**

1. **70 Wikipedia articles** about ancient Egypt are stored as vector embeddings in **ChromaDB**
2. The system retrieves the most relevant text chunks using semantic search
3. **Groq API (llama-3.3-70b-versatile)** natively processes the prompt and generates a grounded, conversational answer in the user's original language
4. The response is returned via a **FastAPI** REST API

**This service connects with:**

- 🖼️ A **computer vision** module that identifies Egyptian monuments from images
- 🌐 A **frontend web application** that displays answers to users
- 📱 Any **mobile or desktop client** via the REST API

**Key technologies:**

| Component       | Technology                       |
|-----------------|----------------------------------|
| LLM             | Groq API — llama-3.3-70b-versatile  |
| Embeddings      | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector Database | ChromaDB (persistent)            |
| Backend Server  | FastAPI + Uvicorn                |

> [!IMPORTANT]
> **No GPU required.** The LLM runs on Groq's cloud infrastructure. You only need a free API key.

---

## 💻 System requirements

| Requirement      | Minimum               |
|------------------|-----------------------|
| Python           | 3.10+                 |
| RAM              | 4GB                   |
| Disk space       | 2GB free              |
| Operating system | Linux / macOS / Windows |
| Groq API key     | Free at [console.groq.com](https://console.groq.com/keys) |

---

## 🔑 Required environment variables

Before running, export the following keys (or add them to the root `.env` file):

```bash
# Required — Groq LLM API key (RAG answers + narration text generation)
export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Required for Voice Tour Guide audio (optional — service degrades gracefully without these)
export ELEVENLABS_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export ELEVENLABS_VOICE_EN=<english-voice-id>   # ElevenLabs voice ID for English TTS
export ELEVENLABS_VOICE_AR=<arabic-voice-id>    # ElevenLabs voice ID for Arabic TTS
```

Get a free Groq key at 👉 [console.groq.com/keys](https://console.groq.com/keys)  
Get a free ElevenLabs key at 👉 [elevenlabs.io](https://elevenlabs.io)

> [!NOTE]
> If `ELEVENLABS_API_KEY` is not set, the `/narrate` endpoint still works — it returns `narration_text` with `audio_url: null`.

---

## 📁 Project structure

```
chatbot_LLM/
│
├── data/
│   ├── raw/                        ← 70 Wikipedia .txt files with metadata headers
│   └── chunks/
│       └── all_chunks.json         ← processed text chunks (auto-generated)
│
├── pipeline/
│   ├── audit_raw_files.py          ← scans raw text files for issues
│   ├── build_chunks.py             ← converts raw text → JSON chunks
│   └── build_vectordb.py           ← embeds chunks → ChromaDB vector database
│
├── core/
│   ├── __init__.py
│   └── rag_engine.py               ← Groq client + ChromaDB, handles RAG logic
│
├── models/
│   ├── db_models.py                ← SQLAlchemy model: ArtifactNarration
│   └── vectordb/                   ← ChromaDB persistent storage (auto-generated)
│
├── routers/
│   └── narration.py                ← POST /api/artifacts/{id}/narrate endpoint
│
├── services/
│   └── voice_service.py            ← Groq narration + ElevenLabs TTS pipeline
│
├── scripts/
│   └── pregenerate_narrations.py   ← Batch pre-generate narrations for all monuments
│
├── database.py                     ← SQLite session / engine for narration cache
├── main.py                         ← FastAPI server entry point
├── requirements.txt                ← Python dependencies
├── Dockerfile                      ← container build file
└── README.md                       ← this file
```

---

## 🚀 Installation — step by step

### Step 1: Create and activate a virtual environment

```bash
python -m venv venv
```

```bash
# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\activate

# Windows (CMD)
venv\Scripts\activate.bat
```

---

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

> [!NOTE]
> This installs ChromaDB, sentence-transformers, Groq SDK, FastAPI, and multilingual tools.  
> No large model downloads — the LLM runs on Groq's servers.

---

### Step 3: Set your Groq API key

```bash
export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### Step 4: Prepare the data

```bash
python scripts/step1_prepare_data.py
```

**Output:** Creates `data/chunks/all_chunks.json` with ~5,493 text chunks from 70 Wikipedia articles.

---

### Step 5: Build the vector database

```bash
python scripts/step2_build_vectordb.py
```

**Output:** Creates `models/vectordb/` with embedded vectors for all chunks. Takes ~20 seconds on CPU.

---

### Step 6: Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Server:** `http://localhost:8001`  
**API docs:** `http://localhost:8001/docs`

---

## 🐳 Running with Docker

Make sure the `.env` file at the project root contains your `GROQ_API_KEY`, then:

```bash
docker-compose up --build
```

```bash
# Run in background
docker-compose up --build -d

# Stop
docker-compose down
```

> [!NOTE]
> No NVIDIA runtime or GPU required for Docker deployment.

---

## 📡 API endpoints

**Base URL:** `http://localhost:8001`  
**Interactive docs:** `http://localhost:8001/docs`

---

### GET /health

Check if the server is running and the knowledge base is loaded.

**Response:**

```json
{
  "status": "ok",
  "version": "2.0.0",
  "total_chunks": 5493,
  "supported_languages": ["ar", "de", "en", "ru"],
  "response_time_ms": 1.2
}
```

---

### POST /ask

Ask any free-form question about ancient Egypt. Language is detected automatically.

**Request:**

```json
{
  "question": "مَن هو إخناتون؟",
  "topic": "pharaoh",
  "history": [
    {"role": "user", "content": "What is the capital of his era?"},
    {"role": "assistant", "content": "The capital was Amarna."}
  ]
}
```

> `topic` and `history` are optional.

**Response:**

```json
{
  "answer": "إخناتون كان فرعوناً من الأسرة الثامنة عشرة...",
  "sources": [
    {
      "topic_name": "Amarna Period",
      "topic_id": "amarna_period",
      "category": "historical_era",
      "section": "Introduction",
      "period": "New Kingdom, c. 1353–1336 BC",
      "location": "Amarna, Egypt",
      "similarity": 0.8124
    }
  ],
  "latency_ms": 1250.4
}
```

---

### POST /describe

Generate a visitor-friendly description of any Egyptian monument.

**Request:**

```json
{
  "monument_name": "Karnak Temple",
  "language": "en"
}
```

**Response:**

```json
{
  "description": "Karnak Temple is one of the largest religious complexes ever built..."
}
```

---

### POST /identify

For the **computer vision team**: takes a detected monument name (English) + a question (any language).

**Request:**

```json
{
  "monument_name": "Great Pyramid of Giza",
  "question": "When was it built and how tall is it?"
}
```

**Response:**

```json
{
  "answer": "The Great Pyramid of Giza was built around 2600 BC...",
  "monument": "Great Pyramid of Giza",
  "latency_ms": 980.1
}
```

---

### POST /api/artifacts/{artifact_id}/narrate

**Voice Tour Guide** — Generate an AI narration for an artifact and convert it to speech.

Narrations are cached in the local SQLite database (`khemet.db`). Subsequent calls for the same `artifact_id` + `language` pair return the cached result instantly with `cached: true`.

**Request body:**

```json
{
  "language": "en",
  "artifact_name": "Mask of Tutankhamun",
  "artifact_description": "Golden burial mask of the pharaoh Tutankhamun."
}
```

`language` must be `"en"` or `"ar"`.

**Response (first call — generates narration):**

```json
{
  "narration_text": "Before you stands one of the most iconic treasures of ancient Egypt...",
  "audio_url": "/static/audio/narration_<id>_en.mp3",
  "cached": false
}
```

**Response (subsequent calls — cache hit):**

```json
{
  "narration_text": "Before you stands one of the most iconic treasures of ancient Egypt...",
  "audio_url": "/static/audio/narration_<id>_en.mp3",
  "cached": true
}
```

`audio_url` is `null` if `ELEVENLABS_API_KEY` is not set or TTS fails. The service still returns `narration_text` in that case.

**curl example:**

```bash
curl -X POST http://localhost:8001/api/artifacts/test-uuid-123/narrate \
  -H "Content-Type: application/json" \
  -d '{"language": "en", "artifact_name": "Mask of Tutankhamun",
       "artifact_description": "Golden burial mask of the pharaoh Tutankhamun."}'
```

---

## 🔗 CV team integration

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User uploads   │────▶│   CV model       │────▶│  POST /identify  │
│   monument image │     │   identifies     │     │  RAG API         │
│                  │     │   "Great Sphinx" │     │                  │
└─────────────────┘     └─────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │   Frontend       │
                                                 │   displays       │
                                                 │   answer         │
                                                 └─────────────────┘
```

**Python example:**

```python
import requests

response = requests.post("http://localhost:8001/identify", json={
    "monument_name": "Great Sphinx of Giza",
    "question": "What is the history of this monument?"
})

data = response.json()
print(data["answer"])
```

**JavaScript example:**

```javascript
const response = await fetch("http://localhost:8001/identify", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    monument_name: "Karnak Temple",
    question: "What is the history of this monument?"
  })
});

const data = await response.json();
console.log(data.answer);
```

---

## 📚 How to add more Egyptian data

The knowledge base can be expanded **without any model changes**.

### 1. Create a new text file in `data/raw/`

```
topic_id: karnak_temple
topic_name: Karnak Temple
category: monument
period: New Kingdom, c. 1550–1070 BC
location: Luxor, Egypt

---

Your Wikipedia or reference text goes here...
```

**Valid categories:** `monument`, `pharaoh`, `historical_era`, `religion`, `daily_life`, `art`, `geography`, `artifact`

### 2. Re-process and rebuild

```bash
python scripts/step1_prepare_data.py
python scripts/step2_build_vectordb.py
```

### 3. Restart the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

---

## ⚠️ Common errors and fixes

### `GROQ_API_KEY not set` / authentication error

**Fix:** Set the environment variable before starting:

```bash
export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### `ValueError: Collection 'egyptian_knowledge' does not exist`

**Fix:** The vector database has not been built yet. Run:

```bash
python scripts/step2_build_vectordb.py
```

---

### `ModuleNotFoundError: No module named 'step3_rag_engine'`

**Fix:** Run the server from inside the `chatbot_LLM/` directory:

```bash
cd chatbot_LLM
uvicorn main:app --host 0.0.0.0 --port 8001
```

---

### Port 8001 already in use

```bash
# Linux/macOS
lsof -ti:8001 | xargs kill -9

# Windows
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

---

## 🧪 Quick test

```bash
# Health check
curl http://localhost:8001/health

# Ask a question
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who built the Great Pyramid?"}'

# Describe a monument
curl -X POST http://localhost:8001/describe \
  -H "Content-Type: application/json" \
  -d '{"monument_name": "Great Sphinx of Giza", "language": "en"}'

# CV team endpoint
curl -X POST http://localhost:8001/identify \
  -H "Content-Type: application/json" \
  -d '{"monument_name": "Great Pyramid", "question": "Who built it?"}'
```

---

## 📞 Contact and support

| Role               | Name          | Email       |
|--------------------|---------------|-------------|
| AI/ML Lead         | _____________ | ___@___.com |
| Backend Lead       | _____________ | ___@___.com |
| CV Team Lead       | _____________ | ___@___.com |
| Frontend Lead      | _____________ | ___@___.com |
| Project Supervisor | _____________ | ___@___.com |

---

<p align="center">
  <i>Egyptian AI Guide — Graduation Project 2026</i>
</p>
