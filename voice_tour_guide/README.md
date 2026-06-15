# KHEMET Voice Tour Guide Service

Lightweight microservice with a single responsibility: convert Egyptian
artifact description text into MP3 audio using ElevenLabs TTS.

All caching and database operations are handled by the Node.js backend.

---

## Overview

| Property     | Value                                   |
|--------------|-----------------------------------------|
| Port         | `8003` (internal — not exposed to host) |
| Language     | Python 3.11                             |
| Framework    | FastAPI                                 |
| TTS Engine   | ElevenLabs API (`eleven_multilingual_v2`) |

---

## Architecture

```
Frontend Request
      │
      ▼
POST /api/voice/artifacts/{id}/narrate  (Node.js backend — port 3000)
      │
      ├─ Check artifact_narrations cache in PostgreSQL
      │
      │  Cache hit → return immediately
      │
      │  Cache miss ─────────────────────────────────────┐
      │                                                   ▼
      │                              POST /generate  (voice-tour-guide — port 8003, internal)
      │                                                   │
      │                                            ElevenLabs TTS API
      │                                                   │
      │                                        Returns { audio_url } to backend
      │                                                   │
      └─ Backend saves row to PostgreSQL ─────────────────┘
             │
             ▼
      Returns { narration_text, audio_url, cached } to frontend
```

> **Note:** The voice-tour-guide service is not exposed to the host machine
> directly. It is accessed internally by the Node.js backend only.
> To test the endpoint, use the backend route:
> ```
> POST http://localhost:3000/api/voice/artifacts/{id}/narrate
> ```

---

## API Endpoints

### `POST /generate`

Converts text to an MP3 audio file using ElevenLabs TTS.
**Called internally by the Node.js backend only — not by the frontend.**

**Request Body**

| Field         | Type   | Description                              |
|---------------|--------|------------------------------------------|
| `artifact_id` | string | Artifact UUID (used for MP3 filename)    |
| `language`    | string | `"en"` or `"ar"`                         |
| `text`        | string | Text to convert to speech                |

**Response**

| Field       | Type           | Description                       |
|-------------|----------------|-----------------------------------|
| `audio_url` | string or null | Relative URL to the generated MP3 |

**Example (internal backend call)**

```bash
curl -X POST http://localhost:8003/generate \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_id": "some-uuid",
    "language": "en",
    "text": "A golden burial mask of the pharaoh Tutankhamun."
  }'
```

---

### `GET /health`

Returns the service status.

```json
{ "status": "ok", "service": "voice_tour_guide", "port": 8003 }
```

---

## Environment Variables

| Variable              | Required    | Description                                          |
|-----------------------|-------------|------------------------------------------------------|
| `ELEVENLABS_API_KEY`  | ✅ Required  | ElevenLabs API key (returns `null` audio if missing) |
| `ELEVENLABS_VOICE_EN` | ⚠️ Optional  | ElevenLabs voice ID for English (default: George)   |
| `ELEVENLABS_VOICE_AR` | ⚠️ Optional  | ElevenLabs voice ID for Arabic  (default: Sarah)    |

> **No DATABASE_URL required.** This service has zero database access.
> All caching is handled by the Node.js backend.

---

## TTS Pipeline

| Engine  | Description                                              |
|---------|----------------------------------------------------------|
| Primary | ElevenLabs API (`eleven_multilingual_v2`)                |
| Fallback | None — returns `audio_url: null` if ElevenLabs fails   |

**Default voices (free-tier compatible):**

| Language | Voice  | Voice ID                 | Style                         |
|----------|--------|--------------------------|-------------------------------|
| English  | George | `JBFqnCBsd6RMkjVDRZzb`  | Warm, captivating storyteller |
| Arabic   | Sarah  | `EXAVITQu4vr4xnSDxMaL`  | Mature, reassuring, confident |

---

## Running with Docker

```bash
# Start only this service (no DB dependency)
docker compose up --build voice-tour-guide

# Follow logs
docker compose logs -f voice-tour-guide
```

> **Note:** The voice-tour-guide service is not exposed to the host machine directly.
> It is accessed internally by the Node.js backend only.
> To test the endpoint use the backend route:
> ```
> POST http://localhost:3000/api/voice/artifacts/{id}/narrate
> ```

---

## File Structure

```
voice_tour_guide/
├── main.py                        # FastAPI application entry point
├── generate.py                    # POST /generate endpoint (TTS only)
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container definition
└── services/
    └── elevenlabs_service.py      # TTS — ElevenLabs API
```

---

## Project Context

This service is part of the **KHEMET** platform, developed as a graduation project
at the **Faculty of Computers and Artificial Intelligence, University of Sadat City**,
under the supervision of **Dr. Sara Shehab** and **Eng. Abanoub Shawky**.
