# Friday Mark 2 🤖

> A production-grade, hybrid LLM personal assistant — the smarter successor to JARVIS.  
> Built for real autonomy: multi-agent planning, 6-layer persistent memory, hybrid cloud+local LLM execution, and a strict layered architecture designed for long-term scale.

---

## What Friday Can Do Right Now ✅

### 🧠 Hybrid LLM with Automatic Fallback
Friday runs on a **two-slot LLM system**. Slot 1 is any OpenAI-compatible cloud API (Gemini, DeepSeek, OpenRouter, Groq, Claude via OpenRouter, etc.). Slot 2 is always-available local **Ollama**. If Slot 1 fails — auth error, rate limit, network timeout — Friday silently falls back to Ollama with zero interruption to the user. Callers never know which slot served the request.

### ⚡ 3-Tier Smart Router
Every user message is classified and routed to the right handler automatically:

| Tier | Handler | When Used | Target Latency |
|------|---------|-----------|----------------|
| **Simple** | `SimpleHandler` | Greetings, facts, calendar lookups. Now fully context-aware via lightweight LLM for chat. | < 500ms |
| **Medium** | `MediumHandler` (AgentLoop) | Requires 1-3 tools (web search, calculator, memory) | < 2s |
| **Complex** | `MultiAgentPlanner` + ExecutionEngine | Multi-step research, parallel agent coordination | Background |

Classification uses an **LLM router** with a **regex fallback classifier** — if the LLM is slow, regex takes over instantly.

### 🗃️ 6-Layer Persistent Memory
Friday's memory is not a flat conversation log. It's a structured, tiered system stored in SQLite:

| Layer | What it stores | Lifetime |
|-------|---------------|----------|
| 1 — Working | Current task context | Session only |
| 2 — Short-Term | Recent conversation | Hours–Days |
| 3 — Episodic | Events, calendar facts with dates | Days–Weeks |
| 4 — Semantic | General user knowledge & facts | Weeks–Months |
| 5 — Procedural | Habits and learned patterns | Months |
| 6 — Profile | Core identity: name, age, goals | Permanent |

Memory is automatically written in the background after **every** user message via an async pipeline — the user never waits for it.

### 🔍 Hybrid Memory Search
Memory retrieval uses a **vector + BM25 keyword hybrid search**:
- **sqlite-vec** for cosine-similarity vector search (semantic understanding)
- **SQLite FTS5** for BM25 keyword search (exact term matching)
- **LIKE fallback** when FTS5 returns nothing
- **Redis caching** on top of all search results for speed
- Results are automatically injected into every LLM call as context

### 📅 Calendar & Conflict Detection
- Friday **automatically extracts events** from natural language (e.g. "I have a meeting on Friday at 3pm")
- Stores them in the episodic memory layer with start/end times
- Runs a **mathematical conflict linter** to detect overlapping events
- Injects an **[ABSOLUTE CONTINUOUS ITINERARY]** into every LLM call so Friday always knows your upcoming schedule
- Sends **day-before reminders** proactively when you next talk to Friday

### 🪞 Self-Learning via Reflection Agent
Every 12 messages, a background **Reflection Agent** silently scans recent conversation history and extracts stable user facts/preferences (occupation, tone, response style, etc.) and saves them to the profile — without any user prompting.

### 🗜️ Token-Aware Context Compaction
When conversation history gets long, Friday compacts it using a **3-stage progressive summarization**:
1. **Stage 1**: Full chunked summarization (preserves UUIDs, hashes, file paths exactly)
2. **Stage 2**: Partial — excludes oversized messages, summarizes the rest
3. **Stage 3**: Hard text fallback — guaranteed to never crash

This runs in the background after each response. Zero latency impact on the user.

### 🔐 Feedback & Preference Learning
Friday learns how you want to be addressed, your response style, emoji preference, and tone — either through natural conversation (Reflection Agent) or instant pattern detection (FeedbackDetector).

### 🖥️ Terminal Interface (Active)
The only currently active interface is the terminal (`python terminal_chat.py`). Fully functional with history, memory, tools, and all routing tiers.

---

## What Has Been Built But Is Not Fully Wired Yet ⚠️

### Multi-Agent Execution Engine
The full `execution/` layer is architected and coded:
- `ExecutionEngine` — async task runner
- `ParallelExecutor` — `asyncio.gather` for concurrent agents
- `SeriesExecutor` — chain agents where output feeds next
- `MemoryAwareExecutor` — pre-checks memory before running a step
- `LearningEngine` — records success/failure post-execution
- `ExecutionStateManager` — tracks all running executions

**Status**: The `MultiAgentPlanner` (Complex tier handler) calls the engine, but **specialized agents** (`researcher`, `telegram`, `whatsapp`) are empty stubs — they inherit from `BaseAgent` but have no real implementation yet.

### Awareness System
The full awareness layer is designed:
- `ContextTracker`, `ExecutionMonitor`, `WorldModel`, `StatusReporter`

**Status**: The semantic memory, calendar, reminders, and profile data are now **fully injected into every LLM call across all tiers** (Simple, Medium, Complex) via the `ContextAssembler`. The execution and world model features (like weather/time) are still being wired.

### Background Scheduler & Goal Tracker
`scheduler.py` (APScheduler-based) and `goal_tracker.py` are implemented.

**Status**: Not yet started on server boot — scheduler is defined but not launched in `terminal_chat.py`.

### Redis Embedding Cache
`cache/embedding_cache.py` and `cache/redis_client.py` exist.

**Status**: Redis search caching is active in `HybridSearcher`. Embedding caching is written but **not yet injected** into `EmbeddingManager`.

---

## What Has NOT Been Integrated Yet ❌

| Feature | Files | Status |
|---------|-------|--------|
| **Telegram Bot** | `interfaces/messaging/telegram/adapter.py` | Empty stub — `# TODO: implement` |
| **WhatsApp Webhook** | `interfaces/messaging/whatsapp/adapter.py` | Empty stub |
| **LiveKit Voice Interface** | `interfaces/livekit/adapter.py` | Empty stub |
| **FastAPI REST Server** | `interfaces/api/server.py` | Empty stub |
| **Researcher Agent** | `agents/specialized/researcher/agent.py` | Exists, not implemented |
| **Google Calendar/Gmail** | `background/google_workspace.py` | Planned, not started |
| **MCP Protocol Tools** | `tools/mcp/__init__.py` | Placeholder only |
| **Browser Tools** | `tools/browser/__init__.py` | Placeholder only |
| **Memory Promotion Engine** | `memory/promotion.py` | Designed in architecture, file not yet created |
| **Tests** | `friday/tests/` | Folder exists, no tests written |

---

## Known Flaws & Limitations 🐛

1. **Redis is optional but silently fails** — if Redis isn't running, the search cache silently no-ops. This is by design, but no warning is shown to the user.
2. **Complex tier is untested end-to-end** — the `MultiAgentPlanner` → `ExecutionEngine` → specialized agent chain has no real agents at the end of it.
3. **No tests** — the `friday/tests/` folder is empty. `test_full_integration.py` and `test_llm_slots.py` exist at the root but are standalone scripts, not proper pytest suites.
4. **Tool registry auto-discovery is designed but partial** — `tools/registry.py` has the pattern, but not all tool categories are populated (browser, MCP, memory tools).
5. **Single DB connection (no pooling)** — `db_manager.get_connection()` returns a single connection without a pool, which will be a bottleneck under any concurrent load (e.g., when the API or Telegram adapters are added).
6. **Context compaction summary drift** — the `_summary_cache` is stored in the `meta` table but is never invalidated or versioned. If the model changes, old summaries in a different "style" will persist.
7. **No authentication or rate limiting** — the FastAPI server stub has no auth layer yet. This needs to be addressed before any public interface is deployed.

---

## Architecture Overview

```
User Input
    │
    ▼
SmartRouter (platform-agnostic)
    │
    ├── Simple  ──► SimpleHandler   (DB + memory context, no tools)
    ├── Medium  ──► MediumHandler   (AgentLoop, up to 3 tools)
    └── Complex ──► MultiAgentPlanner ──► ExecutionEngine
                         │                     ├── ParallelExecutor
                         │                     └── SeriesExecutor
                         └── agents/specialized/<name>/agent.py

AgentLoop (the core brain)
    ├── SlottedProvider (API Slot → Ollama fallback)
    ├── HybridSearcher (vector + BM25 + Redis cache) — runs before every LLM call
    ├── FactStore (calendar/episodic memory + conflict detection)
    ├── FeedbackDetector (instant preference learning)
    ├── Reflection Agent (background, every 12 messages)
    └── Context Compaction (3-stage, runs after response)
```

**Strict Layer Rule**: `interfaces → router → agents → tools/memory/reasoning/awareness/execution`. Nothing imports upward. The core brain never knows which interface is active.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Local LLM | Ollama (llama3.1:8b default) |
| Cloud LLM | Any OpenAI-compatible API (Gemini, DeepSeek, OpenRouter, Groq, etc.) |
| Database | SQLite + sqlite-vec (vector extension) + FTS5 (keyword search) |
| Cache | Redis (optional) |
| Async runtime | asyncio |
| Config | YAML + `.env` |
| API framework | FastAPI + Uvicorn (stub, not active) |

---

## Setup & Running

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running with `llama3.1:8b` pulled
- Redis (optional — search caching degrades gracefully without it)

### Install
```bash
# Clone the repo
git clone https://github.com/pranithvelar/Friday_mark2.git
cd Friday_mark2

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configure
```bash
# Copy the example env file
copy .env.example .env

# Edit .env — add your API key if you want cloud LLM (optional)
# Leave LLM_API_KEY empty to run fully local with Ollama
```

### Run
```bash
python terminal_chat.py
```

---

## Project Structure

```
friday_mark2/
├── terminal_chat.py          # Entry point — starts the terminal interface
├── requirements.txt
├── .env.example              # Template for environment variables
├── friday/
│   ├── agents/               # All intelligent entities (AgentLoop + specialized agents)
│   ├── awareness/            # Real-time context (execution status, world model)
│   ├── background/           # Autonomous scheduled tasks, goal tracker
│   ├── cache/                # Redis caching for search and embeddings
│   ├── config/               # friday.yaml + settings
│   ├── execution/            # Multi-agent task runner with parallel/series support
│   ├── interfaces/           # I/O adapters: terminal ✅ | telegram/whatsapp/API ❌ (stubs)
│   ├── llm/                  # Slotted LLM provider (API → Ollama fallback)
│   ├── memory/               # 6-layer persistent memory + pipeline
│   ├── reasoning/            # Chain-of-thought, reflection, explainability
│   ├── router/               # Smart router: intent classification + tier dispatch
│   ├── search/               # Hybrid vector + BM25 search
│   ├── security/             # Input filters
│   ├── tools/                # Tool registry + general/calendar/browser/mcp tools
│   └── utils/                # Logger, batching helpers
```

---

## Roadmap

- [ ] Implement Researcher specialized agent (web research + report generation)
- [ ] Wire Telegram adapter
- [ ] Wire FastAPI REST server with auth
- [ ] Start background scheduler on boot
- [ ] LiveKit voice interface (STT + TTS)
- [ ] Write test suite (pytest)
- [ ] Google Calendar/Gmail integration
- [ ] MCP protocol tool support
- [ ] DB connection pooling for concurrent interfaces

---

## Inspiration

Friday Mark 2 is the successor to my JARVIS project — rebuilt from scratch with production architecture in mind. The goal is a personal AI assistant that genuinely learns, plans autonomously, and degrades gracefully without internet access.

---

*Built by [@pranithvelar](https://github.com/pranithvelar)*
