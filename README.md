# Friday Mark 2 🤖

> A production-grade, hybrid LLM personal assistant — the smarter successor to JARVIS.  
> Built for real autonomy: multi-agent planning, 6-layer persistent memory, hybrid cloud+local LLM execution, chain-of-thought reasoning, and a strict layered architecture designed for long-term scale.

---

## What Friday Can Do Right Now ✅

### 🧠 Hybrid LLM with Automatic Fallback
Friday runs on a **two-slot LLM system**. Slot 1 is any OpenAI-compatible cloud API (Gemini, DeepSeek, OpenRouter, Groq, Claude via OpenRouter, etc.). Slot 2 is always-available local **Ollama**. If Slot 1 fails — auth error, rate limit, network timeout — Friday silently falls back to Ollama with zero interruption to the user. Callers never know which slot served the request.

### ⚡ 4-Tier Smart Router
Every user message is classified and routed to the right handler automatically:

| Tier | Handler | When Used | Target Latency |
|------|---------|-----------|----------------|
| **Simple** | `SimpleHandler` | Greetings, facts, calendar lookups. Fully context-aware via lightweight LLM. | < 500ms |
| **Medium** | `MediumHandler` (AgentLoop) | Requires 1-3 discrete tool calls (web search, calculator, memory) | < 2s |
| **Complex** | `MultiAgentPlanner` + `ExecutionEngine` | Any multi-step project where the intent is clear enough to plan | Background |
| **Clarify** | Graceful Fallback | Request is genuinely too vague to form any plan. Max 3 questions, then auto-escalates to Complex. | < 1s |

Classification uses an **LLM router** (now intent-first, not trigger-phrase-first) with a **regex fallback classifier** — regex takes over instantly if the LLM is slow. The Clarify tier has a 3-strike auto-escalation: after 3 consecutive clarification questions in a session, Friday stops asking and starts building.

### 🔧 Tool Registry & Auto-Discovery
Every tool in `friday/tools/` that subclasses `BaseTool` is **automatically discovered and registered** into the `AgentLoop` at startup:
- Zero configuration — just drop a file in the right folder
- `BaseTool.to_schema()` auto-generates the LLM-facing JSON schema
- Duplicate names are skipped (existing BRAIN tools are never overridden)
- Registry is stored on `loop._tool_registry` so both `AgentLoop` and `ExecutionEngine` can dispatch through it
- Scope field (`"general"` vs `"brain"`) controls which tiers see the tool

### 🔗 Chain-of-Thought Reasoning (All Tiers)
Every LLM response in Friday — whether answering directly, calling a tool, or generating a multi-step plan — is now preceded by a private `<thought>` reasoning block:

```
<thought>
1. What is the user actually asking for?
2. Scan the tool list. Is there a tool that covers this? YES → call it. NO → state limitation honestly.
3. Does memory context directly answer the question, or is it just background?
</thought>
```

The thought block is **preserved in session history** (helps multi-step reasoning) but is **always stripped before showing output to the user**. The `MultiAgentPlanner` uses a separate CoT block for strategy before generating the execution plan JSON.

### 🗃️ 6-Layer Persistent Memory (All Layers Now Implemented)
Friday's memory is a structured, tiered system stored in SQLite — all 6 layers are now fully implemented and wired:

| Layer | What it stores | Lifetime | Status |
|-------|---------------|----------|--------|
| 1 — Working | Current task context | Session only | ✅ Active |
| 2 — Short-Term | Recent conversation recall | Hours–Days | ✅ Active |
| 3 — Episodic | Events, calendar facts with dates | Days–Weeks | ✅ Active |
| 4 — Semantic | Inferred stable user facts (with confidence decay) | Weeks–Months | ✅ Active |
| 5 — Procedural | Behavioral patterns + self-learned lessons from failures | Months | ✅ Active |
| 6 — Profile | Core identity: name, preferences, explicit stated facts | Permanent | ✅ Active |

Memory is automatically written in the background after **every** user message via an async pipeline — the user never waits for it.

### ⚖️ Memory Promotion Engine (Now Wired)
The `PromotionEngine` is now **fully wired** into the memory pipeline. After every message, memories that have been seen and reinforced enough times are automatically promoted from short-term (Layer 2) to longer-term storage. Stale entries are pruned. This was previously initialized but not connected — it is now.

### 📉 Temporal Memory Decay (Layer 4)
Semantic facts in Layer 4 now **decay over time** via a background `MemoryDecayWatcher` that runs every 6 hours:
- Uses **exponential decay** with a **45-day half-life** (mirrors cognitive forgetting curves)
- Facts below a confidence threshold become invisible in context injection (but are NOT deleted)
- **Immune** from decay: facts the user explicitly stated (`source='stated'`), and facts confirmed 5+ times
- Returns nothing to the user — completely silent background job

### 🔎 Self-Critique & Self-Learning (ReflectionAgent — Layer 5)
`friday/reasoning/reflection.py` was previously an empty stub. It is now a fully implemented **Self-Critique Engine**:

- **On tool failure**: when the AgentLoop exhausts retries and can't find a valid tool, `ReflectionAgent.on_tool_failure()` fires as a background task — it diagnoses WHY it failed and writes a behavioral lesson to Layer 5
- **On user correction**: a zero-cost **regex gate** (`check_for_correction()`) runs before every LLM call. If the user says something like "that's wrong" or "not what I asked", a single diagnostic LLM call fires (temperature=0 for determinism) and extracts a structured lesson: `{trigger, fix, confidence}`
- Lessons are written to `ProceduralMemory` (Layer 5), which is already read by `ContextAssembler` on every request — **no new injection wiring needed**
- Fire-and-forget via `asyncio.create_task()` — zero latency impact on the user

> **The key distinction**: `_maybe_reflect()` in `agent.py` learns about the **USER** (facts, preferences) every 12 messages. `ReflectionAgent` learns about **FRIDAY'S OWN BEHAVIOR** (tool errors, corrections) and stores lessons so they don't repeat.

### 📅 Calendar & Conflict Detection (via EventEngine)
- Friday **automatically extracts events** from natural language (e.g. "I have a meeting on Friday at 3pm")
- Stores them in the episodic memory layer with start/end times
- Runs a **mathematical conflict linter** to detect overlapping events
- The `EventEngine` hot-path (pure regex + SQLite, ~10ms, zero LLM) detects events before any LLM call
- Injects an **[ABSOLUTE CONTINUOUS ITINERARY]** into every LLM call via the `LiveContextState`
- Sends **day-before reminders** proactively when you next talk to Friday

### 🏗️ Project Chronicle
A JARVIS-style per-project documentation system:
- `ProjectRegistry` — manages project folders in `memory/projects/`
- `ProjectClassifier` — classifies each message to the relevant project
- `ProjectDreamer` — runs every 30 minutes in background, synthesizes new documentation from recent activity
- **Safety system**: `ChronicleCircuitBreaker` (trips after 5 errors), `SafeChronicle` (atomic writes + per-project locks), startup health check + repair

### 🪞 Self-Learning via Background Agents (Three Independent Learners)
Friday now has **three separate background learning systems** that never block the user:

| Agent | Trigger | Writes to | Purpose |
|-------|---------|-----------|---------|
| **Reflection Agent** (in `_maybe_reflect`) | Every 12 messages | Layer 6 (Profile) | Extracts stable user facts from conversation |
| **Knowledge Extractor** | Every 20 messages | Layers 4 + 5 | Deep extraction of semantic facts and behavioral patterns |
| **ReflectionAgent** (self-critique) | Tool failure / user correction | Layer 5 (Procedural) | Learns from Friday's own mistakes |

### ⏱️ Live Context Loop (Always-On Brain Awareness)
A background loop refreshes Friday's awareness every **8 seconds**, independently of user input:
- Current time (timezone-aware)
- Upcoming calendar events / itinerary  
- Running execution status
- Day-before reminders
- Pending plan approvals
- Scheduling conflicts

This is injected into `AgentLoop._build_system_prompt()` on every LLM call. The brain is **always aware** even between conversations.

### 🗜️ Token-Aware Context Compaction
When conversation history gets long, Friday compacts it using a **3-stage progressive summarization**:
1. **Stage 1**: Full chunked summarization (preserves UUIDs, hashes, file paths exactly)
2. **Stage 2**: Partial — excludes oversized messages, summarizes the rest
3. **Stage 3**: Hard text fallback — guaranteed to never crash

This runs in the background after each response. Zero latency impact on the user.

### 🔐 Preference Learning & Address Override
Friday respects how you want to be addressed — with a strict priority: the `address_as` preference (explicitly set by user, e.g. "call me Sir") **always wins** over the stored real name. The system prompt enforces this with `ALWAYS address the user as "X" — never use their real name, never deviate`.

### 🔍 Search: FTS5 Core + RAG Agent for Deep Research
The core memory system uses **pure SQLite FTS5** for fast keyword search (<10ms). For deep document research and vector similarity search, a **specialized RAG Agent** (`agents/specialized/rag/`) handles the heavy lifting:
- Cosine-similarity vector search via `embedding_manager.py`
- Maximal Marginal Relevance (MMR) ranking
- Optional Redis caching for embeddings

### 🖥️ Terminal Interface (Active)
The only currently active interface is the terminal (`python terminal_chat.py`). Fully functional with all tiers, memory, tools, live context, and all learning systems.

---

## What Has Been Built But Is Not Fully Wired Yet ⚠️

### Multi-Agent Execution Engine
The full `execution/` layer is fully implemented and now dispatches real tools:
- `ExecutionEngine` — async task runner with **real tool dispatching** (not a stub)
- **4-stage dispatch**: direct name match → `_CATEGORY_TOOL_MAP` lookup → `_llm_execute_step` reasoning → stub fallback
- `_llm_execute_step` — when no real tool is registered for a step, the LLM reasons through it directly
- `ParallelExecutor` — `asyncio.gather` for concurrent agents
- `SeriesExecutor` — chain agents where output feeds next
- `MemoryAwareExecutor` — pre-checks memory before running a step
- `LearningEngine` — records success/failure post-execution
- `ExecutionStateManager` — tracks all running executions

**Status**: Plans are generated, approved, and dispatched to the `ExecutionEngine` which now routes to real tools. **Specialized agents** (`researcher`, `telegram`, `whatsapp`) are still empty stubs — complex tasks resolve via LLM reasoning steps until those agents are built.

### Background Scheduler & Goal Tracker
`scheduler.py` (APScheduler-based) and `goal_tracker.py` are implemented. `ProactiveEventsWatcher`, `GoogleWorkspaceWatcher`, and the new `MemoryDecayWatcher` are all registered.

**Status**: The scheduler is launched on startup via `asyncio.create_task()`. Google Workspace integration (`GoogleWorkspaceWatcher`) is a no-op stub — OAuth wiring not started.

---

## What Has NOT Been Integrated Yet ❌

| Feature | Files | Status |
|---------|-------|--------|
| **Telegram Bot** | `interfaces/messaging/telegram/adapter.py` | Empty stub |
| **WhatsApp Webhook** | `interfaces/messaging/whatsapp/adapter.py` | Empty stub |
| **LiveKit Voice Interface** | `interfaces/livekit/adapter.py` | Empty stub |
| **FastAPI REST Server** | `interfaces/api/server.py` | Empty stub — no auth |
| **Researcher Agent** | `agents/specialized/researcher/agent.py` | Exists, not implemented |
| **Google Calendar/Gmail** | `background/google_workspace.py` | Stub — OAuth not wired |
| **MCP Protocol Tools** | `tools/mcp/__init__.py` | Placeholder only |
| **Browser Tools** | `tools/browser/__init__.py` | Placeholder only |
| **Tests** | `friday/tests/` | Folder exists, no tests written |

---

## Known Flaws & Limitations 🐛

1. **Specialized agents are stubs** — `researcher`, `telegram`, `whatsapp` agents exist but don't do real work. Complex plan steps resolve via LLM reasoning (`_llm_execute_step`) until these are built.
2. **No tests** — `friday/tests/` is empty. `test_full_integration.py` at root is a standalone script, not a pytest suite.
3. **Tool registry auto-discovery covers `BaseTool` subclasses only** — any tool not subclassing `BaseTool` won't be auto-discovered. Browser and MCP slots are defined but empty.
4. **Single DB connection (no pooling)** — `db_manager.get_connection()` returns a single connection without a pool, which will bottleneck under concurrent load when API/Telegram adapters are added.
5. **Context compaction summary drift** — `_summary_cache` is stored in the `meta` table but never invalidated on model change. Old summaries from a different model style will persist.
6. **No authentication or rate limiting** — The FastAPI server stub has no auth layer yet.
7. **ReflectionAgent has no dedup window** — Multiple user corrections in quick succession can fire multiple diagnostic LLM calls (mitigated by the `_running` lock, but only one at a time, not rate-limited over a longer window).

---

## Architecture Overview

```
User Input
    │
    ├── [BACKGROUND] MemoryPipeline.process()          → FTS5 index + PromotionEngine (non-blocking)
    │
    ├── [BACKGROUND] ReflectionAgent.check_for_correction() → regex gate, zero cost if no match
    │
    ├── ContextAssembler.build()                        → L2+L3+L4+L5+L6 bundle
    │
    ▼
SmartRouter (platform-agnostic, context-aware)
    │
    ├── CLARIFY  ──► Ask 1 question (max 3, then auto-escalate to Complex)
    ├── SIMPLE   ──► SimpleHandler(bundle)   — LLM + full context, <500ms
    ├── MEDIUM   ──► MediumHandler(bundle)   — AgentLoop, up to 3 tools, <2s
    └── COMPLEX  ──► MultiAgentPlanner(bundle)
                          │     <thought> → plan JSON → _format_plan_for_approval()
                          │     User approves → fire_plan()
                          └── ExecutionEngine → agents/specialized/<name>/agent.py

AgentLoop (the core brain)
    ├── LiveContextState (8s refresh: time, itinerary, execution status, conflicts)
    ├── SlottedProvider (API Slot → Ollama fallback)
    ├── <thought> CoT block (stripped before showing user, kept in history)
    ├── FTSSearcher (keyword search, <10ms) — runs before every LLM call
    ├── FactStore (calendar/episodic memory + conflict detection)
    ├── Reflection Agent _maybe_reflect() (background, every 12 msgs → Layer 6)
    ├── Knowledge Extractor (background, every 20 msgs → Layers 4+5)
    ├── ReflectionAgent self-critique (background, on failure/correction → Layer 5)
    └── Context Compaction (3-stage, background, after response)

Background Scheduler (always running)
    ├── ProactiveEventsWatcher  — day-before reminders
    ├── GoogleWorkspaceWatcher  — stub (OAuth not wired)
    └── MemoryDecayWatcher      — Layer 4 decay, every 6h, 45-day half-life

Project Chronicle (always running)
    └── ProjectDreamer loop     — synthesizes project docs every 30 minutes
```

**Strict Layer Rule**: `interfaces → router → agents → tools/memory/reasoning/awareness/execution`. Nothing imports upward. The core brain never knows which interface is active.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Local LLM | Ollama (llama3.1:8b default) |
| Cloud LLM | Any OpenAI-compatible API (Gemini, DeepSeek, OpenRouter, Groq, etc.) |
| Database | SQLite + FTS5 (keyword search) |
| Vector Search | sqlite-vec via RAG Agent (optional, isolated) |
| Cache | Redis (optional, RAG Agent only) |
| Async runtime | asyncio |
| Config | YAML + `.env` |
| API framework | FastAPI + Uvicorn (stub, not active) |

---

## Setup & Running

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running with `llama3.1:8b` pulled
- Redis (optional — only used by RAG Agent if enabled)

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
├── terminal_chat.py          # Entry point
├── requirements.txt
├── .env.example
├── friday/
│   ├── agents/
│   │   ├── friday/           # AgentLoop — the core brain
│   │   └── specialized/
│   │       ├── rag/          # ✅ RAG Agent (vector search + MMR)
│   │       ├── researcher/   # ❌ Stub
│   │       ├── telegram/     # ❌ Stub
│   │       └── whatsapp/     # ❌ Stub
│   ├── awareness/
│   │   └── live_context.py   # ✅ LiveContextState + 8s refresh loop
│   ├── background/
│   │   ├── scheduler.py      # ✅ BackgroundScheduler
│   │   ├── proactive_events.py # ✅ Day-before reminders
│   │   ├── memory_decay.py   # ✅ Layer 4 temporal decay (45-day half-life)
│   │   ├── knowledge_extractor.py # ✅ Deep L4+L5 extraction (every 20 msgs)
│   │   └── google_workspace.py # ❌ Stub
│   ├── execution/            # ✅ ExecutionEngine, Parallel/Series/MemoryAware executors
│   ├── interfaces/
│   │   ├── terminal/         # ✅ Active
│   │   └── messaging/        # ❌ Telegram/WhatsApp stubs
│   ├── llm/                  # ✅ SlottedProvider (API → Ollama fallback)
│   ├── memory/
│   │   ├── pipeline.py       # ✅ Background indexing + promotion
│   │   ├── context_assembler.py # ✅ Builds L2+L3+L4+L5+L6 bundle
│   │   ├── event_engine.py   # ✅ Hot-path event detection (~10ms)
│   │   ├── memory_core.py    # ✅ Single public API for all agents
│   │   ├── project_chronicle/ # ✅ Per-project documentation + safety system
│   │   └── layers/
│   │       ├── layer_3_episodic.py  # ✅ Calendar + conflict detection
│   │       ├── layer_4_semantic.py  # ✅ Inferred facts + confidence decay
│   │       ├── layer_5_procedural.py # ✅ Behavioral patterns + self-learned lessons
│   │       └── layer_6_profile.py   # ✅ Explicit profile
│   ├── reasoning/
│   │   └── reflection.py     # ✅ Self-critique engine (was TODO stub)
│   ├── router/
│   │   ├── smart_router.py   # ✅ 4-tier dispatch + clarify auto-escalation
│   │   ├── llm_router.py     # ✅ Intent-first classification (no trigger-phrase dependency)
│   │   └── handlers/
│   │       ├── simple_handler.py  # ✅ LLM + bundle context
│   │       ├── medium_handler.py  # ✅ AgentLoop + bundle
│   │       └── complex_handler.py # ✅ CoT planning + approval gate
│   ├── search/
│   │   ├── fts_search.py     # ✅ Core: pure FTS5 keyword search
│   │   └── indexer.py        # ✅ Text indexer for FTS5
│   └── tools/                # ✅ Tool registry + general tools
```

---

## Roadmap

- [ ] Implement Researcher specialized agent (web research + report generation)
- [ ] Wire Telegram adapter
- [ ] Wire FastAPI REST server with auth
- [ ] LiveKit voice interface (STT + TTS)
- [ ] Write test suite (pytest)
- [ ] Google Calendar/Gmail integration (OAuth)
- [ ] MCP protocol tool support
- [ ] DB connection pooling for concurrent interfaces
- [ ] ReflectionAgent dedup window (rate-limit diagnostic calls over longer window)

---

## Inspiration

Friday Mark 2 is the successor to my JARVIS project — rebuilt from scratch with production architecture in mind. The goal is a personal AI assistant that genuinely learns, plans autonomously, and degrades gracefully without internet access.

---

*Built by [@pranithvelar](https://github.com/pranithvelar)*
