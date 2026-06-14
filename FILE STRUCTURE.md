# Friday Mark 2 — Complete Architecture Implementation Plan
### Chain-of-Thought Design for Infinite Scalability

---

## Chain-of-Thought Reasoning (How I Arrived at This)

### Step 1 — Mapping the 9 Capabilities to Structural Concerns

| Capability | Where It Lives | Why |
|---|---|---|
| Real Tool Execution | `tools/` | Tools are dumb functions. They do ONE thing. They must be callable from ANY agent without importing the agent. |
| Explainability & Transparency | `reasoning/` | Every routing decision, every plan step, every tool call must log its reasoning. This is a cross-cutting module, not tied to any single layer. |
| Real Multi-Agent Coordination | `agents/` + `execution/` | Agents are the intelligence. Execution engine spawns them in parallel or series. These are two separate concerns — planning ≠ executing. |
| Advanced Reasoning | `reasoning/` | CoT, reflection, tree-of-thoughts are reasoning strategies. They belong next to explainability, not inside the agent or the router. |
| Real-Time World Awareness | `tools/general/` + `awareness/` | Web search / news = tools. The model's understanding of "what's happening NOW" = awareness module injected into context. |
| Situational Awareness / Context Model | `awareness/` | A live context snapshot (current execution, recent memory updates, what the user is focused on) injected into EVERY LLM call. |
| Agentic Self-Loop / Autonomous Goals | `background/` | Friday sets and tracks its own goals. Background scheduler checks and executes them proactively. The goal STORE lives in memory. |
| True Self-Learning | `memory/pipeline.py` | This is the most critical insight: learning must happen on EVERY input, not just on tool calls. A parallel asyncio task fires for every message. |
| Voice / Multi-Modal | `interfaces/livekit/` | LiveKit handles all STT/TTS. This is purely an interface adapter. The core brain never changes regardless of input modality. |

### Step 2 — The 3 Core Architectural Laws

**Law 1: Every layer can only import DOWNWARD.**
```
interfaces/ → router/ → agents/ → tools/
                      → memory/
                      → reasoning/
                      → awareness/
                      → execution/
```
`tools/` never imports `agents/`. `memory/` never imports `router/`. Breaking this causes circular imports and tightly couples systems that must stay independent.

**Law 2: Every input fires TWO concurrent pipelines.**
```
User input
   ├── Response Pipeline (user is waiting)
   │     Router → Handler → Agent → Reply
   └── Memory Pipeline (background, asyncio.create_task)
         Extract → Embed → Store → Promote → Update profile
```
This is how JARVIS learns from every word Tony says — not just commands.

**Law 3: Adding a new agent or tool = dropping 1 file in the right folder.**
- Drop `agents/specialized/telegram/agent.py` → telegram agent is auto-discovered
- Drop `tools/general/calculator.py` → calculator tool is auto-discovered
- Zero changes to router, execution engine, or any other file.

### Step 3 — The Decision Tree (Where Does X Go?)

```
Is it an input/output channel? (terminal, telegram, whatsapp, livekit)
   → interfaces/

Is it an intelligent entity that makes decisions and can call tools?
   → agents/specialized/

Is it a capability (a thing Friday can DO — a function with inputs/outputs)?
   → tools/

Is it about routing/classifying user intent?
   → router/

Is it about storing or retrieving information?
   → memory/

Is it about HOW Friday reasons about a problem?
   → reasoning/

Is it about WHAT Friday knows is happening right now?
   → awareness/

Is it about running tasks in the background?
   → execution/ (task engine) or background/ (scheduled/proactive)

Is it about Redis, embedding cache, search cache?
   → cache/

Is it a shared utility (logging, batching)?
   → utils/
```

---

## File Arrangement — Complete Folder Skeleton

```
friday_mark2/
└── friday/                        ← ROOT of the new clean project
    │
    ├── ARCHITECTURE.md            ← ★ THE LIVING DOCUMENT (read before adding anything)
    ├── terminal_chat.py           ← Entry point (thin wrapper, calls interfaces/terminal)
    │
    ├── 🔀 router/                 ← TRAFFIC CONTROL ONLY. No business logic.
    │   ├── __init__.py
    │   ├── smart_router.py        ← Central coordinator (FROM: src/routing/smart_router.py)
    │   ├── llm_router.py          ← LLM tier classifier (FROM: src/routing/llm_router.py)
    │   ├── intent_classifier.py   ← Regex fallback (FROM: src/routing/intent_classifier.py)
    │   └── handlers/
    │       ├── __init__.py
    │       ├── simple_handler.py  ← 0 tools, LLM+context (FROM: src/routing/simple/)
    │       ├── medium_handler.py  ← 1-3 tools, AgentLoop (FROM: src/routing/medium/)
    │       └── complex_handler.py ← Multi-agent planner (FROM: src/routing/complex_multi_agent_planning/planner.py)
    │
    ├── 🧠 agents/                 ← ALL intelligent entities
    │   ├── __init__.py
    │   ├── base_agent.py          ← ★ NEW: Abstract base every agent inherits
    │   ├── registry.py            ← ★ NEW: Auto-discovers all agents in specialized/
    │   ├── friday/                ← The main Friday brain
    │   │   ├── __init__.py
    │   │   ├── agent.py           ← Main AgentLoop (FROM: src/agent/loop.py)
    │   │   ├── prompts.py         ← System prompt templates (FROM: src/agent/prompts.py)
    │   │   ├── session.py         ← Session R/W (FROM: src/agent/session.py)
    │   │   ├── session_repair.py  ← Transcript repair (FROM: src/agent/session_transcript_repair.py)
    │   │   ├── llm.py             ← Ollama client wrapper (FROM: src/agent/llm.py)
    │   │   └── privacy.py         ← PII filter (FROM: src/agent/privacy.py)
    │   └── specialized/           ← DROP NEW AGENTS HERE
    │       ├── researcher/        ← Research agent (FROM: subagent_registry.py patterns)
    │       │   ├── __init__.py
    │       │   ├── agent.py       ← EMPTY STUB
    │       │   └── prompts.py     ← EMPTY STUB
    │       ├── telegram/          ← Future Telegram agent
    │       │   ├── __init__.py
    │       │   └── agent.py       ← EMPTY STUB
    │       └── whatsapp/          ← Future WhatsApp agent
    │           ├── __init__.py
    │           └── agent.py       ← EMPTY STUB
    │
    ├── 🔧 tools/                  ← ALL capabilities. Dumb functions. No LLM logic inside.
    │   ├── __init__.py
    │   ├── base_tool.py           ← ★ NEW: Abstract base every tool inherits
    │   ├── registry.py            ← ★ NEW: Auto-discovers all tools. Builds schemas.
    │   ├── general/               ← Available to BOTH medium AND complex handlers
    │   │   ├── __init__.py
    │   │   ├── web_search.py      ← DuckDuckGo / Tavily / SerpAPI (EMPTY STUB)
    │   │   ├── calculator.py      ← Math eval (EMPTY STUB)
    │   │   └── datetime_tool.py   ← Current time / timezone (EMPTY STUB)
    │   ├── memory/                ← Memory R/W tools (FROM: src/agent/tools.py, memory section)
    │   │   ├── __init__.py
    │   │   ├── search_memory.py   ← Vector + BM25 hybrid search
    │   │   ├── write_memory.py    ← Store a new fact/event
    │   │   ├── recall_fact.py     ← Retrieve a specific fact
    │   │   └── update_preference.py ← Update user preference
    │   ├── calendar/              ← Calendar event tools
    │   │   ├── __init__.py
    │   │   ├── add_event.py
    │   │   ├── cancel_event.py
    │   │   └── list_events.py
    │   ├── browser/               ← Browser automation tools (FROM: src/tools/browser/)
    │   │   ├── __init__.py
    │   │   └── (browser tools go here)
    │   └── mcp/                   ← MCP protocol tools (FROM: src/tools/mcp/)
    │       └── __init__.py
    │
    ├── 🧩 memory/                 ← THE 6-LAYER MEMORY SYSTEM
    │   ├── __init__.py
    │   ├── pipeline.py            ← ★ NEW: Runs asyncio.create_task on EVERY input
    │   │                             Extracts facts → embeds → stores → promotes
    │   ├── db_manager.py          ← SQLite R/W (FROM: src/database/db_manager.py)
    │   └── layers/
    │       ├── __init__.py
    │       ├── layer_1_working.py     ← Current task context (what's happening NOW)
    │       ├── layer_2_short_term.py  ← Recent conversation (FROM: src/memory/short_term.py)
    │       ├── layer_3_episodic.py    ← Events / calendar facts (FROM: src/memory/facts.py)
    │       ├── layer_4_semantic.py    ← Stable user knowledge (FROM: personalization part)
    │       ├── layer_5_procedural.py  ← Learned habits / patterns
    │       ├── layer_6_profile.py     ← Core profile (FROM: src/memory/personalization.py)
    │       ├── promotion.py           ← (FROM: src/memory/promotion.py)
    │       ├── dreaming.py            ← (FROM: src/memory/dreaming.py)
    │       └── temporal_decay.py      ← (FROM: src/memory/temporal_decay.py)
    │
    ├── 🔍 search/                 ← HOW Friday retrieves from memory
    │   ├── __init__.py
    │   ├── hybrid_search.py       ← BM25 + vector (FROM: src/search/hybrid_search.py)
    │   ├── mmr.py                 ← Maximal Marginal Relevance (FROM: src/search/mmr.py)
    │   └── embedding_manager.py   ← Embed text → vectors (FROM: src/embeddings/)
    │
    ├── 🧪 reasoning/              ← HOW Friday thinks about problems
    │   ├── __init__.py
    │   ├── chain_of_thought.py    ← ★ NEW: CoT prompting wrapper
    │   ├── reflection.py          ← ★ NEW: Self-critique loop (did I answer correctly?)
    │   ├── explainability.py      ← ★ NEW: Logs EVERY routing decision with reasoning
    │   └── transparency.py        ← ★ NEW: Human-readable decision audit trail
    │
    ├── 👁️  awareness/             ← WHAT Friday knows is happening RIGHT NOW
    │   ├── __init__.py
    │   ├── context_tracker.py     ← ★ NEW: Live snapshot (execution state + memory updates)
    │   ├── execution_monitor.py   ← ★ NEW: Watches the execution engine in real time
    │   ├── world_model.py         ← ★ NEW: Real-time world info (time, weather, news)
    │   └── status_reporter.py     ← ★ NEW: Formats "what am I doing?" responses
    │
    ├── ⚙️  execution/             ← HOW Friday runs tasks (tools AND agents)
    │   ├── __init__.py
    │   ├── engine.py              ← Main engine (FROM: src/execution/execution_engine.py)
    │   │                             UPGRADED: can spawn agents, not just tool stubs
    │   ├── state_manager.py       ← Track everything running (FROM: src/execution/state_manager.py)
    │   │                             UPGRADED: agent-level tracking, not just step-level
    │   ├── parallel_executor.py   ← ★ NEW: asyncio.gather() wrapper for multi-agent parallel runs
    │   ├── series_executor.py     ← ★ NEW: Series agent calls (output of one → input of next)
    │   ├── subagent_registry.py   ← Spawn depth/breadth limits (FROM: src/routing/complex_multi_agent_planning/subagent_registry.py)
    │   ├── memory_aware_executor.py ← (FROM: src/execution/memory_aware_executor.py)
    │   └── learning.py            ← Learn from outcomes (FROM: src/execution/learning.py)
    │
    ├── 🕒 background/             ← WHAT Friday does autonomously
    │   ├── __init__.py
    │   ├── scheduler.py           ← (FROM: BACKGROUND_WORKER/scheduler.py)
    │   ├── proactive_events.py    ← (FROM: BACKGROUND_WORKER/proactive_events.py)
    │   ├── goal_tracker.py        ← ★ NEW: Autonomous goal store and checker
    │   ├── context_summarizer.py  ← (FROM: BACKGROUND_WORKER/context_summarizer.py)
    │   └── google_workspace.py    ← (FROM: BACKGROUND_WORKER/google_workspace.py)
    │
    ├── 📡 interfaces/             ← EVERYTHING that touches the outside world
    │   ├── __init__.py
    │   ├── base.py                ← ★ NEW: Abstract Interface (receive_input, send_output)
    │   ├── terminal/              ← ★ ACTIVE NOW (testing)
    │   │   ├── __init__.py
    │   │   └── chat.py            ← (FROM: terminal_chat.py)
    │   ├── messaging/             ← Future channels (EMPTY STUBS)
    │   │   ├── __init__.py
    │   │   ├── telegram/          ← EMPTY
    │   │   │   ├── __init__.py
    │   │   │   └── adapter.py     ← EMPTY STUB
    │   │   └── whatsapp/          ← EMPTY
    │   │       ├── __init__.py
    │   │       └── adapter.py     ← EMPTY STUB
    │   ├── livekit/               ← EMPTY (LiveKit STT/TTS — future integration)
    │   │   ├── __init__.py
    │   │   └── adapter.py         ← EMPTY STUB
    │   └── api/                   ← EMPTY (REST API — future)
    │       ├── __init__.py
    │       └── server.py          ← (FROM: src/api/server.py)
    │
    ├── 🗄️  cache/
    │   ├── __init__.py
    │   ├── redis_client.py        ← (FROM: src/cache/redis_client.py)
    │   ├── embedding_cache.py     ← (FROM: src/cache/embedding_cache_redis.py)
    │   └── search_cache.py        ← (FROM: src/cache/search_cache_redis.py)
    │
    ├── 🔒 security/
    │   ├── __init__.py
    │   └── filters.py             ← (FROM: src/security/filters.py)
    │
    ├── 🛠️  utils/
    │   ├── __init__.py
    │   ├── logger.py              ← (FROM: src/utils/logger.py)
    │   └── batching.py            ← (FROM: src/utils/batching.py)
    │
    ├── ⚙️  config/
    │   ├── __init__.py
    │   ├── settings.py            ← Central config object (model, db path, etc.)
    │   └── friday.yaml            ← User-editable config
    │
    └── 🧪 tests/
        ├── test_routing.py        ← (FROM: test_routing_system.py)
        ├── test_memory.py         ← (FROM: test_facts_system.py)
        └── test_stress.py         ← (FROM: stress_test_full.py)
```

---

## The ARCHITECTURE.md Living Document

> This will be placed at `friday_mark2/friday/ARCHITECTURE.md`
> I will read this file before making ANY changes in the future.

---

## Migration Phases

> [!IMPORTANT]
> **Zero logic changes during migration.** Every file keeps its existing code. Only location and imports change. Terminal chat works after each phase.

| Phase | Action | Risk |
|---|---|---|
| **0** | Create the empty folder skeleton (no file moves) | None |
| **1** | Move `agents/friday/` (loop.py, session, prompts, privacy, llm) | Low |
| **2** | Move `router/` and `router/handlers/` | Low |
| **3** | Move `tools/` (split tools.py into memory/, calendar/, general/) | Medium |
| **4** | Move `memory/` and `search/` | Low |
| **5** | Move `execution/` and `background/` | Low |
| **6** | Move `cache/`, `security/`, `utils/`, `config/` | None |
| **7** | Wire `interfaces/terminal/chat.py` as new entry point | Medium |
| **8** | Add NEW files: `memory/pipeline.py`, `agents/registry.py`, `tools/registry.py`, `execution/parallel_executor.py`, all `reasoning/`, all `awareness/` | New code only |
| **9** | Update all imports, run test suite, verify terminal chat | Verify |

**Execution starts with Phase 0 immediately upon your approval.**
