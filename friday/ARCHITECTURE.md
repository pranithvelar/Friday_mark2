# Friday — Architecture Living Document
> **READ THIS BEFORE ADDING OR CHANGING ANYTHING.**
> This document is the single source of truth for where every file belongs.
> Last updated: 2026-06-13

---

## TL;DR — The 3 Laws

1. **Layers import downward only.** `interfaces → router → agents → tools/memory/reasoning/awareness/execution`. Nothing imports upward.
2. **Every input fires two parallel pipelines.** Response pipeline (user waits) AND Memory pipeline (background `asyncio.create_task`). Always. No exceptions.
3. **Adding a new agent or tool = one file in the right folder.** Zero changes elsewhere.

---

## Decision Tree — Where Does X Go?

```
Is it an input/output channel? (terminal, telegram, whatsapp, livekit, API)
   └─→ interfaces/

Is it an intelligent entity that reasons, makes decisions, and calls tools?
   └─→ agents/specialized/<name>/agent.py

Is it a capability — a dumb function with inputs and outputs?
   └─→ tools/<category>/<name>.py

Is it about classifying or routing user intent?
   └─→ router/

Is it about storing, retrieving, or forgetting information?
   └─→ memory/

Is it about HOW Friday reasons (CoT, reflection, explainability)?
   └─→ reasoning/

Is it about WHAT Friday knows is happening right now?
   └─→ awareness/

Is it about running tasks asynchronously or coordinating agents?
   └─→ execution/

Is it a scheduled/proactive autonomous task?
   └─→ background/

Is it a Redis or embedding cache?
   └─→ cache/

Is it a shared utility (logging, batching)?
   └─→ utils/
```

---

## Module Reference

### `router/` — Traffic Control
**Purpose:** Classify intent, pick a tier, hand off. Nothing else.
**Files:**
- `smart_router.py` — Main coordinator. Receives every user message. Calls classifier, picks handler, fires memory pipeline in parallel.
- `llm_router.py` — LLM-based tier classifier (simple/medium/complex based on tool count).
- `intent_classifier.py` — Regex fallback classifier used when LLM times out.
- `handlers/simple_handler.py` — Tier 1: 0 tools. Calls LLM with profile+calendar context injected.
- `handlers/medium_handler.py` — Tier 2: 1-3 tools. Runs AgentLoop with tool access.
- `handlers/complex_handler.py` — Tier 3: Multi-agent plan. Calls agents in parallel or series via execution engine.

**Rules:**
- Router never imports from `agents/` directly — it calls agents via `agents/registry.py`.
- Router never touches memory directly — it calls `memory/pipeline.py` as a background task.
- Router never does business logic — only dispatch.

---

### `agents/` — All Intelligent Entities
**Purpose:** Every entity that can reason, plan, and call tools.

**`agents/base_agent.py`** — Abstract base class. Every agent must implement:
```python
agent_id: str          # Unique identifier (e.g. "researcher", "whatsapp")
description: str       # What this agent does (used by registry + complex handler)
capabilities: list     # List of capability keywords (e.g. ["web_search", "write_report"])
tool_scope: list       # Which tools this agent can use
async def run(task: str, context: dict) -> dict
```

**`agents/registry.py`** — Auto-discovers all agents in `agents/specialized/` on startup.
The complex handler queries: `registry.find_agents_for(task)` → returns list of matching agents.

**`agents/friday/agent.py`** — The main Friday brain (AgentLoop). Uses tools from `tools/` registry.

**`agents/specialized/`** — Drop new agents here. Each has its own folder with:
- `agent.py` — Inherits from `BaseAgent`
- `prompts.py` — Agent-specific system prompts
- `tools.py` (optional) — Agent-specific tools not available globally

**Adding a new agent:**
1. Create `agents/specialized/<name>/agent.py`
2. Implement `BaseAgent` interface
3. Done. Registry auto-discovers it on next startup.

---

### `tools/` — All Capabilities
**Purpose:** Dumb functions that DO things. No LLM logic inside tools.

**`tools/base_tool.py`** — Abstract base class. Every tool must implement:
```python
name: str              # Tool name used in tool call JSON
description: str       # What it does (used for schema generation)
scope: str             # "general" (medium+complex) | "agent:<id>" (one agent only)
parameters: dict       # JSON schema of parameters
async def run(**kwargs) -> dict
```

**`tools/registry.py`** — Auto-discovers all tools. Provides:
- `get_schema_for_agent(agent_id)` — Returns tool schemas for a specific agent
- `get_schema_general()` — Returns schemas for medium handler
- `execute(tool_name, **kwargs)` — Dispatches to the correct tool

**Tool categories:**
- `tools/general/` — web_search, calculator, datetime. Available to medium AND complex.
- `tools/memory/` — search_memory, write_memory, recall_fact, update_preference.
- `tools/calendar/` — add_event, cancel_event, list_events.
- `tools/browser/` — open_url, scrape_page, screenshot.
- `tools/mcp/` — MCP protocol tools.

**Adding a new tool:**
1. Create `tools/<category>/<name>.py`
2. Implement `BaseTool` interface, set `scope`
3. Done. Registry auto-discovers it.

---

### `memory/` — The 6-Layer Memory System
**Purpose:** Everything Friday knows, organized by how stable and important it is.

**`memory/pipeline.py`** — ★ CRITICAL. Called as `asyncio.create_task()` on EVERY user input.
```
Input arrives
└─→ pipeline.run(text, session_id)
       ├── Extract entities and facts from text
       ├── Embed the text (embedding_manager)
       ├── Write to Layer 3 (episodic) if event-like
       ├── Write to Layer 4 (semantic) if fact-like
       ├── Trigger promotion check (is this stable enough for Layer 6?)
       └── Update FeedbackDetector patterns
```
**This is what makes Friday truly learn from every conversation.**

**The 6 Layers:**
| Layer | File | What it stores | Stability |
|---|---|---|---|
| 1 | `layer_1_working.py` | Current task context | Session-only |
| 2 | `layer_2_short_term.py` | Recent conversation | Hours-days |
| 3 | `layer_3_episodic.py` | Events, calendar facts | Days-weeks |
| 4 | `layer_4_semantic.py` | General user knowledge | Weeks-months |
| 5 | `layer_5_procedural.py` | Habits, patterns | Months |
| 6 | `layer_6_profile.py` | Core profile (name, age, goals) | Permanent |

**`memory/promotion.py`** — Scores memories. Promotes from lower layers to higher based on recurrence, importance, and user feedback.

---

### `reasoning/` — How Friday Thinks
**Purpose:** Reasoning strategies and decision explainability.

- `chain_of_thought.py` — Wraps LLM calls with "think step by step" prompting for complex problems.
- `reflection.py` — After a response, asks "was that correct?" — self-critique loop.
- `explainability.py` — Logs every routing decision with its reasoning: why simple/medium/complex, which agent, which tools.
- `transparency.py` — Formats explainability logs into human-readable audit trails for the user.

---

### `awareness/` — What Friday Knows is Happening NOW
**Purpose:** Continuous real-time context injected into every LLM call.

- `context_tracker.py` — Maintains a live snapshot: current execution status, recent memory updates, user's current focus, last tool results.
- `execution_monitor.py` — Watches the execution engine. Updates context_tracker when steps complete.
- `world_model.py` — Real-time world info: current time, upcoming events, weather (via tools).
- `status_reporter.py` — Generates "what am I doing right now?" natural language summaries.

**The context block injected into every LLM call looks like:**
```
[FRIDAY AWARENESS]
Current time: Friday, June 13, 2026, 12:31 PM IST
Active execution: "Research AI trends" — Step 3/7 (43%)
Last learned: user is building Friday Mark 2 (stored in Layer 4)
User focus: architecture planning
```

---

### `execution/` — Running Tasks and Coordinating Agents
**Purpose:** Async task runner with retry, replan, and multi-agent support.

- `engine.py` — Main execution engine. Runs plans in background (`asyncio.create_task`). Knows about both TOOLS and AGENTS.
- `state_manager.py` — Tracks every running execution. Source of truth for `awareness/execution_monitor.py`.
- `parallel_executor.py` — `asyncio.gather(agent_a.run(), agent_b.run())` wrapper with error handling.
- `series_executor.py` — Sequential agent calls where output of agent_a feeds agent_b.
- `subagent_registry.py` — Depth and breadth limits (max 3 nesting levels, max 5 parallel children).
- `memory_aware_executor.py` — Pre-step memory check (does Friday already know this?).
- `learning.py` — Records success/failure patterns. Updates memory after execution.

---

### `interfaces/` — Everything That Touches the Outside World
**Purpose:** Input/output adapters. The core brain never knows which interface is active.

**`interfaces/base.py`** — Abstract base:
```python
async def receive_input() -> str   # Get text from the channel
async def send_output(text: str)   # Send text back to the channel
```

**Active now:**
- `interfaces/terminal/chat.py` — CLI chat loop.

**Empty stubs (future):**
- `interfaces/messaging/telegram/adapter.py` — Telegram bot adapter
- `interfaces/messaging/whatsapp/adapter.py` — WhatsApp webhook adapter
- `interfaces/livekit/adapter.py` — LiveKit voice I/O (STT+TTS handled by LiveKit SDK)
- `interfaces/api/server.py` — FastAPI REST endpoint

**Adding a new interface:**
1. Create `interfaces/<name>/adapter.py`
2. Implement `BaseInterface`
3. Import and start it in `terminal_chat.py` (or alongside it)
4. Zero changes to router, agents, memory, or any other module.

---

### `background/` — Autonomous Background Tasks
**Purpose:** Things Friday does on its own without being asked.

- `scheduler.py` — Cron-style scheduler. Triggers background jobs on a schedule.
- `proactive_events.py` — Monitors upcoming events and sends proactive reminders.
- `goal_tracker.py` — ★ NEW: Friday's autonomous goal store. Friday can set goals like "remind user about Goa trip at 9am". Scheduler checks and executes them.
- `context_summarizer.py` — Periodically summarizes long conversation history into compact memory.
- `google_workspace.py` — Google Calendar/Gmail integration (future).

---

## The 9 Capabilities — Where Each One Lives

| Capability | Primary Module | Supporting Modules |
|---|---|---|
| Real Tool Execution | `tools/` + `tools/registry.py` | `execution/engine.py` |
| Explainability & Transparency | `reasoning/explainability.py` | `reasoning/transparency.py` |
| Real Multi-Agent Coordination | `agents/` + `execution/parallel_executor.py` | `agents/registry.py`, `execution/subagent_registry.py` |
| Advanced Reasoning | `reasoning/chain_of_thought.py` | `reasoning/reflection.py` |
| Real-Time World Awareness | `tools/general/web_search.py` | `awareness/world_model.py` |
| Situational Awareness | `awareness/context_tracker.py` | `awareness/execution_monitor.py`, `awareness/status_reporter.py` |
| Agentic Self-Loop / Goals | `background/goal_tracker.py` | `background/scheduler.py`, `memory/layers/layer_1_working.py` |
| True Self-Learning | `memory/pipeline.py` | All 6 memory layers, `memory/promotion.py` |
| Voice / Multi-Modal | `interfaces/livekit/adapter.py` | LiveKit SDK (external) |

---

## Conventions

- All files are lowercase with underscores (`snake_case`).
- Every folder has an `__init__.py`.
- Every module has a module-level docstring explaining its purpose and what it imports from.
- No circular imports. Ever. Check with `python -c "import friday"` after every change.
- Stub files contain only: `# TODO: implement` and the class signature.
