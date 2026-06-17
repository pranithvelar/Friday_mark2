import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from friday.config.settings import IntelligentMemoryConfig
from friday.memory.db_manager import MemoryDatabaseManager
from friday.search.indexer import TextIndexer
from friday.search.fts_search import FTSSearcher
from friday.tools.tools_legacy import ToolSystem, reindex_existing_files
from friday.agents.friday.agent import AgentLoop
from friday.router.smart_router import build_smart_router
from friday.agents.friday.session import SessionManager
from friday.memory.layers.layer_6_profile import UserPersonalization
from friday.memory.layers.dreaming import MemoryDreamer
from friday.memory.layers.promotion import promote_top_memories, prune_stale_entries, get_promotion_stats, PromotionEngine
from friday.memory.layers.layer_3_episodic import groom_facts, FactStore
from friday.memory.pipeline import MemoryPipeline
from friday.memory.context_assembler import ContextAssembler
from friday.memory.project_chronicle import ProjectRegistry, ProjectClassifier, ProjectDreamer
from friday.memory.project_chronicle.safety import (
    ChronicleCircuitBreaker, SafeChronicle, check_health
)
from friday.awareness.live_context import LiveContextState, LiveContextLoop
from friday.llm import build_provider
from friday.memory.layers.layer_4_semantic import SemanticMemory
from friday.memory.layers.layer_5_procedural import ProceduralMemory
from friday.background.knowledge_extractor import KnowledgeExtractor

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SESSION_ID = "terminal_default"


async def main():
    print("Initializing Intelligent Memory System...")
    config = IntelligentMemoryConfig.load()
    workspace = config.workspace_dir
    os.makedirs(workspace, exist_ok=True)

    # 0. LLM Provider (Slot 1: API, Slot 2: Ollama fallback)
    llm_provider = build_provider(config)
    print(f"[OK] LLM: {llm_provider.active_slot_name}")

    # 1. Database (single source of truth for ALL state)
    db_path = os.path.join(workspace, "memory.db")
    db_manager = MemoryDatabaseManager(db_path)
    db_manager.ensure_schema()
    print("[OK] Database initialized (all state in SQLite)")
    
    # 2. Text Indexer
    indexer = TextIndexer(db_manager)

    # 3. Searcher
    searcher = FTSSearcher(db_manager)

    # 4. Personalization (DB-backed)
    personalization = UserPersonalization(db_manager)
    facts = personalization.profile.get("facts", {})
    prefs = personalization.profile.get("preferences", {})
    if len(facts) + len(prefs) > 0:
        print(f"[OK] User profile loaded ({len(facts)} facts, {len(prefs)} preferences)")
    else:
        print("[OK] User profile initialized (empty)")

    # 5. Dreaming
    dreamer = MemoryDreamer(workspace, model=config.llama_model, llm_provider=llm_provider)
    print("[OK] Dreaming pipeline ready")

    # 6. Reindex
    reindexed = reindex_existing_files(workspace, db_manager)
    if reindexed > 0:
        print(f"[OK] Reindexed {reindexed} files")

    # 7. Startup pruning
    pruned = prune_stale_entries(db_manager)
    if pruned > 0:
        print(f"[OK] Pruned {pruned} stale recall entries")
        
    try:
        await groom_facts(db_manager, None, workspace_dir=workspace)
        print("[OK] Groomed expired temporal facts")
    except Exception as e:
        print(f"[WARN] Failed to groom facts: {e}")

    # 8. Tools
    tools = ToolSystem(
        workspace_dir=workspace,
        db_manager=db_manager,
        personalization=personalization,
        dreamer=dreamer
    )
    tools.setup_default_tools(searcher=searcher)
    print("[OK] Tools registered:", list(tools.tools.keys()))

    # 9. Session (DB-backed)
    session_mgr = SessionManager(db_manager)
    prior_count = len(session_mgr.load_session(SESSION_ID))
    if prior_count > 0:
        print(f"[OK] Restored {prior_count} messages from previous session")
    else:
        print("[OK] Starting fresh session")

    # 10. Agent Loop
    loop = AgentLoop(
        workspace_dir=workspace,
        model=config.llama_model,
        session_manager=session_mgr,
        session_id=SESSION_ID,
        personalization=personalization,
        db_manager=db_manager,
        llm_provider=llm_provider,
    )
    for name, schema in zip(tools.tools.keys(), tools.schemas):
        loop.register_tool(name, tools.tools[name], schema)
    loop._status_callback = lambda msg: print(f"  [{msg}]")
    loop._searcher = searcher  # enables proactive memory cross-reference on every message
    print(f"[OK] Agent loop ready ({llm_provider.active_slot_name}) — proactive memory enabled")

    # 11. Memory Pipeline — learns from EVERY input in background
    fact_store       = FactStore(db_manager)
    promotion_engine = PromotionEngine(db_manager)   # adapter: wraps promote_top_memories + prune_stale_entries
    memory_pipeline  = MemoryPipeline(
        db_manager         = db_manager,
        personalization    = personalization,
        indexer            = indexer,
        fact_store         = fact_store,
        promotion_engine   = promotion_engine,        # was None → now wired: promotes memories after every message
    )
    print("[OK] Memory pipeline ready — continuous parallel learning + promotion active")

    # 11b. Project Chronicle — JARVIS-style per-project documentation
    proj_registry   = ProjectRegistry(workspace)
    proj_classifier = ProjectClassifier(None)
    proj_dreamer    = ProjectDreamer(proj_registry, llm_provider=llm_provider)

    # ── Safety Layer 1: Health check + repair at startup ────────────────────
    import os as _os
    health = check_health(_os.path.join(workspace, "memory", "projects"))
    if health.issues_found:
        print(f"[Chronicle] Health check: {health.issues_fixed or health.unfixable}")
    if health.unfixable:
        print(f"[Chronicle] WARNING: unfixable issues: {health.unfixable}")

    # ── Safety Layer 2: Circuit breaker ────────────────────────────────
    chronicle_breaker = ChronicleCircuitBreaker(trip_threshold=5, reset_seconds=600)

    # Pre-embed all existing projects at startup so classifier is warm
    await proj_classifier.load_all_projects(proj_registry)

    # Mark projects dormant if idle > 7 days
    dormant_slugs = proj_registry.check_and_mark_dormant()
    if dormant_slugs:
        print(f"[OK] Marked {len(dormant_slugs)} project(s) dormant: {dormant_slugs}")

    # Bundle for pipeline + router injection
    chronicle = {
        "registry":   proj_registry,
        "classifier": proj_classifier,
        "dreamer":    proj_dreamer,
    }
    # ── Safety Layer 3: SafeChronicle shell ─────────────────────────────
    # Wraps every operation: atomic writes, per-project lock, circuit breaker.
    # Injected into chronicle dict as 'safe' for router + pipeline.
    safe_chronicle = SafeChronicle(chronicle, chronicle_breaker)
    chronicle["safe"]    = safe_chronicle
    chronicle["breaker"] = chronicle_breaker

    # Inject chronicle into the memory pipeline so it can create project folders
    memory_pipeline.chronicle = chronicle
    n_projects = len(proj_registry.list_projects())
    print(f"[OK] Project Chronicle ready — {n_projects} project(s) tracked | safety system active")

    # 12. Layer 4 — Semantic Memory (inferred stable facts)
    semantic_memory = SemanticMemory(db_manager)
    sf_count = len(semantic_memory.get_all_active())
    print(f"[OK] Layer 4 (Semantic Memory) ready — {sf_count} fact(s)")

    # 13. Layer 5 — Procedural Memory (behavioral patterns via LearningEngine)
    from friday.execution.learning import LearningEngine
    learning_engine = LearningEngine(db_manager)
    procedural_memory = ProceduralMemory(learning_engine)
    print("[OK] Layer 5 (Procedural Memory) ready")

    # 14. Knowledge Extractor (background L4+L5 updater)
    knowledge_extractor = KnowledgeExtractor(
        llm_provider      = llm_provider,
        semantic_memory   = semantic_memory,
        procedural_memory = procedural_memory,
        profile           = personalization,   # L6 cross-check: won't re-add what Reflection already stored
    )
    loop._knowledge_extractor = knowledge_extractor
    print("[OK] KnowledgeExtractor ready — deep learning active (every 20 messages)")

    # 14b. Reflection Agent — Self-Critique: learns from Friday's own failures → Layer 5
    # Triggered by: (a) tool failure after retry, (b) user correction keywords in message.
    # Writes lessons to ProceduralMemory so ContextAssembler injects them next time.
    from friday.reasoning.reflection import ReflectionAgent
    reflection_agent = ReflectionAgent(
        llm_provider      = llm_provider,
        procedural_memory = procedural_memory,
    )
    loop._reflection_agent = reflection_agent
    print("[OK] Reflection Agent ready — self-critique on failures + user corrections")


    # 15. Context Assembler — builds full context bundle for ALL route tiers
    context_assembler = ContextAssembler(
        searcher          = searcher,
        fact_store        = fact_store,
        personalization   = personalization,
        semantic_memory   = semantic_memory,
        procedural_memory = procedural_memory,
    )
    print("[OK] Context assembler ready — L2/L3/L4/L5/L6 all wired")

    # 16. Register BRAIN tools for L4 + L5 + personalization
    #     The LLM sees these in SYSTEM_PROMPT KNOWLEDGE PROTOCOL and calls them silently.

    def _remember_fact(subject: str, predicate: str, object: str, confidence: float = 0.7) -> str:
        fact_id = semantic_memory.add_fact(subject, predicate, object, confidence=confidence, source="stated")
        return f"Stored: {subject} {predicate} {object}"

    loop.register_tool("remember_fact", _remember_fact, {
        "name": "remember_fact",
        "description": "Store a stable fact about the user into Layer 4 (semantic memory)",
        "parameters": {
            "type": "object",
            "properties": {
                "subject":    {"type": "string", "description": "Who — usually the user's name or 'the user'"},
                "predicate":  {"type": "string", "description": "Relationship — e.g. 'is studying', 'works at'"},
                "object":     {"type": "string", "description": "The value — e.g. 'Computer Science'"},
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0 (default 0.7)"},
            },
            "required": ["subject", "predicate", "object"]
        }
    })

    def _update_fact(fact_id: str, new_value: str, confidence: float = 0.85) -> str:
        ok = semantic_memory.update_fact(fact_id, new_object=new_value, new_confidence=confidence)
        return "Updated" if ok else f"Fact {fact_id[:8]} not found"

    loop.register_tool("update_fact", _update_fact, {
        "name": "update_fact",
        "description": "Correct an existing Layer 4 semantic fact. Use when user corrects something FRIDAY believes.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact_id":    {"type": "string", "description": "Full UUID of the fact to update"},
                "new_value":  {"type": "string", "description": "The corrected object value"},
                "confidence": {"type": "number"},
            },
            "required": ["fact_id", "new_value"]
        }
    })

    def _forget_fact(fact_id: str) -> str:
        ok = semantic_memory.delete_fact(fact_id)
        return "Deleted" if ok else f"Fact {fact_id[:8]} not found"

    loop.register_tool("forget_fact", _forget_fact, {
        "name": "forget_fact",
        "description": "Soft-delete a Layer 4 semantic fact. Use when user says something is no longer true.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact_id": {"type": "string", "description": "Full UUID of the fact to delete"},
            },
            "required": ["fact_id"]
        }
    })

    async def _remember_pattern(trigger: str, behavior: str, context: str = "general") -> str:
        await procedural_memory.add_pattern(trigger, behavior, context=context)
        return f"Pattern saved: '{trigger}' → '{behavior}'"

    loop.register_tool("remember_pattern", _remember_pattern, {
        "name": "remember_pattern",
        "description": "Save a behavioral pattern about how the user prefers to work (Layer 5)",
        "parameters": {
            "type": "object",
            "properties": {
                "trigger":  {"type": "string", "description": "When this happens — e.g. 'asks for a summary'"},
                "behavior": {"type": "string", "description": "Do this — e.g. 'bullet points, max 5'"},
                "context":  {"type": "string", "description": "Optional context tag (default: general)"},
            },
            "required": ["trigger", "behavior"]
        }
    })

    async def _update_pattern(trigger: str, old_behavior: str, new_behavior: str, context: str = "general") -> str:
        await procedural_memory.correct_pattern(trigger, old_behavior, new_behavior, context=context)
        return f"Pattern corrected: '{trigger}'"

    loop.register_tool("update_pattern", _update_pattern, {
        "name": "update_pattern",
        "description": "Correct a behavioral pattern when the user explicitly changes a preference (Layer 5)",
        "parameters": {
            "type": "object",
            "properties": {
                "trigger":      {"type": "string"},
                "old_behavior": {"type": "string"},
                "new_behavior": {"type": "string"},
                "context":      {"type": "string"},
            },
            "required": ["trigger", "old_behavior", "new_behavior"]
        }
    })

    async def _delete_pattern(trigger: str, context: str = "general") -> str:
        ok = await procedural_memory.delete_pattern(trigger, context=context)
        return "Deleted" if ok else f"Pattern '{trigger}' not found"

    loop.register_tool("delete_pattern", _delete_pattern, {
        "name": "delete_pattern",
        "description": "Remove a behavioral pattern completely (Layer 5)",
        "parameters": {
            "type": "object",
            "properties": {
                "trigger": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["trigger"]
        }
    })

    def _store_personalization(key: str, value: str) -> str:
        """General personalization tool — LLM decides what to store."""
        if any(k in key.lower() for k in ["name", "call", "address"]):
            personalization.update_preference("address_as", value)
        elif any(k in key.lower() for k in ["style", "length", "format"]):
            personalization.update_preference("response_style", value)
        elif any(k in key.lower() for k in ["tone"]):
            personalization.update_preference("tone", value)
        else:
            personalization.update_fact(key, value)
        return f"Personalization saved: {key}={value}"

    loop.register_tool("store_personalization", _store_personalization, {
        "name": "store_personalization",
        "description": "Store any user preference or personal detail into Layer 6 (profile)",
        "parameters": {
            "type": "object",
            "properties": {
                "key":   {"type": "string", "description": "Category — e.g. 'name', 'response_style', 'tone'"},
                "value": {"type": "string", "description": "The value to store"},
            },
            "required": ["key", "value"]
        }
    })

    def _list_facts(query: str = "") -> str:
        results = semantic_memory.search_facts(query) if query else semantic_memory.get_all_active(limit=20)
        if not results:
            return "No semantic facts stored yet."
        lines = [f"[{f['id'][:8]}] {f['subject']} {f['predicate']} {f['object']} (conf={f['confidence']:.2f})" for f in results]
        return "\n".join(lines)

    loop.register_tool("list_facts", _list_facts, {
        "name": "list_facts",
        "description": "Show all stored semantic facts about the user (Layer 4). Use when user asks what you know.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional keyword to filter facts"},
            },
        }
    })

    async def _list_patterns(query: str = "") -> str:
        patterns = await procedural_memory.get_all_patterns(min_confidence=0.3)
        if not patterns:
            return "No behavioral patterns stored yet."
        lines = [f"[{p.pattern_id[:8]}] trigger='{p.key}' → '{p.value}' (conf={p.confidence:.2f})" for p in patterns]
        return "\n".join(lines)

    loop.register_tool("list_patterns", _list_patterns, {
        "name": "list_patterns",
        "description": "Show all behavioral patterns about the user (Layer 5). Use when user asks how you adapt.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional keyword to filter patterns"},
            },
        }
    })

    print(f"[OK] 9 BRAIN tools registered (L4 facts + L5 patterns + personalization)")

    # 12b. ToolRegistry — auto-discover tools from tools/ subdirectories
    # Any file that subclasses BaseTool is picked up automatically.
    # Tools are registered into the AgentLoop so ALL tiers (BRAIN, Medium, Complex) see them.
    # Safe: wrapped in try/except — discovery failure never blocks startup.
    # Skip: if a tool name is already registered (no override of existing working tools).
    from friday.tools.registry import ToolRegistry
    _tool_registry = ToolRegistry()
    try:
        _discovered = _tool_registry.discover()
        _newly_registered = 0
        for _tname in _tool_registry.list_all():
            if _tname in loop.tools:
                # Already registered by ToolSystem or BRAIN tools — don't override.
                logger.debug(f"[ToolRegistry] '{_tname}' already registered — skipping")
                continue
            _tinst = _tool_registry.get(_tname)
            loop.register_tool(_tname, _tinst.run, _tinst.to_schema())
            _newly_registered += 1
        if _discovered > 0:
            print(f"[OK] ToolRegistry: {_discovered} tool(s) discovered, "
                  f"{_newly_registered} newly registered")
        else:
            print("[OK] ToolRegistry: ready — add tools to friday/tools/ to auto-register")
    except Exception as _reg_err:
        logger.warning(f"[ToolRegistry] Discovery failed (non-fatal): {_reg_err}")
        _tool_registry = None

    # Store registry on loop so ExecutionEngine can dispatch by tool_category.
    loop._tool_registry = _tool_registry

    # 13. Smart Router (3-Tier Routing: Simple → Medium → Complex Multi-Agent)
    router = build_smart_router(
        agent_loop        = loop,
        db_manager        = db_manager,
        personalization   = personalization,
        searcher          = searcher,
        llm_provider      = llm_provider,
        memory_pipeline   = memory_pipeline,
        context_assembler = context_assembler,
    )
    # Give SimpleHandler access to the AgentLoop for LLM-backed responses
    router.simple.set_agent_loop(loop)
    # Inject project chronicle into router for per-message classification + logging
    router.set_chronicle(chronicle)
    # Inject EventEngine — instant hot-path event detection + conflict check
    # Zero LLM, zero API, pure regex + SQLite (~10ms). Runs before every LLM call.
    from friday.memory.event_engine import EventEngine
    event_engine = EventEngine(fact_store)
    router.set_event_engine(event_engine)
    print("[OK] Smart Router ready (Simple / Medium / Complex — all context-aware)")
    print("[OK] EventEngine ready (hot-path event detection, zero LLM, ~10ms)")


    # Schedule ProjectDreamer to run every 30 minutes in background
    async def _project_dream_loop():
        while True:
            await asyncio.sleep(30 * 60)  # 30 minutes
            try:
                synthesized = await proj_dreamer.run_cycle()
                if synthesized:
                    logger.info(f"[ProjectDreamer] Cycle complete: {synthesized}")
            except Exception as e:
                logger.warning(f"[ProjectDreamer] Cycle error: {e}")
    asyncio.create_task(_project_dream_loop())

    # ── LIVE CONTEXT LOOP ───────────────────────────────────────────────
    # Background loop that continuously refreshes the MAIN BRAIN's awareness.
    # Runs every 8 seconds, independently of user input.
    # The brain reads this on EVERY LLM call via AgentLoop._build_system_prompt().
    from friday.execution.state_manager import ExecutionStateManager
    live_ctx = LiveContextState()
    # Get the state_manager from the router's planner
    _state_mgr = getattr(getattr(router, 'planner', None), 'execution_engine', None)
    _state_mgr = getattr(_state_mgr, 'state_manager', None)
    live_loop = LiveContextLoop(
        live_ctx      = live_ctx,
        fact_store    = fact_store,
        state_manager = _state_mgr,
        interval      = 8,
    )
    # Inject live_ctx into AgentLoop so _build_system_prompt reads it always
    loop._live_ctx = live_ctx
    # Inject live_ctx into router so plan approval gate can write pending_plan_block
    router.set_live_ctx(live_ctx)
    # Start background refresh
    asyncio.create_task(live_loop.run())
    print("[OK] Live context loop started — brain always aware (8s refresh)")


    # Start Background Worker
    from friday.background.scheduler import BackgroundScheduler
    from friday.background.proactive_events import ProactiveEventsWatcher
    from friday.background.google_workspace import GoogleWorkspaceWatcher
    from friday.background.memory_decay import MemoryDecayWatcher          # Layer 4 confidence decay
    bg_scheduler = BackgroundScheduler(db_manager, config)
    bg_scheduler.register_watcher(ProactiveEventsWatcher(workspace))
    bg_scheduler.register_watcher(GoogleWorkspaceWatcher())
    bg_scheduler.register_watcher(MemoryDecayWatcher())                   # silent: runs every 6h, 45-day half-life
    bg_scheduler._llm_provider = llm_provider  # passed to watchers at runtime
    asyncio.create_task(bg_scheduler.run())

    # Memory stats
    try:
        conn = db_manager.get_connection()
        count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        print(f"[OK] {count} memory chunks in database")
    except Exception:
        pass

    # Promotion stats
    pstats = get_promotion_stats(db_manager)
    if pstats["total_tracked"] > 0:
        print(f"[OK] Promotion: {pstats['total_tracked']} tracked, {pstats['promoted']} promoted")

    print("\n" + "=" * 55)
    print("  Friday — Intelligent Memory System")
    print("  All state stored in SQLite (no flat files)")
    print("-" * 55)
    print("  Commands:")
    print("    exit/quit  — Stop the chat")
    print("    clear      — Reset session history")
    print("    dream      — Trigger dreaming + promotion")
    print("    profile    — Show your user profile")
    print("    status     — Show system stats")
    print("    promote    — Run promotion scoring now")
    print("=" * 55 + "\n")

    while True:
        try:
            print("You: ", end="", flush=True)
            event_loop = asyncio.get_event_loop()
            user_input = (await event_loop.run_in_executor(None, sys.stdin.readline)).strip()
            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                break

            if user_input.lower() == 'clear':
                session_mgr.clear_session(SESSION_ID)
                loop._history = []
                loop._summary_cache = ""
                loop._compacted_up_to = 0
                loop._history_loaded = True
                # Clear summary from DB meta table
                try:
                    conn = db_manager.get_connection()
                    conn.execute("DELETE FROM meta WHERE key = ?", (f"summary:{SESSION_ID}",))
                    conn.commit()
                except Exception:
                    pass
                print("[OK] Session cleared.\n")
                continue

            if user_input.lower() == 'dream':
                print("Running dreaming + promotion...")
                result = await tools.tools["dream_now"]()
                print(f"\n{result}\n")
                continue

            if user_input.lower() == 'promote':
                print("Running promotion scoring...")
                promoted = promote_top_memories(db_manager)
                if promoted:
                    print(f"\nPromoted {len(promoted)} memories to PROMOTED.md:")
                    for p in promoted:
                        print(f"  -> {p['snippet'][:80]}... (score: {p['score']:.3f})")
                else:
                    print("\nNo memories meet promotion threshold yet.")
                    ps = get_promotion_stats(db_manager)
                    print(f"  Tracked: {ps['total_tracked']}, Max score: {ps['max_score']:.3f}")
                print()
                continue

            if user_input.lower() == 'profile':
                ctx = personalization.get_context_string()
                print(f"\n{ctx}\n" if ctx else "\n[No profile data yet.]\n")
                continue

            if user_input.lower() == 'status':
                try:
                    conn = db_manager.get_connection()
                    chunks = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
                    sessions = len(session_mgr.load_session(SESSION_ID))
                    fact_count = conn.execute("SELECT COUNT(*) as c FROM facts WHERE status='active'").fetchone()["c"]
                    ps = get_promotion_stats(db_manager)
                    print(f"\n  Database chunks: {chunks}")
                    print(f"  Active facts: {fact_count}")
                    print(f"  Session messages: {sessions}")
                    print(f"  Search: FTS5 (Pure Keyword)")
                    print(f"  Context window: last 10 + summary")
                    print(f"  --- Promotion ---")
                    print(f"  Tracked: {ps['total_tracked']}, Promoted: {ps['promoted']}, Pending: {ps['pending']}")
                    print(f"  Scores: {ps['min_score']:.3f} - {ps['max_score']:.3f} (avg {ps['avg_score']:.3f})\n")
                except Exception as e:
                    print(f"  Status error: {e}\n")
                continue

            print("Processing...")
            try:
                # route_and_learn() also indexes FRIDAY's reply into vector DB
                # so "what did you say about X" is searchable in future
                result = await router.route_and_learn(user_input, SESSION_ID)
                print(f"\nFriday: {result['text']}\n")
                # Uncomment for debug tier info:
                # print(f"  [tier={result['complexity']} | {result['latency_ms']:.0f}ms]\n")
            except asyncio.TimeoutError:
                print("\n[ERROR] Response timed out. Try a simpler question.\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            traceback.print_exc()

    db_manager.close()
    print("Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
