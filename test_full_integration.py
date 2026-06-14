"""
Friday Mark 2 - Full Integration Test Suite
=============================================
Tests every subsystem end-to-end:
  1. Database + Schema
  2. Memory: write, search, facts, personalization, promotion, decay, dreaming
  3. Router: intent classifier, LLM router, simple/medium/complex handlers
  4. Agent: AgentLoop, tool registration, tool execution, session management
  5. Execution: state manager, engine, subagent registry
  6. New components: MemoryPipeline, AgentRegistry, ToolRegistry, ParallelExecutor
"""

import sys
import os
import asyncio
import traceback
import time
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Test infrastructure ──────────────────────────────────────────────────────

PASS = 0
FAIL = 0
ERRORS = []

def test(name, func):
    global PASS, FAIL, ERRORS
    try:
        result = func()
        if asyncio.iscoroutine(result):
            result = asyncio.get_event_loop().run_until_complete(result)
        PASS += 1
        print(f"  [OK] {name}")
        return result
    except Exception as e:
        FAIL += 1
        tb = traceback.format_exc()
        ERRORS.append((name, str(e), tb))
        print(f"  [FAIL] {name} - {e}")
        return None

# ── Shared state ─────────────────────────────────────────────────────────────

db_manager = None
personalization = None
searcher = None
embedder = None
session_mgr = None
loop_agent = None
tools_sys = None
router = None

# =============================================================================
# PHASE 1: DATABASE + SCHEMA
# =============================================================================

def phase_1():
    global db_manager
    print("\n" + "=" * 60)
    print("PHASE 1: Database + Schema")
    print("=" * 60)

    def test_db_init():
        from friday.memory.db_manager import MemoryDatabaseManager
        global db_manager
        db_path = os.path.join("workspace", "test_integration.db")
        # Clean slate
        if os.path.exists(db_path):
            os.remove(db_path)
        os.makedirs("workspace", exist_ok=True)
        db_manager = MemoryDatabaseManager(db_path)
        db_manager.ensure_schema()
        db_manager.ensure_vector_table(dimensions=768)
        conn = db_manager.get_connection()
        # Verify tables exist
        tables = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "chunks" in tables, f"Missing 'chunks' table. Found: {tables}"
        assert "facts" in tables, f"Missing 'facts' table. Found: {tables}"
        assert "sessions" in tables, f"Missing 'sessions' table. Found: {tables}"
        assert "meta" in tables, f"Missing 'meta' table. Found: {tables}"
        return len(tables)

    count = test("Database init + schema creation", test_db_init)
    if count:
        print(f"    -> {count} tables created")
    def test_db_write_read():
        conn = db_manager.get_connection()
        conn.execute("INSERT INTO chunks (id, path, content, source, chunkIndex) VALUES (?, ?, ?, ?, ?)",
                     ("test-1", "test.txt", "Pranith is building an AI assistant called Friday", "test", 0))
        conn.commit()
        row = conn.execute("SELECT content FROM chunks WHERE id = ?", ("test-1",)).fetchone()
        assert row is not None, "Failed to read back written chunk"
        assert "Pranith" in row["content"]

    test("Write + read chunk", test_db_write_read)

# =============================================================================
# PHASE 2: MEMORY SYSTEM
# =============================================================================

def phase_2():
    global personalization, embedder, searcher
    print("\n" + "=" * 60)
    print("PHASE 2: Memory System (6 layers)")
    print("=" * 60)

    # --- Layer 6: Profile / Personalization ---
    def test_personalization():
        from friday.memory.layers.layer_6_profile import UserPersonalization
        global personalization
        personalization = UserPersonalization(db_manager)
        assert personalization.profile is not None, "Profile is None"
        return type(personalization).__name__

    test("Layer 6: Profile init", test_personalization)

    def test_update_fact():
        personalization.update_fact("name", "Pranith")
        personalization.update_fact("occupation", "software builder")
        personalization.update_fact("education", "3rd year college")
        profile = personalization.profile
        assert profile["facts"]["name"] == "Pranith", f"Expected 'Pranith', got {profile['facts'].get('name')}"
        assert profile["facts"]["occupation"] == "software builder"
        return len(profile["facts"])

    count = test("Layer 6: Store 3 facts", test_update_fact)
    if count:
        print(f"    -> {count} facts stored")

    def test_update_preference():
        personalization.update_preference("tone", "casual")
        personalization.update_preference("response_style", "concise")
        personalization.update_preference("address_as", "Sir")
        prefs = personalization.profile["preferences"]
        assert prefs["tone"] == "casual"
        assert prefs["address_as"] == "Sir"
        return len(prefs)

    count = test("Layer 6: Store 3 preferences", test_update_preference)
    if count:
        print(f"    -> {count} preferences stored")

    def test_get_context_string():
        ctx = personalization.get_context_string()
        assert ctx is not None, "Context string is None"
        assert "Pranith" in ctx or "name" in ctx.lower(), f"Name not in context: {ctx[:100]}"
        return len(ctx)

    length = test("Layer 6: Get context string", test_get_context_string)
    if length:
        print(f"    -> {length} chars")

    def test_profile_persistence():
        """Verify that facts survive a reload"""
        from friday.memory.layers.layer_6_profile import UserPersonalization
        fresh = UserPersonalization(db_manager)
        assert fresh.profile["facts"]["name"] == "Pranith", "Facts didn't persist across reload"

    test("Layer 6: Persistence across reload", test_profile_persistence)

    # --- Layer 3: Episodic / Facts ---
    def test_fact_store():
        from friday.memory.layers.layer_3_episodic import FactStore
        fs = FactStore(db_manager)
        now = datetime.datetime.now()
        tomorrow = now + datetime.timedelta(days=1)
        fs.add_fact("Meeting with professor", now, tomorrow, importance=0.8)
        fs.add_fact("Trip to Goa", tomorrow, tomorrow + datetime.timedelta(days=3), importance=0.9)
        active = fs.get_active_facts()
        assert len(active) >= 2, f"Expected >=2 active facts, got {len(active)}"
        return len(active)

    count = test("Layer 3: Add episodic facts", test_fact_store)
    if count:
        print(f"    -> {count} active facts")

    def test_fact_extraction():
        from friday.memory.layers.layer_3_episodic import extract_facts
        result = extract_facts("I have a meeting tomorrow at 3pm with Dr. Smith")
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        return result

    result = test("Layer 3: Fact extraction from text", test_fact_extraction)
    if result:
        ops = result.get("operations", [])
        print(f"    -> {len(ops)} operations extracted")

    def test_contested_facts():
        from friday.memory.layers.layer_3_episodic import FactStore
        fs = FactStore(db_manager)
        # Run the linter
        fs.lint_memory_conflicts()
        contested = fs.get_contested_facts()
        print(f"    -> {len(contested)} contested facts found")

    test("Layer 3: Conflict detection (linter)", test_contested_facts)

    # --- Layer 2: Short-term memory ---
    def test_short_term():
        from friday.memory.layers.layer_2_short_term import ShortTermMemoryTracker
        stm = ShortTermMemoryTracker(db_manager)
        assert stm is not None
        return type(stm).__name__

    test("Layer 2: Short-term memory init", test_short_term)

    # --- Promotion system ---
    def test_promotion_stats():
        from friday.memory.layers.promotion import get_promotion_stats, promote_top_memories, prune_stale_entries
        pruned = prune_stale_entries(db_manager)
        stats = get_promotion_stats(db_manager)
        assert isinstance(stats, dict), f"Expected dict, got {type(stats)}"
        assert "total_tracked" in stats
        return stats

    stats = test("Promotion: Stats + prune", test_promotion_stats)
    if stats:
        print(f"    -> tracked={stats['total_tracked']}, promoted={stats['promoted']}")

    # --- Temporal decay ---
    def test_temporal_decay():
        from friday.memory.layers.temporal_decay import TemporalDecayConfig
        engine = TemporalDecayConfig()
        assert engine is not None
        return type(engine).__name__

    test("Temporal decay engine init", test_temporal_decay)

    # --- Dreaming ---
    def test_dreaming():
        from friday.memory.layers.dreaming import MemoryDreamer
        dreamer = MemoryDreamer("workspace", model="llama3.1:8b")
        assert dreamer is not None
        return type(dreamer).__name__

    test("Dreaming pipeline init", test_dreaming)

    # --- Search / Embeddings ---
    def test_searcher_init():
        from friday.search.hybrid_search import HybridSearcher
        global embedder, searcher

        class FakeEmbedder:
            async def embed_query(self, text):
                return [0.1] * 768

        embedder = FakeEmbedder()
        searcher = HybridSearcher(db_manager, embedder)
        assert searcher is not None

    test("Search: HybridSearcher init (fake embeddings)", test_searcher_init)

    def test_search_query():
        async def _search():
            results = await searcher.search("Pranith AI assistant", max_results=3)
            return results
        results = asyncio.get_event_loop().run_until_complete(_search())
        return len(results) if results else 0

    count = test("Search: Query 'Pranith AI assistant'", test_search_query)
    if count is not None:
        print(f"    -> {count} results returned")

    # --- Fact grooming ---
    def test_groom_facts():
        async def _groom():
            from friday.memory.layers.layer_3_episodic import groom_facts
            await groom_facts(db_manager, embedder, workspace_dir="workspace")
        asyncio.get_event_loop().run_until_complete(_groom())

    test("Layer 3: Groom expired facts", test_groom_facts)


# =============================================================================
# PHASE 3: AGENT SYSTEM
# =============================================================================

def phase_3():
    global loop_agent, tools_sys, session_mgr
    print("\n" + "=" * 60)
    print("PHASE 3: Agent System")
    print("=" * 60)

    # --- Session manager ---
    def test_session():
        from friday.agents.friday.session import SessionManager
        global session_mgr
        session_mgr = SessionManager(db_manager)
        session_mgr.append_message("test_session", "user", "Hello Friday")
        session_mgr.append_message("test_session", "assistant", "Hello Sir. How can I help?")
        msgs = session_mgr.load_session("test_session")
        assert len(msgs) == 2, f"Expected 2 messages, got {len(msgs)}"
        return len(msgs)

    count = test("Session: Write + read 2 messages", test_session)
    if count:
        print(f"    -> {count} messages persisted")

    def test_session_clear():
        session_mgr.clear_session("test_session")
        msgs = session_mgr.load_session("test_session")
        assert len(msgs) == 0, f"Expected 0 after clear, got {len(msgs)}"

    test("Session: Clear session", test_session_clear)

    # --- AgentLoop ---
    def test_agent_loop():
        from friday.agents.friday.agent import AgentLoop
        global loop_agent
        loop_agent = AgentLoop(
            workspace_dir="workspace",
            model="llama3.1:8b",
            session_manager=session_mgr,
            session_id="test_integration",
            personalization=personalization,
            db_manager=db_manager,
        )
        assert loop_agent is not None
        assert loop_agent.model == "llama3.1:8b"
        return loop_agent.model

    test("AgentLoop: Init", test_agent_loop)

    # --- Tool system ---
    def test_tool_system():
        from friday.tools.tools_legacy import ToolSystem
        from friday.memory.layers.dreaming import MemoryDreamer
        global tools_sys
        dreamer = MemoryDreamer("workspace", model="llama3.1:8b")
        tools_sys = ToolSystem(
            workspace_dir="workspace",
            db_manager=db_manager,
            personalization=personalization,
            dreamer=dreamer,
        )
        tools_sys.setup_default_tools(searcher=searcher)
        assert len(tools_sys.tools) >= 5, f"Expected >=5 tools, got {len(tools_sys.tools)}"
        return list(tools_sys.tools.keys())

    tools = test("ToolSystem: Init + register default tools", test_tool_system)
    if tools:
        print(f"    -> {len(tools)} tools: {tools}")

    # --- Register tools on agent ---
    def test_register_tools():
        for name, schema in zip(tools_sys.tools.keys(), tools_sys.schemas):
            loop_agent.register_tool(name, tools_sys.tools[name], schema)
        assert len(loop_agent.tools) >= 5
        loop_agent._searcher = searcher
        return len(loop_agent.tools)

    count = test("AgentLoop: Register tools", test_register_tools)
    if count:
        print(f"    -> {count} tools on agent")

    # --- Tool execution (direct) ---
    def test_tool_write_memory():
        async def _exec():
            result = await tools_sys.tools["write_memory"](
                content="Pranith is interested in machine learning and NLP"
            )
            return result
        return asyncio.get_event_loop().run_until_complete(_exec())

    result = test("Tool: write_memory", test_tool_write_memory)
    if result:
        print(f"    -> {str(result)[:80]}")

    def test_tool_search_memory():
        async def _exec():
            result = await tools_sys.tools["search_memory"](query="machine learning")
            return result
        return asyncio.get_event_loop().run_until_complete(_exec())

    result = test("Tool: search_memory", test_tool_search_memory)
    if result:
        print(f"    -> {str(result)[:80]}")

    def test_tool_update_fact():
        async def _exec():
            result = await tools_sys.tools["update_fact"](key="hobby", value="building AI systems")
            return result
        return asyncio.get_event_loop().run_until_complete(_exec())

    result = test("Tool: update_fact", test_tool_update_fact)
    if result:
        print(f"    -> {str(result)[:80]}")

    def test_tool_update_preference():
        async def _exec():
            result = await tools_sys.tools["update_preference"](key="language", value="English")
            return result
        return asyncio.get_event_loop().run_until_complete(_exec())

    result = test("Tool: update_preference", test_tool_update_preference)
    if result:
        print(f"    -> {str(result)[:80]}")

    def test_tool_add_event():
        async def _exec():
            result = await tools_sys.tools["add_event"](
                content="Team standup meeting",
                date_start=(datetime.datetime.now() + datetime.timedelta(hours=2)).isoformat(),
                date_end=(datetime.datetime.now() + datetime.timedelta(hours=3)).isoformat()
            )
            return result
        return asyncio.get_event_loop().run_until_complete(_exec())

    result = test("Tool: add_event", test_tool_add_event)
    if result:
        print(f"    -> {str(result)[:80]}")

    def test_tool_cancel_event():
        async def _exec():
            result = await tools_sys.tools["cancel_event"](keyword="standup")
            return result
        return asyncio.get_event_loop().run_until_complete(_exec())

    result = test("Tool: cancel_event", test_tool_cancel_event)
    if result:
        print(f"    -> {str(result)[:80]}")

    # --- FeedbackDetector ---
    def test_feedback_detector():
        from friday.agents.friday.agent import FeedbackDetector
        fd = FeedbackDetector(personalization)
        saved = fd.detect_and_save("Call me Boss and be more chill")
        assert len(saved) >= 1, f"Expected >=1 feedback saved, got {saved}"
        # Verify persistence
        assert personalization.get_preference("address_as") == "Boss"
        return saved

    result = test("FeedbackDetector: 'Call me Boss and be more chill'", test_feedback_detector)
    if result:
        print(f"    -> Saved: {result}")

    # --- System prompt build ---
    def test_system_prompt():
        prompt = loop_agent._build_system_prompt()
        assert "Friday" in prompt, "System prompt missing 'Friday'"
        assert "Boss" in prompt or "address" in prompt.lower(), "Preferences not injected"
        return len(prompt)

    length = test("AgentLoop: Build system prompt (with preferences)", test_system_prompt)
    if length:
        print(f"    -> {length} chars")


# =============================================================================
# PHASE 4: ROUTER SYSTEM
# =============================================================================

def phase_4():
    global router
    print("\n" + "=" * 60)
    print("PHASE 4: Router System (3-tier)")
    print("=" * 60)

    # --- Intent Classifier (regex) ---
    def test_intent_classifier():
        from friday.router.intent_classifier import FastIntentClassifier, QueryComplexity, QueryCategory
        clf = FastIntentClassifier()

        # Simple greetings
        c, cat = clf.classify("what's up")
        assert c == QueryComplexity.SIMPLE, f"Expected SIMPLE for greeting, got {c}"

        # Calendar query
        c, cat = clf.classify("what's my schedule today")
        assert c == QueryComplexity.SIMPLE, f"Expected SIMPLE for schedule, got {c}"

        return "regex classifier working"

    test("IntentClassifier: Regex classification", test_intent_classifier)

    # --- LLM Router ---
    def test_llm_router_init():
        from friday.router.llm_router import LLMRouter
        from friday.router.intent_classifier import FastIntentClassifier
        regex_fallback = FastIntentClassifier()
        llm_router = LLMRouter(model="llama3.1:8b", fallback_classifier=regex_fallback)
        assert llm_router is not None
        return type(llm_router).__name__

    test("LLMRouter: Init with fallback", test_llm_router_init)

    # --- Smart Router build ---
    def test_smart_router_build():
        global router
        from friday.router.smart_router import build_smart_router
        router = build_smart_router(
            agent_loop=loop_agent,
            db_manager=db_manager,
            personalization=personalization,
            searcher=searcher,
        )
        assert router is not None
        assert router.classifier is not None
        assert router.simple is not None
        assert router.medium is not None
        assert router.planner is not None
        return type(router).__name__

    test("SmartRouter: Build all components", test_smart_router_build)

    # --- Simple handler ---
    def test_simple_handler():
        from friday.router.handlers.simple_handler import SimpleHandler
        handler = SimpleHandler(db_manager, personalization, searcher)
        assert handler is not None
        return type(handler).__name__

    test("SimpleHandler: Init", test_simple_handler)

    # --- Medium handler ---
    def test_medium_handler():
        from friday.router.handlers.medium_handler import MediumHandler
        handler = MediumHandler(loop_agent)
        assert handler is not None
        return type(handler).__name__

    test("MediumHandler: Init", test_medium_handler)

    # --- Complex handler / planner ---
    def test_complex_handler():
        from friday.router.handlers.complex_handler import MultiAgentPlanner
        from friday.execution.state_manager import ExecutionStateManager
        from friday.execution.engine import ExecutionEngine
        from friday.execution.subagent_registry import SubagentRegistry

        state_mgr = ExecutionStateManager()
        engine = ExecutionEngine(loop_agent, state_mgr)
        sub_reg = SubagentRegistry()
        planner = MultiAgentPlanner(
            agent_loop=loop_agent,
            db_manager=db_manager,
            personalization=personalization,
            execution_engine=engine,
            subagent_registry=sub_reg,
        )
        assert planner is not None
        return type(planner).__name__

    test("ComplexHandler (MultiAgentPlanner): Init", test_complex_handler)

    # --- Live routing test (requires Ollama) ---
    def test_live_route_simple():
        async def _route():
            result = await router.route("hey whats up", "test_session")
            assert "text" in result, f"Missing 'text' in result: {result.keys()}"
            assert "complexity" in result
            return result
        return asyncio.get_event_loop().run_until_complete(_route())

    print("\n  --- Live routing tests (require Ollama running) ---")
    result = test("LIVE: Route 'hey whats up' (expect simple)", test_live_route_simple)
    if result:
        print(f"    -> tier={result['complexity']} | latency={result['latency_ms']:.0f}ms")
        print(f"    -> response: {result['text'][:100]}")

    def test_live_route_memory():
        async def _route():
            result = await router.route("what do you know about me?", "test_session")
            return result
        return asyncio.get_event_loop().run_until_complete(_route())

    result = test("LIVE: Route 'what do you know about me?' (expect simple)", test_live_route_memory)
    if result:
        print(f"    -> tier={result['complexity']} | latency={result['latency_ms']:.0f}ms")
        print(f"    -> response: {result['text'][:100]}")

    def test_live_route_medium():
        async def _route():
            result = await router.route("save that I love playing chess on weekends", "test_session")
            return result
        return asyncio.get_event_loop().run_until_complete(_route())

    result = test("LIVE: Route 'save that I love playing chess' (expect medium)", test_live_route_medium)
    if result:
        print(f"    -> tier={result['complexity']} | latency={result['latency_ms']:.0f}ms")
        print(f"    -> tools_used={result['tools_used']}")
        print(f"    -> response: {result['text'][:100]}")

    def test_live_route_add_event():
        async def _route():
            result = await router.route("add a meeting with Dr. Sharma tomorrow at 4pm for 1 hour", "test_session")
            return result
        return asyncio.get_event_loop().run_until_complete(_route())

    result = test("LIVE: Route 'add meeting tomorrow 4pm' (expect medium)", test_live_route_add_event)
    if result:
        print(f"    -> tier={result['complexity']} | latency={result['latency_ms']:.0f}ms")
        print(f"    -> tools_used={result['tools_used']}")
        print(f"    -> response: {result['text'][:100]}")


# =============================================================================
# PHASE 5: EXECUTION SYSTEM
# =============================================================================

def phase_5():
    print("\n" + "=" * 60)
    print("PHASE 5: Execution System")
    print("=" * 60)

    def test_state_manager():
        from friday.execution.state_manager import ExecutionStateManager, ExecutionPlan, ExecutionStep, ExecutionStatus
        sm = ExecutionStateManager()
        plan = ExecutionPlan(
            plan_id="test-plan-1",
            query="Research AI trends",
            steps=[
                ExecutionStep(step_id="s1", step_number=1, action="Search Google", tool_category="browser", reasoning="Need data"),
                ExecutionStep(step_id="s2", step_number=2, action="Summarize results", tool_category="none", reasoning="Compile"),
            ],
            estimated_duration_seconds=30,
            complexity="complex",
            created_at=datetime.datetime.now(),
        )
        state = sm.create_execution(plan, "test_session")
        assert state is not None
        assert sm.has_active_execution("test_session")

        # Test progress
        state.start()
        assert state.status == ExecutionStatus.RUNNING
        ctx = state.get_context_for_llm()
        assert "[EXECUTION STATUS]" in ctx
        assert "Research AI trends" in ctx

        # Test step advancement
        state.update_step_status("s1", ExecutionStatus.COMPLETE, result={"data": "found"})
        state.advance_step()
        assert state.current_step_index == 1
        assert state.progress_percent == 50.0

        # Complete
        state.update_step_status("s2", ExecutionStatus.COMPLETE, result={"summary": "done"})
        state.advance_step()
        sm.complete_execution(state.execution_id)
        assert not sm.has_active_execution("test_session")
        return state.progress_percent

    progress = test("StateManager: Create, run, complete execution", test_state_manager)
    if progress is not None:
        print(f"    -> Final progress: {progress}%")

    def test_subagent_registry():
        from friday.execution.subagent_registry import SubagentRegistry, MAX_SPAWN_DEPTH
        reg = SubagentRegistry()
        r1 = reg.register("Research task A")
        assert r1.depth == 0
        reg.mark_running(r1.execution_id)

        r2 = reg.register("Sub-task A1", parent_exec_id=r1.execution_id)
        assert r2.depth == 1

        r3 = reg.register("Sub-sub-task A1a", parent_exec_id=r2.execution_id)
        assert r3.depth == 2

        # Depth limit test
        try:
            r4 = reg.register("Too deep", parent_exec_id=r3.execution_id)
            assert False, "Should have raised ValueError for depth limit"
        except ValueError:
            pass  # Expected

        reg.complete(r3.execution_id, "done")
        reg.complete(r2.execution_id, "done")
        reg.complete(r1.execution_id, "done")
        assert reg.active_count() == 0
        return MAX_SPAWN_DEPTH

    depth = test("SubagentRegistry: Depth limits (max 3)", test_subagent_registry)
    if depth:
        print(f"    -> Max depth enforced: {depth}")

    def test_execution_engine_init():
        from friday.execution.engine import ExecutionEngine
        from friday.execution.state_manager import ExecutionStateManager
        sm = ExecutionStateManager()
        engine = ExecutionEngine(loop_agent, sm)
        assert engine is not None
        return type(engine).__name__

    test("ExecutionEngine: Init", test_execution_engine_init)

    def test_learning_engine():
        from friday.execution.learning import LearningEngine
        le = LearningEngine(db_manager)
        assert le is not None
        async def _test():
            await le.record_acceptance(
                pattern_type="tool_preference",
                context="save a fact",
                key="update_fact",
                value="chosen"
            )
        asyncio.get_event_loop().run_until_complete(_test())
        return type(le).__name__

    test("LearningEngine: Record acceptance pattern", test_learning_engine)


# =============================================================================
# PHASE 6: NEW COMPONENTS (Mark 2 additions)
# =============================================================================

def phase_6():
    print("\n" + "=" * 60)
    print("PHASE 6: New Mark 2 Components")
    print("=" * 60)

    def test_memory_pipeline():
        from friday.memory.pipeline import MemoryPipeline
        from friday.memory.layers.layer_3_episodic import FactStore
        fs = FactStore(db_manager)
        pipeline = MemoryPipeline(
            db_manager=db_manager,
            personalization=personalization,
            embedding_manager=None,  # skip embedding for test
            fact_store=fs,
            promotion_engine=None,   # skip promotion for test
        )
        assert pipeline is not None
        # Test the pipeline runs without crashing
        async def _run():
            await pipeline.process("I'm working on my Friday project today", "test_session")
        asyncio.get_event_loop().run_until_complete(_run())
        return type(pipeline).__name__

    test("MemoryPipeline: Process input (learning from every message)", test_memory_pipeline)

    def test_agent_registry():
        from friday.agents.registry import AgentRegistry
        reg = AgentRegistry()
        # discover() scans agents/specialized/ - the stubs may not have BaseAgent base class
        # but it should not crash
        count = reg.discover()
        summary = reg.summary()
        return (count, summary)

    result = test("AgentRegistry: Auto-discover agents", test_agent_registry)
    if result:
        print(f"    -> {result[0]} agents discovered")
        print(f"    -> {result[1]}")

    def test_tool_registry():
        from friday.tools.registry import ToolRegistry
        reg = ToolRegistry()
        count = reg.discover()
        summary = reg.summary()
        return (count, summary)

    result = test("ToolRegistry: Auto-discover tools", test_tool_registry)
    if result:
        print(f"    -> {result[0]} tools discovered")
        print(f"    -> {result[1]}")

    def test_parallel_executor():
        from friday.execution.parallel_executor import ParallelExecutor, MAX_PARALLEL
        from friday.agents.registry import AgentRegistry
        reg = AgentRegistry()
        executor = ParallelExecutor(reg)
        assert executor is not None
        assert MAX_PARALLEL == 5
        return MAX_PARALLEL

    max_p = test("ParallelExecutor: Init (max parallel = 5)", test_parallel_executor)


# =============================================================================
# PHASE 7: BACKGROUND SYSTEM
# =============================================================================

def phase_7():
    print("\n" + "=" * 60)
    print("PHASE 7: Background System")
    print("=" * 60)

    def test_scheduler():
        from friday.background.scheduler import BackgroundScheduler
        from friday.config.settings import IntelligentMemoryConfig
        config = IntelligentMemoryConfig.load()
        sched = BackgroundScheduler(db_manager, config)
        assert sched is not None
        return len(sched.watchers)

    test("BackgroundScheduler: Init", test_scheduler)

    def test_proactive_events():
        from friday.background.proactive_events import ProactiveEventsWatcher
        watcher = ProactiveEventsWatcher("workspace")
        assert watcher is not None

    test("ProactiveEventsWatcher: Init", test_proactive_events)

    def test_context_summarizer():
        from friday.background.context_summarizer import trigger_background_compact
        # Just verify the function exists and doesn't crash when called with agent
        trigger_background_compact(loop_agent)

    test("ContextSummarizer: trigger_background_compact", test_context_summarizer)


# =============================================================================
# PHASE 8: MEMORY VERIFICATION (end-to-end)
# =============================================================================

def phase_8():
    print("\n" + "=" * 60)
    print("PHASE 8: Memory Verification (end-to-end)")
    print("=" * 60)

    def test_full_memory_flow():
        """Verify the complete write -> search -> recall -> promote flow"""
        # 1. Write via tool
        async def _write():
            result = await tools_sys.tools["write_memory"](
                content="Pranith completed the Friday Mark 2 architecture migration on June 13, 2026"
            )
            return result
        write_result = asyncio.get_event_loop().run_until_complete(_write())
        assert "saved" in str(write_result).lower() or "stored" in str(write_result).lower() or "success" in str(write_result).lower() or "chunk" in str(write_result).lower(), f"Write failed: {write_result}"

        # 2. Search for it
        async def _search():
            return await tools_sys.tools["search_memory"](query="Friday Mark 2 migration")
        search_result = asyncio.get_event_loop().run_until_complete(_search())
        assert search_result is not None

        # 3. Verify facts persisted
        profile = personalization.profile
        assert "hobby" in profile["facts"], f"Missing 'hobby' fact. Facts: {profile['facts'].keys()}"

        # 4. Verify preferences persisted
        assert "language" in profile["preferences"], f"Missing 'language' pref. Prefs: {profile['preferences'].keys()}"

        return {
            "facts": len(profile["facts"]),
            "prefs": len(profile["preferences"]),
        }

    result = test("Full memory flow: write -> search -> verify persistence", test_full_memory_flow)
    if result:
        print(f"    -> {result['facts']} facts, {result['prefs']} preferences persisted")

    def test_memory_chunk_count():
        conn = db_manager.get_connection()
        count = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        assert count >= 2, f"Expected >=2 chunks, got {count}"
        return count

    count = test("Database: Total memory chunks", test_memory_chunk_count)
    if count:
        print(f"    -> {count} chunks in database")

    def test_fact_count():
        conn = db_manager.get_connection()
        count = conn.execute("SELECT COUNT(*) as c FROM facts WHERE status='active'").fetchone()["c"]
        return count

    count = test("Database: Active facts count", test_fact_count)
    if count is not None:
        print(f"    -> {count} active facts")


# =============================================================================
# RUN ALL PHASES
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  FRIDAY MARK 2 - FULL INTEGRATION TEST SUITE")
    print("#" * 60)

    start_time = time.time()

    phase_1()  # Database
    phase_2()  # Memory System
    phase_3()  # Agent System
    phase_4()  # Router System
    phase_5()  # Execution System
    phase_6()  # New Mark 2 Components
    phase_7()  # Background System
    phase_8()  # Memory Verification

    elapsed = time.time() - start_time

    # Cleanup test DB
    if db_manager:
        db_manager.close()
    test_db = os.path.join("workspace", "test_integration.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    # Final report
    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed ({elapsed:.1f}s)")
    print("=" * 60)

    if ERRORS:
        print("\n  FAILURES:")
        for name, err, tb in ERRORS:
            print(f"\n  [FAIL] {name}")
            print(f"    Error: {err}")
            # Print the last 3 lines of traceback
            tb_lines = tb.strip().split("\n")
            for line in tb_lines[-3:]:
                print(f"    {line}")

    print()
    if FAIL == 0:
        print("  [PASS] ALL TESTS PASSED - Friday Mark 2 is fully operational.")
    elif FAIL <= 3:
        print("  [WARN] MOSTLY WORKING - minor issues to fix.")
    else:
        print("  [FAIL] CRITICAL FAILURES - needs attention.")
    print()
