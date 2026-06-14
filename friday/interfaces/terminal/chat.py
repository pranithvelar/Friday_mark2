import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from friday.config.settings import IntelligentMemoryConfig
from friday.memory.db_manager import MemoryDatabaseManager
from friday.search.embedding_manager import EmbeddingManager
from friday.search.hybrid_search import HybridSearcher
from friday.tools.tools_legacy import ToolSystem, reindex_existing_files
from friday.agents.friday.agent import AgentLoop
from friday.router.smart_router import build_smart_router
from friday.agents.friday.session import SessionManager
from friday.memory.layers.layer_6_profile import UserPersonalization
from friday.memory.layers.dreaming import MemoryDreamer
from friday.memory.layers.promotion import promote_top_memories, prune_stale_entries, get_promotion_stats
from friday.memory.layers.layer_3_episodic import groom_facts
from friday.llm import build_provider

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
    db_manager.ensure_vector_table(dimensions=768)
    print("[OK] Database initialized (all state in SQLite)")

    # 2. Embeddings (real with fallback)
    use_real_embeddings = True
    try:
        embedder = EmbeddingManager(db_manager, model=config.embedding_model, dimension=768)
        test_vec = await embedder.embed_query("test")
        if len(test_vec) != 768:
            raise ValueError(f"Bad dimension: {len(test_vec)}")
        print(f"[OK] Real embeddings active ({config.embedding_model})")
    except Exception as e:
        print(f"[WARN] Embeddings unavailable ({e}). Keyword-only search.")
        use_real_embeddings = False
        class FallbackEmbedder:
            async def embed_query(self, text):
                return [0.0] * 768
        embedder = FallbackEmbedder()

    # 3. Searcher
    searcher = HybridSearcher(db_manager, embedder)

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
        await groom_facts(db_manager, embedder, workspace_dir=workspace)
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

    # 11. Smart Router (3-Tier Routing: Simple → Medium → Complex Multi-Agent)
    router = build_smart_router(
        agent_loop=loop,
        db_manager=db_manager,
        personalization=personalization,
        searcher=searcher,
        llm_provider=llm_provider,
    )
    print("[OK] Smart Router ready (Simple / Medium / Complex Multi-Agent)")

    # Start Background Worker
    from friday.background.scheduler import BackgroundScheduler
    from friday.background.proactive_events import ProactiveEventsWatcher
    from friday.background.google_workspace import GoogleWorkspaceWatcher
    bg_scheduler = BackgroundScheduler(db_manager, config)
    bg_scheduler.register_watcher(ProactiveEventsWatcher(workspace))
    bg_scheduler.register_watcher(GoogleWorkspaceWatcher())
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
                    print(f"  Embeddings: {'Real' if use_real_embeddings else 'Fallback'}")
                    print(f"  Context window: last 10 + summary")
                    print(f"  --- Promotion ---")
                    print(f"  Tracked: {ps['total_tracked']}, Promoted: {ps['promoted']}, Pending: {ps['pending']}")
                    print(f"  Scores: {ps['min_score']:.3f} - {ps['max_score']:.3f} (avg {ps['avg_score']:.3f})\n")
                except Exception as e:
                    print(f"  Status error: {e}\n")
                continue

            print("Processing...")
            try:
                result = await router.route(user_input, SESSION_ID)
                print(f"\nFriday: {result['text']}\n")
                # Debug tier info (remove or comment out when not needed)
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
