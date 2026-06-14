import os
import json
import hashlib
import time
import logging
import uuid
from typing import Dict, Any, Callable, List

logger = logging.getLogger(__name__)


def reindex_existing_files(workspace_dir: str, db_manager) -> int:
    """Index any .txt/.md memory files not yet in the database."""
    memory_dir = os.path.join(workspace_dir, "memory")
    if not os.path.isdir(memory_dir):
        return 0

    conn = db_manager.get_connection()
    indexed = 0
    for fname in os.listdir(memory_dir):
        fpath = os.path.join(memory_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.endswith((".txt", ".md")):
            continue

        # Check if already indexed (DB uses 'path' column)
        existing = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE path = ?",
            (fpath,)
        ).fetchone()
        if existing and existing["c"] > 0:
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                file_content = f.read().strip()
            if not file_content:
                continue

            # Check for duplicate content in DB ('content' column, not 'snippet')
            dup = conn.execute(
                "SELECT COUNT(*) as c FROM chunks WHERE content = ?",
                (file_content[:500],)
            ).fetchone()
            if dup and dup["c"] > 0:
                continue

            chunk_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO chunks (id, path, source, chunkIndex, content) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, fpath, "memory", 0, file_content[:500])
            )
            # Also insert into FTS table
            try:
                conn.execute(
                    "INSERT INTO chunks_fts (id, path, source, content) VALUES (?, ?, ?, ?)",
                    (chunk_id, fpath, "memory", file_content[:500])
                )
            except Exception:
                pass  # FTS insert is best-effort
            conn.commit()
            indexed += 1
        except Exception as e:
            logger.warning(f"Failed to index {fname}: {e}")

    return indexed


class ToolSystem:
    def __init__(self, workspace_dir: str, db_manager, personalization=None, dreamer=None):
        self.workspace_dir = workspace_dir
        self.db_manager = db_manager
        self.personalization = personalization
        self.dreamer = dreamer
        self.schemas = []
        self.tools = {}
        self._write_count = 0

    def register_tool(self, name: str, schema: Dict[str, Any], func: Callable):
        self.schemas.append(schema)
        self.tools[name] = func

    def setup_default_tools(self, searcher):
        self.register_tool(
            "search_memory",
            {
                "name": "search_memory",
                "description": "Search the local AI memory system using hybrid semantic and keyword search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "limit": {"type": "integer", "description": "Number of results.", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            self._wrap_tool(self._tool_search_memory, searcher)
        )

        self.register_tool(
            "write_memory",
            {
                "name": "write_memory",
                "description": "Write a thought, note, or new knowledge into the memory system. Use when the user asks you to remember something.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The text content to save."},
                        "filename": {"type": "string", "description": "Optional name. Defaults to timestamped."}
                    },
                    "required": ["content"]
                }
            },
            self._wrap_tool(self._tool_write_memory)
        )

        self.register_tool(
            "update_preference",
            {
                "name": "update_preference",
                "description": "Save or update a user preference (tone, style, name to use, etc).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Preference key (e.g. 'tone', 'address_as')."},
                        "value": {"type": "string", "description": "Preference value."}
                    },
                    "required": ["key", "value"]
                }
            },
            self._wrap_tool(self._tool_update_preference)
        )

        self.register_tool(
            "update_fact",
            {
                "name": "update_fact",
                "description": "Save or update a fact about the user (name, role, interests).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Fact key (e.g. 'name', 'occupation')."},
                        "value": {"type": "string", "description": "Fact value."}
                    },
                    "required": ["key", "value"]
                }
            },
            self._wrap_tool(self._tool_update_fact)
        )

        self.register_tool(
            "dream_now",
            {
                "name": "dream_now",
                "description": "Trigger the dreaming pipeline to consolidate memories into a narrative.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            },
            self._wrap_tool(self._tool_dream_now)
        )

        self.register_tool(
            "cancel_event",
            {
                "name": "cancel_event",
                "description": "Cancel, delete, or remove an event or fact explicitly by keyword (e.g., 'math exam'). Use when user asks to cancel an event.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "Keyword of the event to cancel."}
                    },
                    "required": ["keyword"]
                }
            },
            self._wrap_tool(self._tool_cancel_event)
        )

        self.register_tool(
            "add_event",
            {
                "name": "add_event",
                "description": "Add a new rigid event or plan to the user's permanent calendar. Use when the user dictates a firm schedule or deadline.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Description of the event."},
                        "date_start": {"type": "string", "description": "Start date in ISO format, e.g., '2026-05-18T10:00:00'."},
                        "date_end": {"type": "string", "description": "End date in ISO format (Optional)."}
                    },
                    "required": ["content", "date_start"]
                }
            },
            self._wrap_tool(self._tool_add_event)
        )

    def _wrap_tool(self, func: Callable, *args):
        async def wrapper(**kwargs):
            return await func(*args, **kwargs)
        return wrapper

    async def _tool_cancel_event(self, keyword: str):
        from friday.memory.layers.layer_3_episodic import FactStore
        store = FactStore(self.db_manager)
        deleted = store.delete_fact(keyword)
        if deleted:
            return f"Event/Fact containing '{keyword}' has been successfully cancelled and removed from active memory."
        else:
            return f"Could not find any active event or fact containing the keyword '{keyword}'."

    async def _tool_add_event(self, content: str, date_start: str, date_end: str = None):
        from friday.memory.layers.layer_3_episodic import FactStore
        from datetime import datetime
        store = FactStore(self.db_manager)
        try:
            start_dt = datetime.fromisoformat(date_start)
            end_dt = datetime.fromisoformat(date_end) if date_end else None
            store.add_fact(content, start_dt, end_dt)
            return f"Event smoothly scheduled: {content} on {date_start}"
        except Exception as e:
            return f"Failed to add event formatting: {e}"

    async def _tool_search_memory(self, searcher, query: str, limit: int = 5):
        try:
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                limit = 5
            # searcher.search() returns List[HybridSearchResult] objects (not dicts)
            results = await searcher.search(query, vector_weight=0.5, text_weight=0.5, max_results=limit)
            if not results:
                return "No memories found for that query."

            # Record recall stats in DB
            self._record_recall_stats(query, results)

            # HybridSearchResult has attributes: .path, .snippet, .score, .chunk_id
            summary = []
            for item in results:
                summary.append(f"Source: {item.path} (Score: {item.score:.3f})\n{item.snippet}")
            return "\n\n".join(summary)
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return f"Error executing search: {e}"

    def _record_recall_stats(self, query: str, results: list):
        """Record which memories were recalled for promotion scoring — using SQL."""
        try:
            conn = self.db_manager.get_connection()
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            today = time.strftime("%Y-%m-%d", time.gmtime())
            query_hash = hashlib.sha1(query.lower().strip().encode()).hexdigest()[:12]

            for result in results:
                path = result.path
                snippet = result.snippet[:300] if result.snippet else ""
                score = result.score
                key = f"memory:{path}"

                existing = conn.execute(
                    "SELECT * FROM short_term_recall WHERE key = ?", (key,)
                ).fetchone()

                if not existing:
                    conn.execute(
                        """INSERT INTO short_term_recall 
                        (key, path, start_line, end_line, source, snippet, recall_count, daily_count,
                         grounded_count, total_score, max_score, first_recalled_at, last_recalled_at,
                         query_hashes, recall_days, concept_tags, claim_hash)
                        VALUES (?, ?, 1, 1, 'memory', ?, 1, 1, 0, ?, ?, ?, ?, ?, ?, '[]', '')""",
                        (key, path, snippet, score, score, now_iso, now_iso,
                         json.dumps([query_hash]), json.dumps([today]))
                    )
                else:
                    query_hashes = json.loads(existing["query_hashes"] or "[]")
                    recall_days = json.loads(existing["recall_days"] or "[]")

                    # Dedupe: don't double-count same query on same day
                    if query_hash in query_hashes and today in recall_days:
                        continue

                    if query_hash not in query_hashes:
                        query_hashes.append(query_hash)
                        if len(query_hashes) > 32:
                            query_hashes = query_hashes[-32:]
                    
                    new_daily = existing["daily_count"]
                    if today not in recall_days:
                        recall_days.append(today)
                        new_daily += 1
                        if len(recall_days) > 16:
                            recall_days = recall_days[-16:]

                    conn.execute(
                        """UPDATE short_term_recall SET
                            recall_count = recall_count + 1,
                            daily_count = ?,
                            total_score = total_score + ?,
                            max_score = MAX(max_score, ?),
                            last_recalled_at = ?,
                            query_hashes = ?,
                            recall_days = ?
                        WHERE key = ?""",
                        (new_daily, score, score, now_iso,
                         json.dumps(query_hashes), json.dumps(recall_days), key)
                    )

            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record recall stats: {e}")

    async def _tool_write_memory(self, content: str, filename: str = None):
        import uuid as _uuid
        if not filename:
            filename = f"note_{int(time.time())}.txt"
        if not filename.endswith((".txt", ".md")):
            filename += ".txt"

        target_path = os.path.join(self.workspace_dir, "memory", filename)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Deduplication via DB meta table
        content_hash = hashlib.md5(content.encode()).hexdigest()
        conn = self.db_manager.get_connection()
        try:
            existing_hash = conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                (f"content_hash:{content_hash}",)
            ).fetchone()
            if existing_hash:
                return f"Memory already exists (duplicate): {existing_hash['value']}"
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                (f"content_hash:{content_hash}", filename)
            )
            conn.commit()
        except Exception:
            pass

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Index into DB with correct columns: id, path, source, chunkIndex, content
        try:
            chunk_id = str(_uuid.uuid4())
            conn.execute(
                "INSERT INTO chunks (id, path, source, chunkIndex, content) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, target_path, "memory", 0, content[:500])
            )
            try:
                conn.execute(
                    "INSERT INTO chunks_fts (id, path, source, content) VALUES (?, ?, ?, ?)",
                    (chunk_id, target_path, "memory", content[:500])
                )
            except Exception:
                pass
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to index written memory: {e}")

        # Auto-dreaming trigger
        self._write_count += 1
        auto_dream_msg = ""
        if self._write_count % 5 == 0 and self.dreamer:
            try:
                dream_result = self.dreamer.dream_sync(self.db_manager)
                if dream_result:
                    auto_dream_msg = f" (Auto-dream triggered)"
                    from friday.memory.layers.promotion import promote_top_memories, prune_stale_entries
                    promoted = promote_top_memories(self.db_manager)
                    pruned = prune_stale_entries(self.db_manager)
                    if promoted:
                        auto_dream_msg += f" Promoted {len(promoted)} memories."
                    if pruned:
                        auto_dream_msg += f" Pruned {pruned} stale entries."
            except Exception as e:
                logger.warning(f"Auto-dreaming failed: {e}")

        return f"Memory saved to {filename} and indexed.{auto_dream_msg}"

    async def _tool_update_preference(self, key: str, value: str):
        if not self.personalization:
            return "Personalization not available."
        self.personalization.update_preference(key, value)
        return f"Preference updated: {key} = {value}"

    async def _tool_update_fact(self, key: str, value: str):
        if not self.personalization:
            return "Personalization not available."
        self.personalization.update_fact(key, value)
        return f"Fact recorded: {key} = {value}"

    async def _tool_dream_now(self):
        if not self.dreamer:
            return "Dreaming pipeline not available."
        try:
            result = self.dreamer.dream_sync(self.db_manager)
            response = result if result else "No memories to consolidate yet."

            from friday.memory.layers.promotion import promote_top_memories, prune_stale_entries
            promoted = promote_top_memories(self.db_manager)
            pruned = prune_stale_entries(self.db_manager)

            if promoted:
                response += f"\n\nPromotion: {len(promoted)} memories promoted to PROMOTED.md"
                for p in promoted:
                    response += f"\n  -> {p['snippet'][:80]}... (score: {p['score']:.3f})"
            if pruned:
                response += f"\nPruned {pruned} stale entries."

            return response
        except Exception as e:
            logger.error(f"Dreaming failed: {e}")
            return f"Dreaming error: {e}"
