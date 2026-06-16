"""
Text Indexer — Pure Database Insertion
======================================
Replaces EmbeddingManager in the core memory pipeline.
Instead of generating vectors, it simply writes text into `chunks` and `chunks_fts`
to make it instantaneously searchable via FTSSearcher.
"""

import uuid
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TextIndexer:
    """
    Inserts text into the SQLite memory tables (chunks, chunks_fts) without embedding.
    """
    def __init__(self, db_manager):
        self.db = db_manager

    async def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Stores the text in chunks and chunks_fts.
        Returns the new chunk_id.
        """
        if not text or not text.strip():
            return None

        chunk_id = str(uuid.uuid4())
        meta = metadata or {}
        
        source = meta.get("source", "unknown")
        session_id = meta.get("session_id")
        role = meta.get("role", "unknown")
        path = meta.get("path", "")
        start_line = meta.get("start_line", 0)
        end_line = meta.get("end_line", 0)
        timestamp = meta.get("timestamp", time.time())

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert metadata into main chunks table
                cursor.execute('''
                    INSERT INTO chunks (id, path, source, chunkIndex, content)
                    VALUES (?, ?, ?, ?, ?)
                ''', (chunk_id, path, source, 0, text))

                # Insert raw text into FTS index
                cursor.execute('''
                    INSERT INTO chunks_fts (rowid, content)
                    VALUES (?, ?)
                ''', (cursor.lastrowid, text))

                conn.commit()
                return chunk_id
                
        except Exception as e:
            logger.error(f"[TextIndexer] Failed to store chunk: {e}")
            return None
