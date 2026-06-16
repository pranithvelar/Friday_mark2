"""
FTS Searcher — Pure Keyword Search
==================================
Replaces the old vector-based HybridSearcher to eliminate latency and DB locks.
Uses SQLite FTS5 (Full Text Search) with BM25 ranking for instantaneous retrieval
of recent conversation history.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FTSResult:
    chunk_id: str
    path: str
    result_source: str
    snippet: str
    start_line: int
    end_line: int
    score: float = 0.0

class FTSSearcher:
    def __init__(self, db_manager):
        self.db = db_manager

    async def search(
        self,
        query: str,
        max_results: int = 10,
        source_filters: Optional[List[str]] = None,
        # Accepted for backward compatibility with old interface, but ignored
        vector_weight: float = 0.0,
        text_weight: float = 1.0,
    ) -> List[FTSResult]:
        """
        Search `chunks_fts` using pure BM25.
        Returns the top `max_results` matches.
        """
        if not query or not query.strip():
            return []

        # Convert query to FTS5 MATCH syntax (basic tokenization)
        # We strip non-alphanumeric chars for safety in FTS MATCH
        import re
        clean_words = [w for w in re.split(r'\W+', query) if w]
        if not clean_words:
            return []
        
        # Simple OR query for keywords to maximize recall
        match_query = " OR ".join(clean_words)

        source_clause = ""
        params = [match_query]

        if source_filters:
            placeholders = ",".join(["?"] * len(source_filters))
            source_clause = f" AND source IN ({placeholders}) "
            params.extend(source_filters)
            
        params.append(max_results)

        sql = f"""
            SELECT 
                c.id, 
                c.path, 
                c.source, 
                c.content,
                bm25(chunks_fts) as rank
            FROM chunks_fts f
            JOIN chunks c ON f.rowid = c.rowid
            WHERE chunks_fts MATCH ? {source_clause}
            ORDER BY rank
            LIMIT ?
        """

        results_dict: Dict[str, FTSResult] = {}
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # FTS5 bm25() returns negative values where more negative = better match.
                # We normalize it to a positive score (roughly).
                for row in rows:
                    chunk_id = row[0]
                    # basic normalization, since bm25 is negative and unbounded
                    score = abs(row[4]) 
                    res = FTSResult(
                        chunk_id=chunk_id,
                        path=row[1] or "",
                        result_source=row[2] or "unknown",
                        snippet=row[3] or "",
                        start_line=0,
                        end_line=0,
                        score=score
                    )
                    results_dict[chunk_id] = res
                    
        except Exception as e:
            logger.warning(f"[FTSSearcher] FTS search failed: {e}")
            return []

        # Sort descending by score
        sorted_results = sorted(results_dict.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:max_results]
