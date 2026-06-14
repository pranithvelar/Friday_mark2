import re
import json
import logging
from typing import List, Dict, Any, Optional
from friday.cache.search_cache import RedisSearchCache

logger = logging.getLogger(__name__)

class HybridSearchResult:
    def __init__(self, chunk_id: str, path: str, result_source: str, snippet: str, start_line: int, end_line: int, vector_score: float = 0.0, text_score: float = 0.0):
        self.chunk_id = chunk_id
        self.path = path
        self.result_source = result_source # 'memory', 'sessions', 'wiki'
        self.snippet = snippet
        self.start_line = start_line
        self.end_line = end_line
        self.vector_score = vector_score
        self.text_score = text_score
        self.score = 0.0 # final combined score

class HybridSearcher:
    def __init__(self, db_manager, embedding_manager):
        self.db = db_manager
        self.embeddings = embedding_manager
        self.redis_cache = RedisSearchCache()

    def build_fts_query(self, raw_query: str) -> str:
        # Use OR so partial matches work (e.g. "favorite color" matches docs with just "color")
        tokens = re.findall(r'[\w]+', raw_query.lower())
        if not tokens:
            return ""
        return " OR ".join(f'"{t}"' for t in tokens)

    def bm25_rank_to_score(self, rank: float) -> float:
        if rank < 0:
            relevance = -rank
            return relevance / (1.0 + relevance)
        return 1.0 / (1.0 + rank)

    async def search_vector(self, query_vec: List[float], limit: int, source_filters: List[str]) -> Dict[str, HybridSearchResult]:
        if not query_vec or limit <= 0:
            return {}
            
        conn = self.db.get_connection()
        
        source_clause = ""
        if source_filters:
            placeholders = ",".join(["?"] * len(source_filters))
            source_clause = f" AND c.source IN ({placeholders})"
            
        # Using sqlite-vec knn
        query = f"""
            SELECT c.id, c.path, c.source, c.content, c.chunkIndex,
                   vec_distance_cosine(v.embedding, ?) AS dist
            FROM chunks_vec v
            JOIN chunks c ON c.id = v.id
            WHERE v.embedding MATCH ? AND k = ? {source_clause}
            ORDER BY dist ASC
        """
        
        vec_bytes = json.dumps(query_vec)
        params = [vec_bytes, vec_bytes, limit * 4]
        if source_filters:
            params.extend(source_filters)
            
        rows = conn.execute(query, params).fetchall()
        
        results = {}
        for row in rows:
            dist = float(row["dist"])
            vector_score = max(0.0, 1.0 - dist)
            results[row["id"]] = HybridSearchResult(
                chunk_id=row["id"],
                path=row["path"],
                result_source=row["source"],
                snippet=row["content"],
                start_line=row["chunkIndex"], # simplified mapping
                end_line=row["chunkIndex"],
                vector_score=vector_score,
                text_score=0.0
            )
            
        return results

    async def search_keyword(self, query: str, limit: int, source_filters: List[str]) -> Dict[str, HybridSearchResult]:
        fts_query = self.build_fts_query(query)
        if not fts_query or limit <= 0:
            return {}
            
        conn = self.db.get_connection()
        
        source_clause = ""
        if source_filters:
            placeholders = ",".join(["?"] * len(source_filters))
            source_clause = f" AND source IN ({placeholders})"
            
        params = [fts_query]
        if source_filters:
            params.extend(source_filters)
            
        # SQLite FTS5 bm25 function
        sql = f"""
            SELECT id, path, source, content, bm25(chunks_fts) as rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ? {source_clause}
            ORDER BY rank ASC
            LIMIT {limit * 4}
        """
        
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:
            logger.warning(f"FTS5 MATCH failed: {e}")
            return {}
            
        results = {}
        for row in rows:
            rank = float(row["rank"])
            text_score = self.bm25_rank_to_score(rank)
            results[row["id"]] = HybridSearchResult(
                chunk_id=row["id"],
                path=row["path"],
                result_source=row["source"],
                snippet=row["content"],
                start_line=0,
                end_line=0,
                vector_score=0.0,
                text_score=text_score
            )
            
        return results

    async def search_like_fallback(self, query: str, limit: int, source_filters: List[str]) -> Dict[str, HybridSearchResult]:
        """Fallback LIKE search when FTS5 returns nothing."""
        conn = self.db.get_connection()
        tokens = re.findall(r'[\w]+', query.lower())
        if not tokens:
            return {}
        # Search for any token appearing in content
        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in tokens])
        params = [f"%{t}%" for t in tokens]

        source_clause = ""
        if source_filters:
            placeholders = ",".join(["?"] * len(source_filters))
            source_clause = f" AND source IN ({placeholders})"
            params.extend(source_filters)

        sql = f"SELECT id, path, source, content FROM chunks WHERE ({conditions}){source_clause} LIMIT {limit * 4}"
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:
            logger.warning(f"LIKE fallback failed: {e}")
            return {}

        results = {}
        for row in rows:
            # Simple scoring: count how many tokens match
            content_lower = row["content"].lower()
            matches = sum(1 for t in tokens if t in content_lower)
            text_score = matches / len(tokens) * 0.5  # max 0.5 for LIKE
            results[row["id"]] = HybridSearchResult(
                chunk_id=row["id"],
                path=row["path"],
                result_source=row["source"],
                snippet=row["content"],
                start_line=0, end_line=0,
                vector_score=0.0, text_score=text_score
            )
        return results

    async def search(
        self, 
        query: str, 
        vector_weight: float = 0.7, 
        text_weight: float = 0.3,
        min_score: float = 0.1,
        max_results: int = 10,
        source_filters: List[str] = None
    ) -> List[HybridSearchResult]:
        try:
            vector_weight = float(vector_weight)
            text_weight = float(text_weight)
            min_score = float(min_score)
            max_results = int(max_results)
        except (ValueError, TypeError):
            pass
        source_filters = source_filters or ["memory", "sessions", "wiki"]
        
        # 1. Try Redis cache first
        cached_results = await self.redis_cache.get_search_results(query, source_filters)
        if cached_results is not None:
            filtered = []
            for res in cached_results:
                res.score = (res.vector_score * vector_weight) + (res.text_score * text_weight)
                if res.score >= min_score:
                    filtered.append(res)
            return sorted(filtered, key=lambda x: x.score, reverse=True)[:max_results]
        
        # 2. Cache miss -> Full search
        query_vec = await self.embeddings.embed_query(query)
        
        vector_results = await self.search_vector(query_vec, max_results, source_filters)
        keyword_results = await self.search_keyword(query, max_results, source_filters)
        
        # Fallback: if FTS5 returned nothing, try LIKE search
        if not keyword_results:
            keyword_results = await self.search_like_fallback(query, max_results, source_filters)
        
        merged: Dict[str, HybridSearchResult] = {}
        
        for k, v in vector_results.items():
            merged[k] = v
            
        for k, v in keyword_results.items():
            if k in merged:
                merged[k].text_score = v.text_score
                if not merged[k].snippet and v.snippet:
                    merged[k].snippet = v.snippet
            else:
                merged[k] = v
                
        final_results = []
        for res in merged.values():
            res.score = (res.vector_score * vector_weight) + (res.text_score * text_weight)
            if res.score >= min_score:
                final_results.append(res)
                
        final_results.sort(key=lambda x: x.score, reverse=True)
        top_results = final_results[:max_results]
        
        # 3. Save into cache
        await self.redis_cache.set_search_results(query, source_filters, top_results)
        
        return top_results
