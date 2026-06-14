import json
import hashlib
import logging
import json
import hashlib
import logging
from typing import List, Optional
from friday.cache.redis_client import RedisManager

logger = logging.getLogger(__name__)

class RedisSearchCache:
    def __init__(self):
        self.redis_mgr = RedisManager()
        
    def _hash_sources(self, sources: Optional[List[str]]) -> str:
        if not sources:
            return "all"
        return hashlib.md5(",".join(sorted(sources)).encode('utf-8')).hexdigest()[:8]

    def _hash_query(self, query: str) -> str:
        return hashlib.sha1(query.strip().encode('utf-8')).hexdigest()[:12]

    async def get_search_results(self, query: str, sources: Optional[List[str]]) -> Optional[list]:
        from friday.search.hybrid_search import HybridSearchResult
        client = await self.redis_mgr.get_client()
        if not client:
            return None
            
        key = f"search:{self._hash_query(query)}:{self._hash_sources(sources)}"
        try:
            cached_json = await client.get(key)
            if not cached_json:
                return None
                
            data = json.loads(cached_json)
            results = []
            for item in data:
                res = HybridSearchResult(
                    chunk_id=item["chunk_id"],
                    path=item["path"],
                    result_source=item["result_source"],
                    snippet=item["snippet"],
                    start_line=item["start_line"],
                    end_line=item["end_line"],
                    vector_score=item["vector_score"],
                    text_score=item["text_score"]
                )
                res.score = item["score"]
                results.append(res)
            return results
        except Exception as e:
            logger.debug(f"Failed to get search cache: {e}")
            return None

    async def set_search_results(self, query: str, sources: Optional[List[str]], results: list):
        client = await self.redis_mgr.get_client()
        if not client:
            return
            
        key = f"search:{self._hash_query(query)}:{self._hash_sources(sources)}"
        ttl = 300  # 5 minutes
        try:
            data = [
                {
                    "chunk_id": r.chunk_id,
                    "path": r.path,
                    "result_source": r.result_source,
                    "snippet": r.snippet,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "vector_score": r.vector_score,
                    "text_score": r.text_score,
                    "score": r.score
                }
                for r in results
            ]
            await client.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.debug(f"Failed to set search cache: {e}")
