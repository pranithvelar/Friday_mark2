import json
import logging
from typing import List, Dict, Any
from friday.cache.redis_client import RedisManager

logger = logging.getLogger(__name__)

class RedisEmbeddingCache:
    def __init__(self):
        self.redis_mgr = RedisManager()
        
    async def get_embeddings(self, provider: str, model: str, hashes: List[str]) -> Dict[str, List[float]]:
        client = await self.redis_mgr.get_client()
        if not client or not hashes:
            return {}
            
        keys = [f"emb:{provider}:{model}:{h}" for h in hashes]
        try:
            result_json = await client.mget(keys)
            result = {}
            for h, r_json in zip(hashes, result_json):
                if r_json:
                    try:
                        result[h] = json.loads(r_json)
                    except json.JSONDecodeError:
                        pass
            return result
        except Exception as e:
            logger.debug(f"Failed to get embeddings from Redis: {e}")
            return {}

    async def set_embeddings(self, provider: str, model: str, entries: List[Dict[str, Any]]):
        client = await self.redis_mgr.get_client()
        if not client or not entries:
            return
            
        ttl = 604800  # 7 days
        try:
            pipeline = client.pipeline()
            for entry in entries:
                key = f"emb:{provider}:{model}:{entry['hash']}"
                val_json = json.dumps(entry["embedding"])
                pipeline.setex(key, ttl, val_json)
            await pipeline.execute()
        except Exception as e:
            logger.debug(f"Failed to set embeddings in Redis: {e}")
