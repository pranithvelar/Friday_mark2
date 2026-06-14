import asyncio
import hashlib
import time
import logging
import json
from typing import List, Dict, Any, Optional
import ollama
from friday.cache.embedding_cache import RedisEmbeddingCache

logger = logging.getLogger(__name__)

class EmbeddingProviderError(Exception):
    pass

class EmbeddingManager:
    def __init__(
        self,
        db_manager,
        model: str = "nomic-embed-text",
        dimension: int = 768,
        batch_size: int = 32,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        cache_enabled: bool = True
    ):
        self.db = db_manager
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.cache_enabled = cache_enabled
        self.provider_id = "ollama"
        self.redis_cache = RedisEmbeddingCache()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _embed_single_with_retry(self, text: str) -> List[float]:
        for attempt in range(self.max_retries):
            try:
                # Wrap the synchronous ollama call in asyncio.wait_for via an executor
                # Using the async ollama client
                client = ollama.AsyncClient()
                response = await asyncio.wait_for(
                    client.embeddings(model=self.model, prompt=text),
                    timeout=self.timeout_seconds
                )
                embedding = response.get("embedding", [])
                
                if len(embedding) != self.dimension:
                    raise EmbeddingProviderError(f"Dimension mismatch. Expected {self.dimension}, got {len(embedding)}")
                
                return embedding

            except asyncio.TimeoutError as e:
                logger.warning(f"Embedding timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise EmbeddingProviderError("Max retries exceeded due to timeout") from e
            except Exception as e:
                logger.warning(f"Embedding error: {e} on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise EmbeddingProviderError(f"Max retries exceeded: {str(e)}") from e
            
            # Exponential backoff
            await asyncio.sleep(self.base_delay * (2 ** attempt))
            
        raise EmbeddingProviderError("Failed to embed text")

    def _get_cached_embeddings(self, hashes: List[str]) -> Dict[str, List[float]]:
        if not self.cache_enabled or not hashes:
            return {}
            
        placeholders = ",".join(["?"] * len(hashes))
        query = f"""
            SELECT hash, embedding FROM embedding_cache
            WHERE provider = ? AND model = ? AND hash IN ({placeholders})
        """
        conn = self.db.get_connection()
        rows = conn.execute(query, [self.provider_id, self.model] + hashes).fetchall()
        
        result = {}
        for row in rows:
            try:
                result[row["hash"]] = json.loads(row["embedding"])
            except Exception as e:
                logger.warning(f"Failed to parse cached embedding for {row['hash']}: {e}")
                
        return result

    def _cache_embeddings(self, entries: List[Dict[str, Any]]):
        if not self.cache_enabled or not entries:
            return
            
        conn = self.db.get_connection()
        try:
            now = int(time.time() * 1000)
            cursor = conn.cursor()
            for entry in entries:
                cursor.execute(
                    """
                    INSERT INTO embedding_cache (provider, model, provider_key, hash, embedding, dims, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, model, provider_key, hash) DO UPDATE SET
                        embedding=excluded.embedding,
                        dims=excluded.dims,
                        updated_at=excluded.updated_at
                    """,
                    (
                        self.provider_id,
                        self.model,
                        "default_key",
                        entry["hash"],
                        json.dumps(entry["embedding"]),
                        len(entry["embedding"]),
                        now
                    )
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to cache embeddings: {e}")
            conn.rollback()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        hashes = [self._hash_text(t) for t in texts]
        
        # 1. Try Redis
        cached_redis = await self.redis_cache.get_embeddings(self.provider_id, self.model, hashes)
        
        # 2. Try SQLite for ones missing from Redis
        missing_from_redis = [h for h in hashes if h not in cached_redis]
        cached_sqlite = self._get_cached_embeddings(missing_from_redis)
        
        final_embeddings = [None] * len(texts)
        missing_indices = []
        missing_texts = []
        missing_hashes = []

        for i, (text, text_hash) in enumerate(zip(texts, hashes)):
            if text_hash in cached_redis:
                final_embeddings[i] = cached_redis[text_hash]
            elif text_hash in cached_sqlite:
                final_embeddings[i] = cached_sqlite[text_hash]
            else:
                missing_indices.append(i)
                missing_texts.append(text)
                missing_hashes.append(text_hash)

        if not missing_texts:
            return final_embeddings

        # Process missing texts in batches
        new_cache_entries = []
        
        for i in range(0, len(missing_texts), self.batch_size):
            batch_texts = missing_texts[i:i + self.batch_size]
            batch_hashes = missing_hashes[i:i + self.batch_size]
            batch_indices = missing_indices[i:i + self.batch_size]
            
            # For simplicity, using sequential async calls for batch execution, 
            # ideally Ollama AsyncClient would support batch generation or we use asyncio.gather
            # Since Ollama standard API expects loop, we'll gather them.
            tasks = [self._embed_single_with_retry(text) for text in batch_texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to embed text '{batch_texts[j][:50]}...': {result}")
                    final_embeddings[batch_indices[j]] = [] # Fallback
                else:
                    final_embeddings[batch_indices[j]] = result
                    new_cache_entries.append({
                        "hash": batch_hashes[j],
                        "embedding": result
                    })

        # Cache the new ones in SQLite
        self._cache_embeddings(new_cache_entries)
        
        # Sync newly fetched and SQLite-retrieved ones up to Redis
        to_redis = new_cache_entries.copy()
        for h, emb in cached_sqlite.items():
            to_redis.append({"hash": h, "embedding": emb})
            
        await self.redis_cache.set_embeddings(self.provider_id, self.model, to_redis)
        
        # Ensure fallback to empty list instead of None
        return [emb if emb is not None else [] for emb in final_embeddings]

    async def embed_query(self, text: str) -> List[float]:
        # Do not cache individual quick queries.
        return await self._embed_single_with_retry(text)
