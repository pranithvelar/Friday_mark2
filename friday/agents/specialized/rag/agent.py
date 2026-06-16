import logging
from typing import Dict, Any

from friday.agents.base_agent import BaseAgent
from friday.agents.specialized.rag.vector_system.embedding_manager import EmbeddingManager
from friday.agents.specialized.rag.vector_system.hybrid_search import HybridSearcher

logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    """
    RAG Agent (Retrieval-Augmented Generation)
    Handles deep vector search and document analysis using embeddings.
    This was removed from the core hot-path to ensure the main Friday loop is instantaneous.
    This agent is spawned ONLY when deep semantic search over massive documents is explicitly needed.
    """
    agent_id: str = "rag_specialist"
    description: str = "Handles document embedding, semantic vector search, and complex RAG queries."
    capabilities: list = ["vector_search", "document_embedding", "semantic_retrieval"]
    tool_scope: list = ["embed_document", "search_vectors"]

    def __init__(self, db_manager, llm_provider=None):
        self.db = db_manager
        # Ensure vector tables exist (only when RAG agent is active)
        # Note: In the future, this should be a dedicated RAG database to keep Friday's core DB lean.
        try:
            from friday.memory.db_manager import _load_sqlite_vec
            _load_sqlite_vec(self.db.get_connection())
            self._ensure_rag_tables()
        except Exception as e:
            logger.warning(f"Failed to initialize sqlite-vec for RAG Agent: {e}")

        # Vector system is initialized ONLY here
        self.embedder = EmbeddingManager(db_manager, model="nomic-embed-text", dimension=768)
        self.searcher = HybridSearcher(db_manager, self.embedder)

    def _ensure_rag_tables(self):
        # Dedicated RAG vector tables for future use
        pass

    async def run(self, task: str, context: dict) -> dict:
        """
        Execute a RAG query.
        """
        logger.info(f"[RAGAgent] Executing deep vector search for task: {task[:50]}")
        # Implementation for RAG tasks will go here when integrated.
        return {
            "status": "success",
            "message": "RAG agent vector operations are standing by."
        }
