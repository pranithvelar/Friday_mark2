import os
import json
from dataclasses import dataclass, field

# ── .env loader (no extra dependencies) ──────────────────────────────────────
def _load_dotenv(path: str = ".env"):
    """
    Minimal .env loader. Reads KEY=VALUE lines and sets them in os.environ
    if they are not already set. This means actual env vars always win.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

# Load .env from the project root (friday_mark2/) before anything reads os.environ
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))


@dataclass
class IntelligentMemoryConfig:
    workspace_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "workspace"))
    llama_model: str = field(default_factory=lambda: os.getenv("LLM_OLLAMA_MODEL", "llama3.1:8b"))
    embedding_model: str = "nomic-embed-text"
    rest_port: int = 8000
    log_level: str = "INFO"

    # Internal Memory parameters
    promotion_limit: int = 5
    promotion_min_score: float = 0.5
    dreaming_enabled: bool = True

    # Optional Redis Config
    redis_enabled: bool = True
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379

    # ── LLM Slot Config ───────────────────────────────────────────────────────
    # Slot 1 (API): Any OpenAI-compatible provider.
    # Set these via .env file in the project root, e.g.:
    #
    #   LLM_API_KEY=sk-or-v1-...
    #   LLM_API_BASE_URL=https://openrouter.ai/api/v1
    #   LLM_API_MODEL=anthropic/claude-3-5-sonnet
    #
    # Provider examples:
    #   OpenRouter:  LLM_API_BASE_URL=https://openrouter.ai/api/v1
    #   Gemini:      LLM_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    #   DeepSeek:    LLM_API_BASE_URL=https://api.deepseek.com/v1
    #   Groq:        LLM_API_BASE_URL=https://api.groq.com/openai/v1
    #   OpenAI:      LLM_API_BASE_URL=  (leave blank)
    #
    # If LLM_API_KEY is empty or unset, Slot 1 is disabled and Ollama is used directly.
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_api_base_url: str = field(default_factory=lambda: os.getenv("LLM_API_BASE_URL", ""))
    llm_api_model: str = field(default_factory=lambda: os.getenv("LLM_API_MODEL", "gpt-4o-mini"))
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, config_path: str = None) -> "IntelligentMemoryConfig":
        if not config_path:
            config_path = os.environ.get("MEMORY_SYSTEM_CONFIG", "config.json")

        if not os.path.exists(config_path):
            return cls()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Env vars override config.json (already loaded by _load_dotenv above)
            instance = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            return instance
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
            return cls()

