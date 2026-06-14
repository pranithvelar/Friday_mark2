import logging
from typing import Optional
import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError
from friday.config.settings import IntelligentMemoryConfig

logger = logging.getLogger(__name__)

class RedisManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
            
        self.config = IntelligentMemoryConfig.load()
        self.enabled = getattr(self.config, 'redis_enabled', False)
        self.client: Optional['redis.Redis'] = None
        self._initialized = True

    async def connect(self) -> bool:
        if not self.enabled:
            return False

        if self.client is not None:
            return True

        try:
            host = getattr(self.config, 'redis_host', '127.0.0.1')
            if host == 'localhost':
                host = '127.0.0.1'
                
            port = getattr(self.config, 'redis_port', 6379)
            
            self.client = redis.Redis(
                host=host, 
                port=port, 
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0
            )
            
            await self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis. Falling back to native system: {e}")
            self.client = None
            self.enabled = False
            return False

    async def get_client(self) -> Optional['redis.Redis']:
        if not self.enabled:
            return None
        if self.client is None:
            success = await self.connect()
            if not success:
                return None
        return self.client
