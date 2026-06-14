import asyncio
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

class BackgroundScheduler:
    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.config = config
        self.watchers = []

    def register_watcher(self, watcher: Any):
        self.watchers.append(watcher)

    async def run(self):
        """Runs the background workers continuously."""
        # Initial wait to let system settle
        await asyncio.sleep(2)
        
        while True:
            for watcher in self.watchers:
                try:
                    alert = await watcher.check(self.db_manager, self.config)
                    if alert:
                        # Print clearly with Friday branding
                        print(f"\n\r[Friday - Background Alert] {alert}\n")
                        # Reprint the "You: " prompt to keep it clean if user is idle
                        print("You: ", end="", flush=True)
                except Exception as e:
                    logger.error(f"Watcher {watcher.__class__.__name__} failed: {e}")
            
            # Sleep for 1 minute before checking again. 
            # 1-minute resolution catches exact 1h / 15m marks precisely without heavy polling.
            await asyncio.sleep(60)
