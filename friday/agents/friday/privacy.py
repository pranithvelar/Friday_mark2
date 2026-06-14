import re
from typing import Dict, Any, List

class PrivacyManager:
    """
    Handles session visibility bounds. 
    Prevents highly sensitive data streams (marked private) from being
    indexed by the vector store or included in the dream narrative.
    """
    def __init__(self, private_prefixes: List[str] = ["private_", "secret_", "incognito_"]):
        self.private_prefixes = private_prefixes

    def is_session_visible(self, session_id: str) -> bool:
        """
        Determine if a session should be tracked by memory.
        """
        for prefix in self.private_prefixes:
            if session_id.startswith(prefix):
                return False
        return True

    def redact_log_path(self, path: str) -> str:
        """
        Redacts identifiable filenames for system logs.
        """
        base = path.split("/")[-1]
        for prefix in self.private_prefixes:
            if base.startswith(prefix):
                return f"[REDACTED_SESSION_{prefix.strip('_').upper()}]"
        return path
