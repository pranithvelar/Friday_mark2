import re

class SafetyFilter:
    """
    Basic heuristic safety filter to sanitize inputs before memory ingestion
    or LLM consolidation.
    """
    def __init__(self):
        # Basic heuristic regexes for demonstration of Component 18 logic
        self.pii_patterns = [
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN REDACTED]"), # SSN
            (re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|[25][1-7][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b'), "[CC REDACTED]"), # Credit Cards
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "[EMAIL REDACTED]") # Email
        ]
        
    def sanitize_content(self, content: str) -> str:
        """
        Redact recognizable PII before saving to memory or sending to LLMs.
        """
        sanitized = content
        for pattern, replacement in self.pii_patterns:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized
