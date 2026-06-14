"""
Friday Mark 2 — Terminal Entry Point
=====================================
Run this file to start Friday in terminal mode.

Usage:
    python terminal_chat.py
"""
import sys
import os

# Add the friday_mark2 root to Python path so `import friday.*` works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Delegate to the actual interface
from friday.interfaces.terminal.chat import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
