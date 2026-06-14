"""
Friday — Intelligent Personal AI Assistant
==========================================
Root package. Exports the main SmartRouter for use by any interface.

Usage (from any interface):
    from friday.router.smart_router import build_smart_router
    router = build_smart_router(...)
    result = await router.route(user_message, session_id)
    print(result["text"])
"""

__version__ = "2.0.0"
__codename__ = "Mark 2"
