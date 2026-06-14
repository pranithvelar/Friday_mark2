"""
src/tools/mcp/__init__.py
==========================
MCP (Model Context Protocol) client tools (future integration).

Will contain:
  - list_servers() → list of available MCP servers
  - call_tool(server: str, tool: str, args: dict) → result
  - list_resources(server: str) → list of resources
  - read_resource(server: str, uri: str) → content

Tool category string: "mcp"

Integration steps when ready:
  1. Set up MCP server configuration.
  2. Implement each function below using the MCP Python SDK.
  3. Register tools in terminal_chat.py.
  4. In execution_engine.py, dispatch "mcp" category here.
"""
