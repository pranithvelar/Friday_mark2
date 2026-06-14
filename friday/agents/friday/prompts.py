import json
from typing import Dict, Any

class PromptBuilder:
    def __init__(self, agent_name: str = "Assistant", agent_role: str = "A helpful AI assistant that manages an intelligent memory system."):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.base_prompt = ""
        self.personalization_context = ""
        self.tools_schema_str = ""
        
    def add_personalization_context(self, context: str):
        self.personalization_context = context
        
    def add_tools_schema(self, tools_schemas: list[Dict[str, Any]]):
        self.tools_schema_str = json.dumps(tools_schemas, indent=2)
        
    def build(self) -> str:
        prompt_parts = [
            f"You are {self.agent_name}. {self.agent_role}",
            "\nYou run in a loop of Thought, Action, Result, and Response.",
            "Use Thought to describe your thoughts.",
            "If you need to use a tool, use Action to specify the tool to call.",
            "The Action should be formatted as a JSON object, e.g.:",
            "```json\n{\n  \"name\": \"tool_name\",\n  \"arguments\": {\n    \"arg1\": \"value1\"\n  }\n}\n```",
            "If you use an Action, STOP generating. The system will run the tool and provide a Result.",
            "If you have the Result or do not need a tool, provide your Response.\n"
        ]
        
        if self.personalization_context:
            prompt_parts.append("--- USER CONTEXT ---")
            prompt_parts.append(self.personalization_context)
            prompt_parts.append("--------------------\n")
            
        if self.tools_schema_str:
            prompt_parts.append("Available tools:")
            prompt_parts.append(self.tools_schema_str)
            
        return "\n".join(prompt_parts)
