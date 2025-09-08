from supabase_client import supabase
from fastapi import HTTPException
import json
from datetime import datetime
from typing import List, Dict, Any


def safe_json_load(value):
    """Safely parse JSON string"""
    try:
        if not value or value.lower() in ["none", "null", ""]:
            return {}
        if isinstance(value, dict):
            return value
        return json.loads(value)
    except Exception:
        return {}


def fetch_agent_metadata(agent_id: str) -> Dict[str, Any]:
    """Fetch agent basic metadata"""
    try:
        result = supabase.table("s_agent_basic_metadata").select("*").eq("id", agent_id).single().execute()
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Agent with id {agent_id} not found")
        
        agent_data = result.data
        tools = agent_data.get("tools", [])
        if isinstance(tools, str):
            tools = safe_json_load(tools)
        
        return {
            "id": agent_data["id"],
            "created_at": agent_data["created_at"],
            "role": agent_data["role"],
            "goal": agent_data["goal"],
            "backstory": agent_data["backstory"],
            "tools": tools,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching agent metadata: {str(e)}")


def fetch_tools_metadata(tool_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch tools metadata from api_metadata table"""
    if not tool_ids:
        return []
    
    try:
        result = supabase.table("api_metadata").select("*").in_("id", tool_ids).execute()
        return result.data or []
    except Exception as e:
        print(f"Warning: Could not fetch tools metadata: {str(e)}")
        return []


def fetch_user_variables(agent_id: str) -> Dict[str, str]:
    """Fetch user variables for agent"""
    try:
        result = supabase.table("s_uservariables_agent").select("id, name").eq("agent_id", agent_id).execute()
        if not result.data:
            return {}
        
        variables = {}
        for var in result.data:
            var_result = supabase.table("s_uservariables_values").select("value").eq("variable_id", var["id"]).single().execute()
            if var_result.data:
                variables[var["name"]] = var_result.data["value"]
        
        return variables
    except Exception as e:
        print(f"Warning: Could not fetch user variables: {str(e)}")
        return {}


def fetch_other_variables(agent_id: str) -> List[Dict[str, Any]]:
    """Fetch other variables for agent"""
    try:
        result = supabase.table("s_othervariables").select("*").eq("agent_id", agent_id).execute()
        return result.data or []
    except Exception as e:
        print(f"Warning: Could not fetch other variables: {str(e)}")
        return []


def fetch_agent_configs() -> Dict[str, Any]:
    """Fetch agent configuration from s_agent_configs"""
    try:
        config_id = "dffeb172-175b-4ffb-bae1-17d0750167c1"
        result = supabase.table("s_agent_configs").select("*").eq("id", config_id).single().execute()
        
        if not result.data:
            return {
                "llm": "groq/llama-3.3-70b-versatile",
                "function_calling_llm": None,
                "max_iter": 20,
                "max_rpm": None,
                "max_execution_time": None,
                "verbose": False,
                "allow_delegation": False,
                "step_callback": None,
                "cache": True,
                "system_template": None,
                "prompt_template": None,
                "response_template": None,
                "allow_code_execution": False,
                "max_retry_limit": 2,
                "respect_context_window": True,
                "code_execution_mode": "safe",
                "multimodal": False,
                "inject_date": False,
                "date_format": "%Y-%m-%d",
                "reasoning": False,
                "max_reasoning_attempts": None,
                "embedder": None,
                "knowledge_sources": None,
                "user_system_prompt": None
            }
        
        return result.data
    except Exception as e:
        print(f"Warning: Could not fetch agent configs: {str(e)}")
        return {
            "llm": "groq/llama-3.3-70b-versatile",
            "function_calling_llm": None,
            "max_iter": 20,
            "max_rpm": None,
            "max_execution_time": None,
            "verbose": False,
            "allow_delegation": False,
            "step_callback": None,
            "cache": True,
            "system_template": None,
            "prompt_template": None,
            "response_template": None,
            "allow_code_execution": False,
            "max_retry_limit": 2,
            "respect_context_window": True,
            "code_execution_mode": "safe",
            "multimodal": False,
            "inject_date": False,
            "date_format": "%Y-%m-%d",
            "reasoning": False,
            "max_reasoning_attempts": None,
            "embedder": None,
            "knowledge_sources": None,
            "user_system_prompt": None
        }


def fetch_agent_chat_history(task_id: str) -> List[Dict[str, str]]:
    """Fetch agent chat history for a task"""
    try:
        result = (
            supabase.table("s_agentchats")
            .select("agent_prompt, response")
            .eq("task_id", task_id)
            .order("created_at", desc=False)
            .execute()
        )
        
        messages = []
        if result.data:
            for chat in result.data:
                if chat.get("agent_prompt"):
                    messages.append({"role": "user", "content": chat["agent_prompt"]})
                if chat.get("response"):
                    messages.append({"role": "assistant", "content": chat["response"]})
        
        return messages
    except Exception as e:
        print(f"Warning: Could not fetch agent chat history: {str(e)}")
        return []


def insert_agent_prompt(task_id: str, agent_id: str, agent_prompt: str) -> str:
    """Insert agent prompt and return the chat ID"""
    try:
        result = supabase.table("s_agentchats").insert({
            "task_id": task_id,
            "agent_id": agent_id,
            "agent_prompt": agent_prompt
        }).execute()
        return result.data[0]["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting agent prompt: {str(e)}")


def update_agent_response(chat_id: str, response: str):
    """Update agent response in the same row"""
    try:
        supabase.table("s_agentchats").update({"response": response}).eq("id", chat_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating agent response: {str(e)}")