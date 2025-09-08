from crewai import Agent, LLM
from crewai.tools import BaseTool
from typing import Dict, List, Any
import requests
import json
import os
import re
from dotenv import load_dotenv
from data_service import (
    fetch_agent_metadata, 
    fetch_tools_metadata, 
    safe_json_load, 
    fetch_agent_configs,
    fetch_user_variables,
    fetch_other_variables
)

load_dotenv()


class DynamicAPICallTool(BaseTool):
    name: str
    description: str
    endpoint_url: str
    http_method: str
    headers: Dict[str, Any] = None
    query_params: Dict[str, Any] = None
    body: Dict[str, Any] = None
    user_variables: Dict[str, str] = None
    other_variables: List[Dict[str, Any]] = None

    def _run(self, **kwargs) -> str:
        try:
            headers = self._resolve_placeholders(self.headers or {})
            params = self._resolve_placeholders(self.query_params or {})
            data = self._resolve_placeholders(self.body or {})

            method = self.http_method.upper()

            response = requests.request(
                method=method,
                url=self.endpoint_url,
                headers=headers,
                params=params,
                json=data if method in ["POST", "PUT", "PATCH"] else None,
                data=data if method not in ["POST", "PUT", "PATCH"] else None
            )

            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"API call failed: {str(e)}"

    def _resolve_placeholders(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve placeholders in API parameters"""
        if not isinstance(data, dict):
            return data
        
        resolved_data = {}
        for key, value in data.items():
            if isinstance(value, str) and '{' in value and '}' in value:
                resolved_data[key] = self._replace_variables(value)
            elif isinstance(value, dict):
                resolved_data[key] = self._resolve_placeholders(value)
            else:
                resolved_data[key] = value
        
        return resolved_data

    def _replace_variables(self, text: str) -> str:
        """Replace variables in text using available contexts"""
        def replacer(match):
            var_name = match.group(1)
            
            # First check user variables
            if self.user_variables and var_name in self.user_variables:
                return self.user_variables[var_name]
            
            # Then check other variables - this requires agent decision
            # For now, return the placeholder as is - agent will handle it
            return match.group(0)
        
        return re.sub(r'\{(\w+)\}', replacer, text)


def build_variable_context(other_variables: List[Dict[str, Any]]) -> str:
    """Build context string for other variables"""
    if not other_variables:
        return ""
    
    context = "\n\nDYNAMIC VARIABLES CONTEXT:\n"
    context += "When making API calls, you may encounter placeholders that need dynamic values. Here are the available variables:\n\n"
    
    for var in other_variables:
        context += f"Variable: {var['name']}\n"
        context += f"Description: {var['description']}\n"
        context += f"Data Type: {var['data_type']}\n"
        
        variables_data = safe_json_load(var.get('variables', {}))
        if variables_data:
            context += f"Available options: {json.dumps(variables_data, indent=2)}\n"
        context += "\n"
    
    context += "Choose the most appropriate value based on the user's request and the variable description.\n"
    return context


def build_tools_from_metadata(tool_data_list: List[Dict[str, Any]], user_variables: Dict[str, str], other_variables: List[Dict[str, Any]]) -> List[BaseTool]:
    """Build tools with dynamic variable support"""
    tools = []
    for tool_data in tool_data_list:
        headers = safe_json_load(tool_data.get("headers"))
        params = safe_json_load(tool_data.get("query_params"))
        body = safe_json_load(tool_data.get("body"))

        tool = DynamicAPICallTool(
            name=tool_data["name"],
            description=tool_data.get("tool_description", "No description"),
            endpoint_url=tool_data["endpoint_url"],
            http_method=tool_data["http_method"],
            headers=headers,
            query_params=params,
            body=body,
            user_variables=user_variables,
            other_variables=other_variables
        )
        tools.append(tool)
    return tools


def build_agent_from_metadata(agent_id: str) -> Agent:
    """Build agent with dynamic variable support"""
    agent_data = fetch_agent_metadata(agent_id)
    user_variables = fetch_user_variables(agent_id)
    other_variables = fetch_other_variables(agent_id)
    tool_data_list = fetch_tools_metadata(agent_data["tools"])
    tools = build_tools_from_metadata(tool_data_list, user_variables, other_variables)
    agent_config = fetch_agent_configs()

    # Enhance backstory with variable context
    variable_context = build_variable_context(other_variables)
    enhanced_backstory = agent_data["backstory"] + variable_context

    llm = LLM(model=agent_config["llm"]) if agent_config.get("llm") else None
    function_calling_llm_config = LLM(model=agent_config["function_calling_llm"]) if agent_config.get("function_calling_llm") else None
    
    agent = Agent(
        config={
            "role": agent_data["role"],
            "goal": agent_data["goal"],
            "backstory": enhanced_backstory,
            "tools": tools,
            "llm": llm,
            "function_calling_llm": function_calling_llm_config,
            "verbose": agent_config.get("verbose", False),
            "allow_delegation": agent_config.get("allow_delegation", False),
            "max_iter": agent_config.get("max_iter", 20),
            "max_rpm": agent_config.get("max_rpm"),
            "max_execution_time": agent_config.get("max_execution_time"),
            "max_retry_limit": agent_config.get("max_retry_limit", 2),
            "allow_code_execution": agent_config.get("allow_code_execution", False),
            "code_execution_mode": agent_config.get("code_execution_mode", "safe"),
            "respect_context_window": agent_config.get("respect_context_window", True),
            "multimodal": agent_config.get("multimodal", False),
            "inject_date": agent_config.get("inject_date", False),
            "date_format": agent_config.get("date_format", "%Y-%m-%d"),
            "reasoning": agent_config.get("reasoning", False),
            "max_reasoning_attempts": agent_config.get("max_reasoning_attempts"),
            "embedder": agent_config.get("embedder"),
            "knowledge_sources": agent_config.get("knowledge_sources"),
            "system_template": agent_config.get("system_template"),
            "prompt_template": agent_config.get("prompt_template"),
            "response_template": agent_config.get("response_template"),
            "step_callback": agent_config.get("step_callback"),
            "cache": agent_config.get("cache", True),
            "use_system_prompt": agent_config.get("use_system_prompt", True),
        }
    )
    return agent