from crewai import Agent, LLM
from crewai.tools import BaseTool
from typing import Dict, List, Any
import requests
import json
import os
import re
import uuid
from dotenv import load_dotenv
from data_service import (
    fetch_agent_metadata, 
    fetch_tools_metadata, 
    safe_json_load, 
    fetch_agent_configs,
    fetch_user_variables,
    fetch_other_variables,
    resolve_final_tool_ids,
    fetch_task_instructions
)

load_dotenv()


class DynamicAPICallTool(BaseTool):
    """Fresh API tool instance created per execution"""
    name: str
    description: str
    endpoint_url: str
    http_method: str
    headers: Dict[str, Any] = None
    query_params: Dict[str, Any] = None
    body: Dict[str, Any] = None
    user_variables: Dict[str, str] = None
    other_variables: List[Dict[str, Any]] = None

    def __init__(self, **data):
        """Initialize with unique instance ID"""
        super().__init__(**data)
        self._instance_id = str(uuid.uuid4())[:8]
        # Enhance description with variable info
        self._enhance_description()

    def _enhance_description(self):
        """Add variable information to tool description"""
        if self.other_variables:
            var_info = "\n\nDYNAMIC VARIABLES: When using this tool, you may need to resolve these variables:\n"
            for var in self.other_variables:
                var_data = safe_json_load(var.get('variables', {}))
                if var_data and not var_data.get('tool_dependency'):
                    # Only show predefined options in tool description
                    options = list(var_data.keys())
                    var_info += f"- {var['name']}: Choose from {options} based on user request\n"
            
            if var_info != "\n\nDYNAMIC VARIABLES: When using this tool, you may need to resolve these variables:\n":
                self.description += var_info

    def _run(self, **kwargs) -> str:
        try:
            # Get variable mappings from kwargs if provided
            variable_mappings = kwargs.pop('variable_mappings', {})
            
            headers = self._resolve_placeholders(self.headers or {}, variable_mappings)
            params = self._resolve_placeholders(self.query_params or {}, variable_mappings)
            data = self._resolve_placeholders(self.body or {}, variable_mappings)

            method = self.http_method.upper()

            response = requests.request(
                method=method,
                url=self.endpoint_url,
                headers=headers,
                params=params,
                json=data if method in ["POST", "PUT", "PATCH"] else None,
                data=data if method not in ["POST", "PUT", "PATCH"] else None,
                timeout=30
            )

            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"API call failed: {str(e)}"

    def _resolve_placeholders(self, data: Dict[str, Any], variable_mappings: Dict[str, str] = None) -> Dict[str, Any]:
        """Resolve placeholders in API parameters"""
        if not isinstance(data, dict):
            return data
        
        variable_mappings = variable_mappings or {}
        resolved_data = {}
        
        for key, value in data.items():
            if isinstance(value, str) and '{' in value and '}' in value:
                resolved_data[key] = self._replace_variables(value, variable_mappings)
            elif isinstance(value, dict):
                resolved_data[key] = self._resolve_placeholders(value, variable_mappings)
            else:
                resolved_data[key] = value
        
        return resolved_data

    def _replace_variables(self, text: str, variable_mappings: Dict[str, str] = None) -> str:
        """Replace variables in text using available contexts"""
        variable_mappings = variable_mappings or {}
        
        def replacer(match):
            var_name = match.group(1)
            
            # First check agent-provided mappings (for other variables)
            if variable_mappings and var_name in variable_mappings:
                return variable_mappings[var_name]
            
            # Then check user variables (for simple replacements)
            if self.user_variables and var_name in self.user_variables:
                return self.user_variables[var_name]
            
            # If no mapping found, return placeholder (will likely cause API error)
            return match.group(0)
        
        return re.sub(r'\{(\w+)\}', replacer, text)


def build_variable_context(other_variables: List[Dict[str, Any]]) -> str:
    """Build context string for other variables"""
    if not other_variables:
        return ""
    
    context = "\n\nDYNAMIC VARIABLES CONTEXT:\n"
    context += "When making API calls, you may encounter placeholders that need dynamic values. There are multiple types:\n\n"
    
    context += "IMPORTANT: When calling tools with dynamic variables, pass them using variable_mappings parameter.\n"
    context += "Example: tool_name(variable_mappings={'city_id': 'BLR', 'quantity': '5'})\n\n"
    
    # Group variables by type for better organization
    contextual_vars = []
    direct_input_vars = []
    tool_dependency_vars = []
    
    for var in other_variables:
        var_type = var.get('variable_type', 'contextual')  # Default to contextual for backward compatibility
        if var_type == 'contextual':
            contextual_vars.append(var)
        elif var_type == 'direct_input':
            direct_input_vars.append(var)
        elif var_type == 'tool_dependency':
            tool_dependency_vars.append(var)
        else:
            # Fallback: try to detect type from variables content
            variables_data = safe_json_load(var.get('variables', {}))
            if variables_data.get('tool_dependency'):
                tool_dependency_vars.append(var)
            elif variables_data.get('extraction_hint'):
                direct_input_vars.append(var)
            else:
                contextual_vars.append(var)
    
    # TYPE 2: Contextual Variables (Predefined Options)
    if contextual_vars:
        context += "TYPE 2 - CONTEXTUAL VARIABLES (Choose from predefined options):\n"
        for var in contextual_vars:
            context += f"Variable: {var['name']}\n"
            context += f"Description: {var['description']}\n"
            context += f"Data Type: {var['data_type']}\n"
            
            variables_data = safe_json_load(var.get('variables', {}))
            if variables_data and not variables_data.get('tool_dependency') and not variables_data.get('extraction_hint'):
                options_str = ", ".join([f"'{k}' (for {v})" for k, v in variables_data.items() if k not in ['tool_dependency', 'instruction', 'extraction_hint']])
                context += f"Available options: {options_str}\n"
                context += f"Usage: Choose appropriate option and pass as {{'{var['name']}': 'chosen_key'}}\n"
            context += "\n"
    
    # TYPE 3: Direct Input Variables (Free Form)
    if direct_input_vars:
        context += "TYPE 3 - DIRECT INPUT VARIABLES (Extract from user request):\n"
        for var in direct_input_vars:
            context += f"Variable: {var['name']}\n"
            context += f"Description: {var['description']}\n" 
            context += f"Data Type: {var['data_type']}\n"
            
            variables_data = safe_json_load(var.get('variables', {}))
            if variables_data.get('extraction_hint'):
                context += f"Extraction Hint: {variables_data['extraction_hint']}\n"
            
            context += f"Usage: Extract value from user request and pass as {{'{var['name']}': 'extracted_value'}}\n"
            context += f"Example: User says 'buy 10 shares' → pass {{'{var['name']}': '10'}}\n"
            context += "\n"
    
    # TYPE 4: Tool Dependency Variables
    if tool_dependency_vars:
        context += "TYPE 4 - TOOL DEPENDENCY VARIABLES (Call other tools first):\n"
        for var in tool_dependency_vars:
            context += f"Variable: {var['name']}\n"
            context += f"Description: {var['description']}\n"
            context += f"Data Type: {var['data_type']}\n"
            
            variables_data = safe_json_load(var.get('variables', {}))
            if variables_data.get('tool_dependency'):
                context += f"Tool Dependency: {variables_data['tool_dependency']}\n"
                context += f"Instructions: {variables_data.get('instruction', 'Use the specified tool to get this value')}\n"
            
            context += f"Usage: Call dependency tool first, then pass result as {{'{var['name']}': 'tool_result'}}\n"
            context += "\n"
    
    context += "EXECUTION STRATEGY:\n"
    context += "1. TYPE 2 (Contextual): Choose from predefined options based on user request\n"
    context += "2. TYPE 3 (Direct Input): Extract numeric/text values directly from user prompt\n" 
    context += "3. TYPE 4 (Tool Dependency): Call specified tools first to get values\n"
    context += "4. Always pass resolved values using variable_mappings parameter\n"
    context += "5. Example: weather_api(variable_mappings={'city_id': 'BLR', 'quantity': '5', 'token': 'xyz123'})\n"
    
    return context


def build_tools_from_metadata(tool_data_list: List[Dict[str, Any]], user_variables: Dict[str, str], other_variables: List[Dict[str, Any]]) -> List[BaseTool]:
    """Build fresh tools with dynamic variable support"""
    tools = []
    for tool_data in tool_data_list:
        headers = safe_json_load(tool_data.get("headers"))
        params = safe_json_load(tool_data.get("query_params"))
        body = safe_json_load(tool_data.get("body"))

        # Create fresh tool instance
        tool = DynamicAPICallTool(
            name=tool_data["name"],
            description=tool_data.get("tool_description", "No description"),
            endpoint_url=tool_data["endpoint_url"],
            http_method=tool_data["http_method"],
            headers=headers,
            query_params=params,
            body=body,
            user_variables=user_variables.copy() if user_variables else {},
            other_variables=other_variables.copy() if other_variables else []
        )
        tools.append(tool)
    return tools


def build_agent_from_metadata(task_id: str, agent_id: str) -> Agent:
    """Build completely fresh agent with dynamic variable support and user-selected tools"""
    
    # Fetch all fresh data - no caching
    agent_data = fetch_agent_metadata(agent_id)
    user_variables = fetch_user_variables(agent_id)
    other_variables = fetch_other_variables(agent_id)
    agent_config = fetch_agent_configs()
    task_instructions = fetch_task_instructions(task_id)
    
    # Get agent's available tools from metadata
    agent_available_tools = agent_data.get("tools", [])
    
    # Resolve final tool IDs based on user selection and agent availability
    final_tool_ids = resolve_final_tool_ids(task_id, agent_id, agent_available_tools)
    
    # Fetch tool metadata for final tool list and build fresh tools
    tool_data_list = fetch_tools_metadata(final_tool_ids)
    tools = build_tools_from_metadata(tool_data_list, user_variables, other_variables)
    
    # Build complete enhanced backstory
    variable_context = build_variable_context(other_variables)
    task_context = f"\n\nTASK INSTRUCTIONS:\n{task_instructions}" if task_instructions else ""
    enhanced_backstory = agent_data["backstory"] + variable_context + task_context

    # Create fresh LLM instance with no caching
    llm = LLM(
        model=agent_config["llm"], 
        temperature=0.1,  # Consistent responses
        max_tokens=None,
        timeout=60
    ) if agent_config.get("llm") else LLM(model="groq/llama-3.3-70b-versatile", temperature=0.1)
    
    function_calling_llm_config = LLM(
        model=agent_config["function_calling_llm"],
        temperature=0.1
    ) if agent_config.get("function_calling_llm") else None
    
    # Create completely fresh agent instance
    agent = Agent(
        role=agent_data["role"],
        goal=agent_data["goal"],
        backstory=enhanced_backstory,
        tools=tools,
        llm=llm,
        function_calling_llm=function_calling_llm_config,
        verbose=agent_config.get("verbose", False),
        allow_delegation=agent_config.get("allow_delegation", False),
        max_iter=agent_config.get("max_iter", 5),
        max_rpm=agent_config.get("max_rpm"),
        max_execution_time=agent_config.get("max_execution_time", 120),
        max_retry_limit=agent_config.get("max_retry_limit", 1),
        allow_code_execution=agent_config.get("allow_code_execution", False),
        code_execution_mode=agent_config.get("code_execution_mode", "safe"),
        respect_context_window=agent_config.get("respect_context_window", True),
        multimodal=agent_config.get("multimodal", False),
        inject_date=agent_config.get("inject_date", False),
        date_format=agent_config.get("date_format", "%Y-%m-%d"),
        reasoning=agent_config.get("reasoning", False),
        max_reasoning_attempts=agent_config.get("max_reasoning_attempts"),
        embedder=agent_config.get("embedder"),
        knowledge_sources=agent_config.get("knowledge_sources"),
        system_template=agent_config.get("system_template"),
        prompt_template=agent_config.get("prompt_template"),
        response_template=agent_config.get("response_template"),
        step_callback=agent_config.get("step_callback"),
        cache=False,  # CRITICAL: No caching between executions
        memory=None,  # CRITICAL: No memory retention
        use_system_prompt=agent_config.get("use_system_prompt", True)
    )
    
    return agent