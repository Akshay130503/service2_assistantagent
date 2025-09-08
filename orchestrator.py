from typing import List, Dict, Any
from agent_builder import build_agent_from_metadata
from data_service import (
    fetch_agent_chat_history,
    insert_agent_prompt,
    update_agent_response
)
from fastapi import HTTPException
import traceback


def execute_agent_task(task_id: str, agent_id: str, agent_prompt: str) -> Dict[str, Any]:
    """Execute agent task and manage chat history"""
    chat_id = None
    
    try:
        # Insert agent prompt and get chat ID
        chat_id = insert_agent_prompt(task_id, agent_id, agent_prompt)
        
        # Build agent from metadata
        agent = build_agent_from_metadata(agent_id)
        
        # Get chat history
        chat_history = fetch_agent_chat_history(task_id)
        
        # Prepare messages for agent
        if not chat_history:
            messages = agent_prompt
        else:
            # Append new prompt to history
            chat_history.append({"role": "user", "content": agent_prompt})
            messages = chat_history
        
        # Execute agent
        result = agent.kickoff(messages)
        response = result.raw
        
        # Update response in the same chat row
        update_agent_response(chat_id, response)
        
        return {
            "success": True,
            "response": response,
            "chat_id": chat_id
        }
        
    except Exception as e:
        error_msg = f"Agent execution error: {str(e)}"
        print(f"Error in execute_agent_task: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Update with error response if chat_id exists
        if chat_id:
            try:
                update_agent_response(chat_id, f"Error: {error_msg}")
            except:
                pass
        
        raise HTTPException(status_code=500, detail=error_msg)