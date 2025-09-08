from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from orchestrator import execute_agent_task
import os

app = FastAPI(
    title="CrewAI Agent Executor API - Service 2",
    description="API for executing individual CrewAI agents with dynamic variables",
    version="1.0.0"
)

class AgentExecutionRequest(BaseModel):
    task_id: str
    agent_id: str
    agent_prompt: str

@app.get("/")
async def root():
    return {
        "message": "CrewAI Agent Executor API - Service 2",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/execute-agent")
async def execute_agent(request: AgentExecutionRequest):
    """
    Execute a specific agent with given prompt
    
    - **task_id**: ID of the task for chat history context
    - **agent_id**: ID of the agent to execute
    - **agent_prompt**: Prompt to send to the agent
    """
    try:
        result = execute_agent_task(
            task_id=request.task_id,
            agent_id=request.agent_id,
            agent_prompt=request.agent_prompt
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Agent Executor API is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)