from fastapi import HTTPException
import app.main  # Import the main module to access global instances

# Dependency functions moved from main.py

async def get_brain_agent():
    # Access the instance from main.py
    if app.main.brain_agent_instance is None: 
        raise HTTPException(
            status_code=500, 
            detail="Brain agent not initialized"
        )
    return app.main.brain_agent_instance

async def get_vector_store():
    # Access the instance from main.py
    if app.main.vector_store is None:
        raise HTTPException(
            status_code=500, 
            detail="Vector store not initialized"
        )
    return app.main.vector_store 