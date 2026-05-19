from fastapi import APIRouter,Depends,HTTPException
from app.models.schemas import ResearchRequest,ResearchResponse
from app.services.agent import ResearchAgent
from app.config import get_settings
import logging

logger=logging.getLogger(__name__)

router=APIRouter()

_agent:ResearchAgent | None=None

def get_agent()->ResearchAgent:
    global _agent
    if _agent is None:
        _agent=ResearchAgent()
    return _agent

@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="Run Autonomous Search",
    description="""Submit a topic and get a structured research report

    The agent will:
    1. Search the web for relevant sources
    2. Embed them into a local FAISS vector store
    3. Retrieve the most relevant context
    4. Synthesize a grounded summary using an LLM
    5. Score the output for hallucination risk
    """
)
async def run_research(
    request:ResearchRequest,
    agent:ResearchAgent=Depends(get_agent)
)->ResearchResponse:
    try:
        return agent.run(request)
    except ValueError as e:
        raise HTTPException(status_code=404,detail=str(e))
    except Exception as e:
        logger.error(f"Research failed: {e}",exc_info=True)
        raise HTTPException(status_code=500,detail="Research agent encountered an error")
