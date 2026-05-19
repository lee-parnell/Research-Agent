from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import research
from app.config import get_settings
from app.models.schemas import HealthResponse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)  # Fixed: changed get_logger to getLogger

settings = get_settings()
if settings.langchain_tracing_v2 == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    logger.info("Langsmith tracing enabled")

app = FastAPI(
    title="Autonomous Research Agent API",
    description="""
    ## What this does
    Submit a topic → get back a fully sourced, hallucination-scored research report.
    
    ## Architecture
    - **Web Search**: Tavily 
    - **Embeddings**: HuggingFace sentence-transformers (local, free)
    - **Vector Store**: FAISS (in-memory, free)
    - **LLM**: Groq llama3-70b 
    - **Tracing**: LangSmith
    
    ## Key Features
    - RAG pipeline with source grounding
    - Hallucination risk score on every response
    - Confidence rating (high/medium/low)
    - Structured JSON output, production-ready
    """,
    version="1.0.0",  # FIXED: changed : to =
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]  # FIXED: changed [*] to ["*"]
)

app.include_router(research.router, prefix="/api/v1", tags=["Research"])

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        models={
            "llm": settings.llm_model,
            "embeddings": settings.embedding_model
        }
    )

@app.get("/", include_in_schema=False)
async def root():
    return {"Message": "Autonomous Research Agent API", "docs": "/docs"}