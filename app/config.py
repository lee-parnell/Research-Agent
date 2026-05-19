from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    groq_api_key:str
    tavily_api_key:str
    langchain_api_key:str=""
    langchain_tracing_v2:str="false"
    langchain_project:str="research-agent"

    app_env:str="development"
    max_search_results:int=5
    max_tokens:int=2048
    llm_model: str = "llama3-70b-8192"
    embedding_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file=".env"

@lru_cache
def get_settings()->Settings:
    return Settings()