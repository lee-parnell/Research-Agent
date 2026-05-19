from pydantic import BaseModel,Field
from typing import List,Optional
from datetime import datetime

class ResearchRequest(BaseModel):
    topic:str=Field(...,min_length=3,max_length=500,description="topic to research")
    depth:int=Field(default=3,ge=1,le=5,description="Search depth: 1=quick, 5=deep")
    language:str=Field(default="en",description="Output Language (e.g. 'en','ur')")

    class Config:
        json_schema_extra={
            "example":{
                "topic":"latest breakthroughs in multimodal AI 2024",
                "depth":3,
                "language":"en"
            }
        }

class Source(BaseModel):
    title:str
    url:str
    snippet:str
    relevance_score:float=Field(ge=0.0,le=1.0)


class ResearchResponse(BaseModel):
    topic:str
    summary:str
    key_findings:List[str]
    sources:List[Source]
    hallucination_score:float=Field(
        ge=0.0,le=1.0,
        description="0= fully grounded, 1=high hallucination risk"
    )
    confidence:str=Field(description="high | medium | low")
    tokens_used:int
    duration_seconds:float
    timestamp:datetime=Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status:str
    version:str
    models:dict
