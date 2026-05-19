"""
Research Agent — Orchestration Layer

Pipeline:
  1. Web search (Tavily)   → retrieve relevant sources
  2. Embed + index (FAISS) → build local vector store from sources
  3. RAG retrieval         → find most relevant snippets for topic
  4. LLM synthesis (Groq)  → generate structured summary
  5. Hallucination check   → score how grounded the output is
"""

import json
import time
import logging
from app.services.search import SearchService
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.utils.hallucination import compute_grounding_score
from app.models.schemas import ResearchRequest,ResearchResponse,Source

logger=logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        logger.info("Initializing Research Agent Service")
        self.SearchService=SearchService()
        self.EmbeddingService=EmbeddingService()
        self.LLMService=LLMService()
        logger.info("Research Agent ready")

    def run(self,request:ResearchRequest)->ResearchResponse:
        start_time=time.time()
        logger.info(f"[AGENT] Starting research: '{request.topic}'")


        ####### WEB SEARCH -------------------------
        sources:list[Source]=self.SearchService.search(
            query=request.topic,
            depth=request.depth
        )

        if not sources:
            raise ValueError(f"No search result found for: {request.topic}")


        ###Build FAISS VECTOR INDEX----------------
        snippets=[s.snippet for s in sources if s.snippet]
        faiss_index,indexed_texts=self.EmbeddingService.build_index(snippets)


        ####RAG - RETRIEVE MOST RELEVANT SNIPPET
        top_chunks=self.EmbeddingService.retrieve(
            query=request.topic,
            index=faiss_index,
            texts=indexed_texts,
            top_k=min(4,len(snippets))

        )
        relevant_snippets=[chunk for chunk,_score in top_chunks]


        ###LLM SYNTHESIS
        raw_response,tokens_used=self.LLMService.synthesize(
            topic=request.topic,
            snippets=relevant_snippets
        )

        llm_data=self._parse_llm_response(raw_response)


        ###HALLUCINATION SCORING
        hallucination_score,confidence=compute_grounding_score(
            summary=llm_data.get("summary",""),
            source_snippets=relevant_snippets
        )

        duration=round(time.time() - start_time,2)
        logger.info(
            f"[AGENT] Done in {duration}s | "
            f"tokens={tokens_used} | "
            f"hallucination={hallucination_score} | "
            f"confidence={confidence}"
        )
        return ResearchResponse(
            topic=request.topic,
            summary=llm_data.get("summary", ""),
            key_findings=llm_data.get("key_findings", []),
            sources=sources,
            hallucination_score=hallucination_score,
            confidence=confidence,
            tokens_used=tokens_used,
            duration_seconds=duration,
        )

    def _parse_llm_response(self,raw:str)->dict:
        """Extract JSON from LLM response, with fallback for malformed output."""
        try:
            clean=raw.strip()
            if clean.startswith("```"):
                clean=clean.split("```")[1]
                if clean.startswith("json"):
                    clean=clean[4:]
            return json.loads(clean.strip())
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON, using fallback parser")
            return {
                "summary":raw[:500],
                "key_findings":[raw[500:1000]] if len(raw) > 500 else [],
                "limitations":"response could not be fully parsed"
            }
