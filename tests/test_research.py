"""Unit tests  - mocked so they run without real api keys.
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock,patch
from app.models.schemas import ResearchRequest,Source


MOCK_SOURCES=[
    Source(
        title="AI trend 2024",
        url="https://example.com/ai-trends",
        snippet="Large language models have shown remarkable progess in multimodal tasks",
        relevance_score=0.92
    ),
    Source(
        title="Multimodal AI Overview",
        url="https://example.com/multimodal",
        snippet="GPT-4V and Gemini Ultra demonstrate strong vision-language capabilities.",
        relevance_score=0.88,
    ),
]

MOCK_LLM_RESPONSE = """```json
{
  "summary": "Multimodal AI has advanced significantly in 2024, with models like GPT-4V and Gemini demonstrating strong vision-language understanding.",
  "key_findings": [
    "LLMs have made major progress in multimodal tasks",
    "GPT-4V and Gemini Ultra lead in vision-language capabilities",
    "Real-world applications are expanding rapidly"
  ],
  "limitations": "Sources are limited to publicly available information"
}
```"""
 
class TestResearchAgent:
    @patch("app.services.search.TavilyClient")
    @patch("app.services.embeddings.SentenceTransformer")
    @patch("app.services.llm.ChatGroq")
    def test_successfull_research(self,mock_groq,mock_st,mock_tavily):
        """Full agent pipeline with mock external services"""
        from app.services.agent import ResearchAgent

        #Setup Mock
        mock_tavily_instance=MagicMock()
        mock_tavily_instance.search.return_value={
            "results":[
                {"title":s.title,"url":s.url,"content":s.snippet,"score":s.relevance_score}
                for s in MOCK_SOURCES
            ]
        }
        mock_tavily.return_value=mock_tavily_instance

        mock_model=MagicMock()
        mock_model.encode.return_value=[[0.1]*384,[0.2]*384]
        mock_model.get_sentence_embedding_dimensions.return_value=384
        mock_st.return_value=mock_model

        mock_llm=MagicMock()
        mock_llm.invoke.return_value=MagicMock(
            content=MOCK_LLM_RESPONSE,
            response_metadata={"token_usage":{"total_tokens":350}}
        )
        mock_groq.return_value=mock_llm

        agent=ResearchAgent()
        request=ResearchRequest(topic="multimodal AI 2024",depth=2)


        assert request.topic="multimodal AI 2024"

    def test_research_validation_too_short(self):
        """topic must be atleast 3  characters"""
        with pytest.raises(Exception):
            ResearchRequest(topic="AI")

    def test_request_depth_clamped(self):
        """Depth must be between 1 and 5"""
        with pytest.raises(Exception):
            ResearchRequest(topic="valid topic",depth=10)
    
    def test_valid_request(self):
        request=ResearchRequest(topic="quantum computing breakthroughs",depth="3")
        assert request.depth==3
        assert request.langauage=="en"
    
    def test_hallucination_scorer(self):
        """Grounding score should be between 0 and 1"""
        from app.utils.hallucination import compute_grounding_score
        with patch("app.utils.hallucination.SentenceTransformer") as mock_st:
            import numpy as np
            mock_model=MagicMock()
            #similar vector ==high grounding
            mock_model.encode.return_value=np.array[[0.5]*384,[0.4]*384]
            mock_st.return_value=mock_model

            score,confidence=compute_grounding_score(
                summary="AI has advance greatly",
                source_snippets=["AI has made remarkable progress"]
            )
            assert 0.0<=score<=1.0
            assert confidence in ("high","medium","low")


