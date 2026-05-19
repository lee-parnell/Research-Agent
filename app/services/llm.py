from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from app.config import get_settings
from typing import Tuple,List
import logging

logger=logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise research assistant. Your job is to synthesize
information STRICTLY from the provided source snippets.

Rules:
- Only use facts that appear in the provided snippets
- If the snippets don't contain enough information, say so clearly
- Never invent statistics, dates, or facts not in the sources
- Always cite which sources support each claim using [Source N] notation
- Be concise but comprehensive
"""

RESEARCH_PROMPT = """Topic: {topic}

Source Snippets (use ONLY these):
{snippets}

Task: Write a structured research summary with:
1. A 3-5 sentence executive summary
2. 4-6 key findings as bullet points
3. Any limitations or gaps in the available information

Format your response as JSON:
{{
  "summary": "executive summary here",
  "key_findings": ["finding 1", "finding 2", ...],
  "limitations": "any gaps or caveats"
}}
"""

class LLMService:
    def __init__(self):
        settings=get_settings()
        self.llm=ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            max_tokens=settings.max_tokens,
            temperature=0.1
        )

    def synthesize(self,topic:str,snippets:List[str])->Tuple[str,int]:
        """Synthesize research from retrieved snippets.
        Returns (raw_response_text, tokens_used)"""

        formatted_snippets="\n\n".join(
            f"[Source {i+1}]: {snippet}" for i,snippet in enumerate(snippets)
        )

        messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=RESEARCH_PROMPT.format(
                topic=topic,
                snippets=formatted_snippets
            ))
        ]

        logger.info(f"Calling Groq LLM for topic: '{topic[:60]}...'")
        response=self.llm.invoke(messages)

        tokens_used=response.response_metadata.get("token_usage",{}).get("total_tokens",0)
        logger.info(f"LLM response received | tokens used: {tokens_used}")

        return response.content,tokens_used
