from tavily import TavilyClient
from app.config import get_settings
from typing import List
from app.models.schemas import Source
import logging

logger=logging.getLogger(__name__)

class SearchService():
    def __init__(self):
        settings=get_settings()
        self.client=TavilyClient(api_key=settings.tavily_api_key)
        self.max_results=settings.max_search_results

    def search(self,query:str,depth:int=3)->List[Source]:
        """
        Search the web using tavily and return ranked sources.
        depth 1-2=basic, depth 3-4=advanced, depth 5-6=deep(cost more API calls)
        """
        search_depth="advanced" if depth>=3 else "basic"
        max_results=min(depth*2,self.max_results)

        logger.info(f"Searching: '{query}' | depth={search_depth} | results={max_results}")

        response=self.client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False
        )

        sources=[]
        for result in response.get("results",[]):
            sources.append(Source(
                title=result.get("title","untitled"),
                url=result.get("url",""),
                snippet=result.get("content",""),
                relevance_score=result.get("score",0.5)
            ))

        sources.sort(key=lambda s:s.relevance_score,reverse=True)
        logger.info(f"Found {len(sources)} sources")
        return sources
