from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List,Tuple
from app.config import get_settings
import logging

logger=logging.getLogger(__name__)

class EmbeddingService:
    """
    Local Embedding + FAISS Vector Store
    """

    def __init__(self):
        settings=get_settings()
        model_name=settings.embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        self.model=SentenceTransformer(model_name)
        self.dimension=self.model.get_embedding_dimension()

    def embed(self,texts:List[str])->np.ndarray:
        """Convert a list of text chunks into embedding vectors"""
        vectors=self.model.encode(texts,convert_to_numpy=True,show_progress_bar=True)
        return vectors.astype("float32")

    def build_index(self,texts:List[str])->Tuple[faiss.IndexFlatIP,List[str]]:
        """Build an in-memory FAISS index from source snippets.
        Returns the index and the original texts (for retrieval)"""

        if not texts:
            raise ValueError("No texts provided to build index")
        vectors=self.embed(texts)
        faiss.normalize_L2(vectors)

        index=faiss.IndexFlatIP(self.dimension)
        index.add(vectors)
        logger.info(f"Built faiss index with {index.ntotal} vectors")
        return index,texts

    def retrieve(
        self,
        query:str,
        index:faiss.IndexFlatIP,
        texts:List[str],
        top_k:int=3,
    )->List[Tuple[str,float]]:
        """
        Return top-k most relevant chunks for query.
        Returns list of (text, similarity_score) tuples"""
        query_vec=self.embed([query])
        faiss.normalize_L2(query_vec)

        scores,indices=index.search(query_vec,top_k)
        results=[]
        for score,idx in zip(scores[0],indices[0]):
            if idx>=0:
                results.append((texts[idx],float(score)))
        return results
