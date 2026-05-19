"""Hallucination control via source grounding check.
Strategy: embed both the LLM summary and the source snippets,
compute cosine similarity. Low similarity = potential hallucination."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List,Tuple
from app.config import get_settings

def compute_grounding_score(summary:str,source_snippets:List[str])->Tuple[float,str]:
    """
    Return (hallucination_score, confidence_label)
    hallucination_score: 0.0=fully grounded, 1.0=high risk
    confidence: 'high' | 'medium' | 'low'
    """
    if not source_snippets:
        return 1.0,"low"

    settings=get_settings()
    model=SentenceTransformer(settings.embedding_model)

    all_texts=[summary]+source_snippets
    vectors=model.encode(all_texts,convert_to_numpy=True,show_progress_bar=False)

    summary_vec=vectors[0]
    source_vecs=vectors[1:]

    def cosine_sim(a,b):
        return float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    similarities=[cosine_sim(summary_vec,sv) for sv in source_vecs]
    max_similarity=max(similarities)
    avg_similarity=sum(similarities)/len(similarities)

    grounding=(max_similarity*0.6) + (avg_similarity*0.4)
    hallucination_score=round(1.0 - grounding,4)

    if grounding > 0.75:
        confidence="high"
    elif grounding > 0.55:
        confidence="medium"
    else:
        confidence="low"

    return hallucination_score,confidence
