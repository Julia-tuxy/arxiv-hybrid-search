from pydantic import BaseModel
from typing import List


class PaperResult(BaseModel):
    arxiv_id: str
    title: str
    abstract_snippet: str
    authors: str
    year: str
    categories: str
    score: float
    vector_score: float
    bm25_score: float
    retrieved_by: str = "both"   # which arm(s) returned this document


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[PaperResult]
