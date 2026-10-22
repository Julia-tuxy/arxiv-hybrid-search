from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import PaperResult, SearchResponse
from search import hybrid_search

app = FastAPI(title="ArXiv Semantic Search", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2, max_length=500, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50),
):
    raw = hybrid_search(query=q.strip(), top_k=top_k)
    return SearchResponse(
        query=q,
        total_results=len(raw),
        results=[PaperResult(**r) for r in raw],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
