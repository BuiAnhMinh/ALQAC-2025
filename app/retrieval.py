"""
Compatibility wrapper pointing to the new two-stage retrieval modules.

- semantic_retrieval.semantic_retrieve: vector search on Zalo amending articles
- lexical_retrieval.bm25_lexical_rerank: BM25-Okapi rerank within provided candidates
- lexical_retrieval.semantic_then_lexical: semantic top-k then BM25 top-k
"""

from typing import List, Dict, Any

from app.semantic_retrieval import semantic_retrieve
from app.lexical_retrieval import (
    bm25_lexical_retrieve,
    bm25_lexical_rerank,
    semantic_then_lexical,
    underthesea_tokenizer,
)

__all__ = [
    "semantic_retrieve",
    "bm25_lexical_retrieve",
    "bm25_lexical_rerank",
    "semantic_then_lexical",
    "underthesea_tokenizer",
    "retrieve_top_articles",
]


def retrieve_top_articles(
    question_text: str,
    semantic_top_k: int = 200,
    lexical_top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Convenience helper: semantic top_k -> BM25 top_k -> return lexical results only.
    """
    result = semantic_then_lexical(
        question_text=question_text,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
    )
    return result["lexical_results"]
