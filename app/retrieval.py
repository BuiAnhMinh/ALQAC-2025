"""
Unified retrieval interface for experiments on Zalo corpus.

Methods:
  - retrieve_bm25:        pure BM25
  - retrieve_embedding:   embedding-only (pgvector)
  - retrieve_hybrid:      semantic -> BM25 rerank
"""

from typing import List, Dict, Any

from app.semantic_retrieval import semantic_retrieve
from app.lexical_retrieval import (
    bm25_pure,
    bm25_lexical_rerank,
    semantic_then_lexical,
    bm25_hybrid_retrieve,
)
from app.embedding.embedding import DEFAULT_MODEL_KEY


# ---------- Public methods ----------

def retrieve_bm25(
    question_text: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Pure BM25 retrieval on Zalo articles.
    """
    return bm25_pure(question_text, top_k=top_k)


def retrieve_embedding_only(
    question_text: str,
    top_k: int = 10,
    model_key: str = DEFAULT_MODEL_KEY,
) -> List[Dict[str, Any]]:
    """
    Embedding-only retrieval via pgvector on Zalo articles.
    """
    return semantic_retrieve(
        question_text=question_text,
        top_k=top_k,
        model_key=model_key,
    )


def retrieve_hybrid(
    question_text: str,
    semantic_top_k: int = 200,
    lexical_top_k: int = 10,
    model_key: str = DEFAULT_MODEL_KEY,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: semantic top-k then BM25 rerank.
    """
    combined = semantic_then_lexical(
        question_text=question_text,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
        model_key=model_key,
    )
    return combined["lexical_results"]


# Backwards-compatible alias used by test_with_one_question.py
def retrieve_top_articles(
    question_text: str,
    semantic_top_k: int = 200,
    lexical_top_k: int = 10,
    model_key: str = DEFAULT_MODEL_KEY,
) -> List[Dict[str, Any]]:
    return retrieve_hybrid(
        question_text=question_text,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
        model_key=model_key,
    )
