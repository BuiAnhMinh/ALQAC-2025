from typing import Any, Dict, List

import numpy as np

from app.config import get_connection
from app.data_loader import load_law_documents
from app.embedding import embed_text

# Map (law_id, article_id) -> doc_id for consistent IDs across JSON/DB
LAW_ART_TO_DOCID: Dict[tuple[str, str], int] = {
    (d["law_id"], d["article_id"]): d["doc_id"] for d in load_law_documents()
}

def _emb_to_pgvector_literal(emb: np.ndarray | List[float]) -> str:
    """
    Convert a numpy array or list of floats into a pgvector literal: '[0.1,0.2,0.3]'.
    """
    if isinstance(emb, np.ndarray):
        emb = emb.tolist()
    return "[" + ",".join(f"{float(x):.6f}" for x in emb) + "]"


def semantic_retrieve(
    question_text: str,
    top_k: int = 200,
    question_embedding: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    """
    Vector retrieval on articles that come from Zalo and are marked as amending.

    Returns top_k candidates ordered by vector distance, each with DB id + metadata.
    """
    # 1) Get embedding for the question
    q_emb = question_embedding if question_embedding is not None else embed_text(question_text)

    # 2) Convert to pgvector literal string: "[0.123456,0.234567,...]"
    q_emb_str = _emb_to_pgvector_literal(q_emb)

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id,
                   law_id,
                   article_id,
                   text,
                   tokens,
                   embedding <=> %s AS distance
            FROM articles
            WHERE embedding IS NOT NULL
              AND source = 'zalo'
              AND is_amending_article = FALSE
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (q_emb_str, q_emb_str, top_k),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    results: List[Dict[str, Any]] = []
    for art_db_id, law_id, article_id, text, tokens, distance in rows:
        doc_id = LAW_ART_TO_DOCID.get((law_id, article_id))
        results.append(
            {
                "id": art_db_id,
                "law_id": law_id,
                "article_id": article_id,
                "doc_id": doc_id,
                "text": text,
                "tokens": tokens,
                "semantic_distance": float(distance),
            }
        )

    return results
