from typing import Any, Dict, List

import numpy as np

from app.config import get_connection
from app.data_loader import load_law_documents
from app.embedding.embedding import embed_text, DEFAULT_MODEL_KEY

# Map (law_id, article_id) -> doc_id for consistent IDs across JSON/DB
LAW_ART_TO_DOCID: Dict[tuple[str, str], int] = {
    (d["law_id"], d["article_id"]): d["doc_id"] for d in load_law_documents()
}


def _emb_to_pgvector_literal(emb: np.ndarray | List[float]) -> str:
    """
    Convert a numpy array or list of floats into a pgvector literal: '[0.1,0.2,0.3]'.
    """
    if isinstance(emb, list):
        emb = np.array(emb, dtype="float32")
    return "[" + ",".join(f"{float(x):.6f}" for x in emb.tolist()) + "]"


def semantic_retrieve(
    question_text: str,
    top_k: int = 200,
    model_key: str = DEFAULT_MODEL_KEY,
) -> List[Dict[str, Any]]:
    """
    Embedding-only retrieval on Zalo articles using the `articles.embedding` column.

      - embed question using the given model_key
      - pgvector search on articles.embedding
      - restrict to laws.source = 'zalo'
      - skip amending articles

    Returns list of dicts:
      {id, law_id, article_id, doc_id, text, tokens, semantic_distance}
    """
    # 1) Embed the question
    q_emb = embed_text(question_text, model_key=model_key)
    q_literal = _emb_to_pgvector_literal(q_emb)

    conn = get_connection()
    cur = conn.cursor()

    try:
        # 2) Vector search directly on articles.embedding
        cur.execute(
            """
            SELECT
                a.id AS article_db_id,
                a.law_id,
                a.article_id,
                a.text,
                a.tokens,
                (a.embedding <=> %s::vector) AS distance
            FROM articles a
            JOIN laws l
                ON l.law_id = a.law_id
            WHERE l.source = 'zalo'
              AND COALESCE(a.is_amending_article, FALSE) = FALSE
              AND a.embedding IS NOT NULL
            ORDER BY a.embedding <=> %s::vector
            LIMIT %s;
            """,
            (q_literal, q_literal, top_k),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    results: List[Dict[str, Any]] = []
    for row in rows:
        art_db_id, law_id, article_id, text, tokens, distance = row
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


def semantic_retrieve_from_embedding(
    q_emb: np.ndarray | List[float],
    top_k: int = 200,
) -> List[Dict[str, Any]]:
    """
    Embedding-only retrieval using a *precomputed* question embedding.

    This mirrors `semantic_retrieve`, but skips the `embed_text` API call and
    instead takes an already-embedded question vector (same dimension as
    `articles.embedding` in Postgres).
    """
    # Ensure numpy float32 1D
    if isinstance(q_emb, list):
        q_emb = np.array(q_emb, dtype="float32")
    else:
        q_emb = np.asarray(q_emb, dtype="float32")

    if q_emb.ndim == 2:
        q_emb = q_emb[0]

    q_literal = _emb_to_pgvector_literal(q_emb)

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT
                a.id AS article_db_id,
                a.law_id,
                a.article_id,
                a.text,
                a.tokens,
                (a.embedding <=> %s::vector) AS distance
            FROM articles a
            JOIN laws l
                ON l.law_id = a.law_id
            WHERE l.source = 'zalo'
              AND COALESCE(a.is_amending_article, FALSE) = FALSE
              AND a.embedding IS NOT NULL
            ORDER BY a.embedding <=> %s::vector
            LIMIT %s;
            """,
            (q_literal, q_literal, top_k),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    results: List[Dict[str, Any]] = []
    for row in rows:
        art_db_id, law_id, article_id, text, tokens, distance = row
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
