from typing import Any, Dict, List

import numpy as np

from app.config import get_connection
from app.data_loader import load_law_documents
from app.embedding import embed_text, DEFAULT_MODEL_KEY

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
    Embedding-only retrieval on Zalo articles:
      - embed question using the given model_key
      - pgvector search on article_embeddings.embedding
      - restrict to laws.source = 'zalo'
    Returns list of dicts:
      {id, law_id, article_id, doc_id, text, tokens, semantic_distance}
    """
    # 1) Embed the question
    q_emb = embed_text(question_text, model_key=model_key)
    q_literal = _emb_to_pgvector_literal(q_emb)

    conn = get_connection()
    cur = conn.cursor()

    try:
        # 2) Vector search
        cur.execute(
            """
            SELECT
                a.id AS article_db_id,
                a.law_id,
                a.article_id,
                a.text,
                a.tokens,
                (e.embedding <=> %s::vector) AS distance
            FROM article_embeddings e
            JOIN articles a
                ON a.id = e.article_id
            JOIN laws l
                ON l.law_id = a.law_id
            WHERE e.model_key = %s
                AND l.source = 'zalo'
                AND COALESCE(a.is_amending_article, FALSE) = FALSE
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
            """,
            (q_literal, model_key, q_literal, top_k),
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
