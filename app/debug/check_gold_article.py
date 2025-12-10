from pathlib import Path
import json

import numpy as np

from app.config import get_connection
from app.semantic_retrieval import _emb_to_pgvector_literal
from app.data_loader import load_zalo_questions

DATA_DIR = Path("data")


def check_gold_rank_for_question(
    target_qid: str,
    gold_law_id: str,
    gold_article_id: str,
    top_n: int = 500,
):
    # 1) Load cached Zalo question embeddings + ids
    q_embs = np.load(DATA_DIR / "zalo_question_embeddings.npy")  # (N_q, d)
    q_ids = json.load(open(DATA_DIR / "zalo_question_ids.json", encoding="utf-8"))

    # 2) Find the question index and embedding
    idx = q_ids.index(target_qid)
    q_emb = q_embs[idx]
    q_literal = _emb_to_pgvector_literal(q_emb)

    # 3) Optional: show the question text
    questions = load_zalo_questions()
    q_obj = next(q for q in questions if q["id"] == target_qid)
    print("Question ID:", target_qid)
    print("Question text:", q_obj["text"])
    print("GOLD:", (gold_law_id, gold_article_id))
    print()

    # 4) Query Postgres for nearest neighbours
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT
                a.law_id,
                a.article_id,
                (a.embedding <=> %s::vector) AS distance
            FROM articles a
            JOIN laws l ON l.law_id = a.law_id
            WHERE l.source = 'zalo'
              AND COALESCE(a.is_amending_article, FALSE) = FALSE
              AND a.embedding IS NOT NULL
            ORDER BY a.embedding <=> %s::vector
            LIMIT %s;
            """,
            (q_literal, q_literal, top_n),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    # 5) Look for the gold article in the ranked list
    found_rank = None
    for rank, (law_id, article_id, distance) in enumerate(rows, start=1):
        if law_id == gold_law_id and article_id == gold_article_id:
            found_rank = rank
            print(
                f"FOUND GOLD at rank {rank} with distance={float(distance):.4f}"
            )

    if found_rank is None:
        print(f"GOLD ({gold_law_id}, {gold_article_id}) not found in top {top_n}.")
    print()

    # 6) Optionally show top 10 for inspection
    print("Top 10 neighbors:")
    for rank, (law_id, article_id, distance) in enumerate(rows[:10], start=1):
        print(
            f"{rank}. {law_id} - Điều {article_id} (distance={float(distance):.4f})"
        )


if __name__ == "__main__":
    # Example for your question:
    # Q: "Công an xã xử phạt lỗi không mang bằng lái xe có đúng không?"
    # Zalo question id (adjust if different):
    target_qid = "0637bf82c8b290c7875c5bfddbf91df5"  # put the actual id from your Zalo JSON here
    gold_law_id = "47/2011/tt-bca"
    gold_article_id = "7"

    check_gold_rank_for_question(target_qid, gold_law_id, gold_article_id)
