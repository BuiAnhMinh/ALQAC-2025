from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from app.config import (
    MAX_CHARS,
    BATCH_SIZE,
    get_client,
    get_connection,
)
from app.embedding import EMBEDDING_MODELS, DEFAULT_MODEL_KEY


def preprocess_batch(texts: List[str]) -> List[str]:
    processed = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        if len(t) > MAX_CHARS:
            t = t[:MAX_CHARS]
        processed.append(t)
    return processed


def _emb_to_pgvector_literal(emb: np.ndarray) -> str:
    # '[0.1,0.2,...]'
    return "[" + ",".join(f"{float(x):.6f}" for x in emb.tolist()) + "]"


def main(model_key: str = DEFAULT_MODEL_KEY) -> None:
    """
    Fill the article_embeddings table for Zalo articles for a given model_key.
    Run e.g.:
      python -m app.db_fill_embeddings          # uses default model_key
      python -m app.db_fill_embeddings other_model
    """
    from sys import argv

    if len(argv) >= 2:
        model_key = argv[1]

    cfg = EMBEDDING_MODELS[model_key]
    model_name = cfg["model_name"]
    print(f"Filling embeddings for model_key={model_key}, model_name={model_name}")

    conn = get_connection()
    cur = conn.cursor()

    try:
        # Fetch Zalo articles that DON'T yet have embeddings for this model_key
        cur.execute(
            """
            SELECT a.id, a.text
            FROM articles a
            JOIN laws l ON l.law_id = a.law_id
            LEFT JOIN article_embeddings e
                ON e.article_id = a.id AND e.model_key = %s
            WHERE l.source = 'zalo'
                AND COALESCE(a.is_amending_article, FALSE) = FALSE
                AND e.id IS NULL
            ORDER BY a.id
            """,
            (model_key,),
        )
        rows: List[Tuple[int, str]] = cur.fetchall()
        print(f"Articles missing embeddings for model_key={model_key}: {len(rows)}")

        if not rows:
            print("Nothing to do.")
            return

        client = get_client()
        total = len(rows)
        for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding batches"):
            batch = rows[i : i + BATCH_SIZE]
            article_ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]

            texts = preprocess_batch(texts)

            resp = client.embeddings.create(
                model=model_name,
                input=texts,
            )
            embeddings = [np.array(d.embedding, dtype="float32") for d in resp.data]

            for art_id, emb in zip(article_ids, embeddings):
                literal = _emb_to_pgvector_literal(emb)
                cur.execute(
                    """
                    INSERT INTO article_embeddings (article_id, model_key, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (article_id, model_key) DO UPDATE
                    SET embedding = EXCLUDED.embedding;
                    """,
                    (art_id, model_key, literal),
                )

            conn.commit()

        print("All embeddings generated and stored in article_embeddings.")
    except Exception as e:
        conn.rollback()
        print("ERROR during embedding fill:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("DB connection closed.")


if __name__ == "__main__":
    main()
