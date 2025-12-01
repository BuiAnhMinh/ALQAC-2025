from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from app.config import (
    EMB_MODEL,
    MAX_CHARS,
    BATCH_SIZE,
    get_client,
    get_connection,
)

# for openai calling then fill the embedding into database (postgres/pgvector docker database)

def preprocess_batch(texts: List[str]) -> List[str]:
    """Match your existing preprocessing (truncate, remove newlines)."""
    cleaned = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        t = t.replace("\n", " ")
        if len(t) > MAX_CHARS:
            t = t[:MAX_CHARS]
        cleaned.append(t)
    return cleaned


def fetch_articles_without_embedding(cur, limit: int) -> List[Tuple[int, str]]:
    """
    Fetch a batch of articles that don't have embeddings yet.
    Returns list of (id, text).
    """
    cur.execute(
        """
        SELECT id, text
        FROM articles
        WHERE embedding IS NULL
        ORDER BY id
        LIMIT %s;
        """,
        (limit,),
    )
    return cur.fetchall()


def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Call OpenAI embeddings API in batch using your config.
    Returns np.ndarray of shape (len(texts), dim).
    """
    client = get_client()
    resp = client.embeddings.create(
        model=EMB_MODEL,
        input=texts,
    )
    embs = [np.array(item.embedding, dtype="float32") for item in resp.data]
    return np.vstack(embs)


def main():
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        total_done = 0

        while True:
            rows = fetch_articles_without_embedding(cur, BATCH_SIZE)
            if not rows:
                print("No more articles without embedding. Done.")
                break

            ids = [r[0] for r in rows]
            raw_texts = [r[1] for r in rows]
            texts = preprocess_batch(raw_texts)

            print(f"Embedding batch of {len(ids)} articles...")
            embs = embed_batch(texts)  # shape: (batch_size, dim)

            # Update each article with its embedding
            for art_id, emb in zip(ids, embs):
                cur.execute(
                    """
                    UPDATE articles
                    SET embedding = %s
                    WHERE id = %s;
                    """,
                    (emb.tolist(), art_id),
                )

            conn.commit()
            total_done += len(ids)
            print(f"Embedded and updated {total_done} articles so far...")

        print("All embeddings generated and stored in DB.")
    except Exception as e:
        conn.rollback()
        print("ERROR during embedding generation:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("DB connection closed.")


if __name__ == "__main__":
    main()
