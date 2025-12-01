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


def preprocess_batch(texts: List[str]) -> List[str]:
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


def main():
    client = get_client()
    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        total_updated = 0

        while True:
            rows = fetch_articles_without_embedding(cur, BATCH_SIZE)
            if not rows:
                print("No more articles without embeddings. Done.")
                break

            ids = [row[0] for row in rows]
            texts = [row[1] for row in rows]
            texts = preprocess_batch(texts)

            print(f"Embedding {len(texts)} articles starting from id {ids[0]}...")
            resp = client.embeddings.create(model=EMB_MODEL, input=texts)
            embs = [np.array(item.embedding, dtype="float32").tolist() for item in resp.data]

            batch_values = list(zip(embs, ids))

            cur.executemany(
                """
                UPDATE articles
                SET embedding = %s
                WHERE id = %s;
                """,
                batch_values,
            )
            conn.commit()

            total_updated += len(ids)
            print(f"Updated embeddings for {total_updated} articles so far...")

        print("All embeddings generated and stored in DB.")
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
