from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from app.config import ARTICLE_EMB_PATH, get_connection
from app.data_loader import load_law_documents


BATCH_SIZE = 1000  # DB update batch size, not OpenAI calls

#using this because no openrouter credit

def main():
    # 1) Load documents and cached embeddings
    docs: List[Dict[str, Any]] = load_law_documents()
    num_docs = len(docs)
    print(f"Loaded {num_docs} docs from loader.")

    if not ARTICLE_EMB_PATH.exists():
        raise FileNotFoundError(f"Embedding file not found: {ARTICLE_EMB_PATH}")

    embs = np.load(ARTICLE_EMB_PATH)
    num_embs = embs.shape[0]
    print(f"Loaded embeddings from {ARTICLE_EMB_PATH}, shape={embs.shape}")

    # 2) Handle mismatch between docs and embeddings
    if num_embs != num_docs:
        print(
            f"WARNING: {num_embs} embeddings vs {num_docs} docs.\n"
            "Likely cause: you generated embeddings when the corpus had fewer docs "
            "(e.g. only ALQAC), and later added more docs (e.g. Zalo).\n"
            "=> We will only update the first N = min(num_embs, num_docs) docs, "
            "and leave the rest with NULL embeddings."
        )
    n = min(num_embs, num_docs)

    print(f"Using N = {n} pairs (doc[i] ↔ emb[i]) to update DB.")

    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    try:
        total = n
        print("Start updating article embeddings in DB from .npy...")

        batch_values = []
        count = 0

        # Only iterate over the first n docs/embs
        for i in range(n):
            d = docs[i]
            emb = embs[i]

            law_id = d["law_id"]
            article_id = d["article_id"]

            # pgvector expects a Python list[float]
            emb_list = emb.tolist()

            batch_values.append((emb_list, law_id, article_id))
            count += 1

            # When batch is full or at the end → execute
            if len(batch_values) >= BATCH_SIZE or i == total - 1:
                cur.executemany(
                    """
                    UPDATE articles
                    SET embedding = %s
                    WHERE law_id = %s AND article_id = %s;
                    """,
                    batch_values,
                )
                conn.commit()
                print(f"Updated {count}/{total} articles so far...")

                batch_values.clear()

        print("Finished updating all available embeddings from .npy.")
        print(
            f"Note: {num_embs} embeddings applied to first {n} docs; "
            f"remaining {num_docs - n} docs still have NULL embedding."
        )
    except Exception as e:
        conn.rollback()
        print("ERROR while filling embeddings from npy:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("DB connection closed.")


if __name__ == "__main__":
    main()