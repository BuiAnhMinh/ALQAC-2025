from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from app.config import ARTICLE_EMB_PATH, get_connection
from app.data_loader import load_law_documents


def main():
    docs: List[Dict[str, Any]] = load_law_documents()
    print("Loaded", len(docs), "docs from loader.")

    article_embeddings = np.load(ARTICLE_EMB_PATH)
    print(
        "Loaded embeddings from", ARTICLE_EMB_PATH,
        ", shape=", article_embeddings.shape,
    )

    if article_embeddings.shape[0] != len(docs):
        raise ValueError(
            f"Mismatch: {article_embeddings.shape[0]} embeddings vs {len(docs)} docs. "
            "You must ensure they were created from the same corpus and order."
        )

    conn = get_connection()
    cur = conn.cursor()

    try:
        updated = 0
        for i, doc in enumerate(tqdm(docs, desc="Filling DB embeddings from .npy")):
            law_id = doc["law_id"]
            article_id = doc["article_id"]
            emb = article_embeddings[i].tolist()

            cur.execute(
                """
                UPDATE articles
                SET embedding = %s
                WHERE law_id = %s AND article_id = %s;
                """,
                (emb, law_id, article_id),
            )
            updated += 1

            if updated % 5000 == 0:
                conn.commit()
                print(f"Committed {updated} embeddings to DB.")

        conn.commit()
        print(f"Finished updating embeddings for {updated} articles.")
    except Exception as e:
        conn.rollback()
        print("ERROR during db_fill_embeddings_from_npy:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("DB connection closed.")


if __name__ == "__main__":
    main()
