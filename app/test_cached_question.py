import json
from pathlib import Path

import numpy as np

from app.semantic_retrieval import semantic_retrieve_from_embedding

DATA_DIR = Path("data")

def main():
    # 1) Load cached question embeddings + IDs
    q_embs = np.load(DATA_DIR / "zalo_question_embeddings.npy")  # shape (Nq, d)
    q_ids = json.load(open(DATA_DIR / "zalo_question_ids.json", encoding="utf-8"))

    # 2) Pick a question by ID (from your Zalo JSON, field "id")
    target_qid = "ade2b2ee4f5b869f75f0d183902382af"  # example
    idx = q_ids.index(target_qid)
    q_emb = q_embs[idx]

    # 3) Retrieve top 10 articles from Postgres via pgvector
    results = semantic_retrieve_from_embedding(q_emb, top_k=10)

    # 4) Inspect
    for r in results:
        print(
            f"{r['rank'] if 'rank' in r else ''} "
            f"{r['law_id']} - Điều {r['article_id']} "
            f"(distance={r['semantic_distance']:.4f})"
        )
        print(r["text"][:200], "...\n")

if __name__ == "__main__":
    main()
