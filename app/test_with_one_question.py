"""
Minimal sanity check for BM25 + cosine using DB embeddings.

Run (inside Docker app container):
    python -m app.test_one_question
"""

import numpy as np
import time

from app import retrieval
from app.data_loader import load_train_data


def test_single_question(
    q_index: int = 0,
    top_k_lexical: int = 200,
    top_k_final: int = 10,
    alpha: float = 0.6,
    logreg_model=None,
):
    # 1) Pick a training question
    train_data = retrieval.train_data  # already loaded in retrieval.py
    q = train_data[q_index]
    qid = q["question_id"]
    qtext = q["text"]

    # 2) Get GOLD doc_ids for this question
    gold_doc_ids = retrieval.QUESTION_GOLD_DOCIDS.get(qid, set())

    print("==========================================")
    print(f"Question index : {q_index}")
    print(f"Question ID    : {qid}")
    print("Question text  :")
    print(qtext)
    print()

    # Show gold articles
    print("GOLD relevant articles:")
    if not gold_doc_ids:
        print("  (no gold articles for this question in train_data)")
    else:
        for d in gold_doc_ids:
            meta = retrieval.DOCID_TO_META[d]
            print(
                f"  - doc_id={d}, "
                f"law_id={meta['law_id']}, article_id={meta['article_id']}"
            )
    print()

    # 3) Use the precomputed embedding for this training question
    q_emb = retrieval.train_question_embeddings[q_index]

    # 4) Run BM25 + cosine (+ optional LogReg) pipeline WITH TIMER
    t0 = time.time()
    results = retrieval.retrieve_and_rerank_with_qemb(
        qtext,
        q_emb,
        top_k_lexical=top_k_lexical,
        top_k_final=top_k_final,
        alpha=alpha,
        logreg_model=logreg_model,
    )
    elapsed = time.time() - t0
    print(f"Retrieval done in {elapsed:.3f} seconds.\n")

    # 5) Print top-k with HIT/MISS vs GOLD
    print(f"Top {top_k_final} retrieved articles (BM25 + cosine):")
    for r in results:
        hit = "HIT" if r["doc_id"] in gold_doc_ids else "MISS"
        print(
            f"- [{hit}] doc_id={r['doc_id']}, "
            f"law_id={r['law_id']}, article_id={r['article_id']}, "
            f"bm25={r['bm25_score']:.4f}, "
            f"cos={r['embedding_score']:.4f}, "
            f"combined={r['combined_score']:.4f}"
        )
    print("==========================================")
    print()


if __name__ == "__main__":
    # you can change q_index to inspect different training questions
    test_single_question(
        q_index=10,
        top_k_lexical=200,
        top_k_final=10,
        alpha=0.6,
        logreg_model=None,  # or pass a trained LogisticRegression
    )
