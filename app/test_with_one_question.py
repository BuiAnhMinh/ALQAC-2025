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
    beta: float = 2.0,
):
    # 1) Pick a training question
    train_data = retrieval.train_data  # already loaded in retrieval.py
    q = train_data[q_index]
    qid = q["question_id"]
    qtext = q["text"]

    print("==========================================")
    print(f"Question index : {q_index}")
    print(f"Question ID    : {qid}")
    print("Question text  :")
    print(qtext)
    print()

    # 2) Build GOLD article set: {(law_id, article_id), ...}
    gold_articles = {
        (a["law_id"], a["article_id"]) for a in q["relevant_articles"]
    }
    print("GOLD relevant articles:")
    if not gold_articles:
        print("  (no gold articles for this question)")
    else:
        for (law_id, article_id) in gold_articles:
            print(f"  - law_id={law_id}, article_id={article_id}")
    print()

    # 3) Use the precomputed embedding for this training question
    q_emb = retrieval.train_question_embeddings[q_index]

    # 4a) BM25-only retrieval (for comparison)
    bm25_results = retrieval.bm25_lexical_retrieve(
        qtext,
        top_k=top_k_final,
    )
    pred_articles_bm25 = {(r["law_id"], r["article_id"]) for r in bm25_results}
    f_bm25 = retrieval.fbeta_for_sets(
        gold_articles,
        pred_articles_bm25,
        beta=beta,
    )

    print(f"Top {top_k_final} BM25-only retrieved articles:")
    for r in bm25_results:
        print(
            f"doc_id={r['doc_id']}, "
            f"law_id={r['law_id']}, article_id={r['article_id']}, "
            f"bm25={r['bm25_score']:.4f}"
        )
    print(f"F{beta:.1f} for BM25-only (this question): {f_bm25:.4f}")
    print()

    # 4b) Run BM25 + cosine (+ optional LogReg) pipeline WITH TIMER
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

    # 5) Print top-k with scores
    print(f"Top {top_k_final} retrieved articles (BM25 + cosine):")
    for r in results:
        print(
            f"doc_id={r['doc_id']}, "
            f"law_id={r['law_id']}, article_id={r['article_id']}, "
            f"bm25={r['bm25_score']:.4f}, "
            f"cos={r['embedding_score']:.4f}, "
            f"combined={r['combined_score']:.4f}"
        )

    # 6) Compute F-beta for the reranked results
    pred_articles_rerank = {(r["law_id"], r["article_id"]) for r in results}
    f_rerank = retrieval.fbeta_for_sets(
        gold_articles,
        pred_articles_rerank,
        beta=beta,
    )

    print(f"\nF{beta:.1f} for BM25 + embeddings (this question): {f_rerank:.4f}")
    print("==========================================")
    print()


if __name__ == "__main__":
    # you can change q_index to inspect different training questions
    test_single_question(
        q_index=0,
        top_k_lexical=200,
        top_k_final=10,
        alpha=0.6,
        logreg_model=None,  # or pass a trained LogisticRegression/XGBoost
        beta=2.0,
    )
