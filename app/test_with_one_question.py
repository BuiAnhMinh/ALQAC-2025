"""
Sanity check for the new two-stage retrieval:
  semantic (pgvector) -> BM25-Okapi rerank (in-app) on Zalo amending articles.

Run (inside Docker app container with DB running):
    python -m app.test_with_one_question
"""

import time
from typing import Dict, Set, Tuple

from app import retrieval
from app.data_loader import load_zalo_questions

def fbeta_for_sets(
    gold_articles: Set[Tuple[str, str]],
    pred_articles: Set[Tuple[str, str]],
    beta: float = 2.0,
) -> float:
    if not gold_articles and not pred_articles:
        return 1.0

    tp = len(gold_articles & pred_articles)
    fp = len(pred_articles - gold_articles)
    fn = len(gold_articles - pred_articles)

    beta2 = beta * beta

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def test_single_question(
    q_index: int = 0,
    semantic_top_k: int = 200,
    lexical_top_k: int = 10,
    beta: float = 2.0,
):
    # 1) Pick a training question
    train_data = load_zalo_questions()
    q = train_data[q_index]
    qid = q.get("question_id", q_index)
    qtext = q["text"]

    print("==========================================")
    print(f"Question index : {q_index}")
    print(f"Question ID    : {qid}")
    print("Question text  :")
    print(qtext)
    print()

    # 2) Build GOLD article set: {(law_id, article_id), ...}
    gold_articles: Set[Tuple[str, str]] = {
        (a["law_id"], a["article_id"]) for a in q.get("relevant_articles", [])
    }
    print("GOLD relevant articles:")
    if not gold_articles:
        print("  (no gold articles for this question)")
    else:
        for (law_id, article_id) in gold_articles:
            print(f"  - law_id={law_id}, article_id={article_id}")
    print()

    # 3) Two-stage retrieval: semantic top_k -> BM25 top_k
    t0 = time.time()
    retrieved: Dict[str, list] = retrieval.semantic_then_lexical(
        question_text=qtext,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
    )
    elapsed = time.time() - t0
    semantic_candidates = retrieved["semantic_candidates"]
    lexical_results = retrieved["lexical_results"]

    print(f"Retrieval done in {elapsed:.3f} seconds.")
    print(f"Semantic candidates: {len(semantic_candidates)}")
    print(f"Lexical (BM25-Okapi) top-{lexical_top_k}: {len(lexical_results)}")
    print()

    # 4) Print top-k lexical results
    print(f"Top {lexical_top_k} BM25-Okapi reranked articles:")
    for r in lexical_results:
        print(
            f"id={r['id']}, law_id={r['law_id']}, article_id={r['article_id']}, "
            f"bm25={r['bm25_score']:.4f}"
        )

    # 5) Compute F-beta for lexical results
    pred_articles = {(r["law_id"], r["article_id"]) for r in lexical_results}
    f_score = fbeta_for_sets(gold_articles, pred_articles, beta=beta)
    print(f"\nF{beta:.1f} for BM25-Okapi top-{lexical_top_k} (this question): {f_score:.4f}")
    print("==========================================")
    print()


if __name__ == "__main__":
    # change q_index to inspect different training questions
    test_single_question(
        q_index=0,
        semantic_top_k=200,
        lexical_top_k=10,
        beta=2.0,
    )
