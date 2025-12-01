import json
from pathlib import Path

from app.retrieval import (
    train_data,
    test_data,
    train_question_embeddings,
    test_question_embeddings,
    macro_f2_bm25_topk,
    macro_f2_rerank,
    build_logreg_model,
    build_predictions_for_questions_with_scores,
)


def main():
    # 1) BM25 baselines
    bm25_f2_top1 = macro_f2_bm25_topk(k=1, beta=2.0, verbose=True)
    bm25_f2_top3 = macro_f2_bm25_topk(k=3, beta=2.0, verbose=True)
    bm25_f2_top5 = macro_f2_bm25_topk(k=5, beta=2.0, verbose=True)

    print("BM25 Macro-F2:")
    print("  top-1:", bm25_f2_top1)
    print("  top-3:", bm25_f2_top3)
    print("  top-5:", bm25_f2_top5)

    # 2) Train logistic regression reranker
    logreg_model = build_logreg_model(top_k_lexical=200)

    # 3) Evaluate reranker
    rerank_f2 = macro_f2_rerank(
        beta=2.0,
        top_k_lexical=200,
        top_k_final=3,
        alpha=0.6,
        logreg_model=logreg_model,
        verbose=True,
    )
    print("Rerank Macro-F2:", rerank_f2)

    # 4) Build predictions for private test set
    test_predictions_with_scores = build_predictions_for_questions_with_scores(
        questions=test_data,
        question_embeddings=test_question_embeddings,
        top_k_lexical=200,
        top_k_final=3,
        alpha=0.6,
        logreg_model=logreg_model,
    )

    out_path = Path("alqac25_test_predictions.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(test_predictions_with_scores, f, ensure_ascii=False, indent=2)

    print("Saved predictions with scores to", out_path)


if __name__ == "__main__":
    main()
